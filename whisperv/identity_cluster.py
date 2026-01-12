import os
import cv2
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import os
_EMBEDDER_CACHE = {}

# Embedding backends: default to MagFace; allow override via IDENTITY_EMBEDDER env
def _build_embedder(device: str = "cuda", batch_size: int = 16):
    backend = os.environ.get("IDENTITY_EMBEDDER", "magface").strip().lower()
    if backend == "magface":
        try:
            from .embedders.magface_embedder import MagFaceEmbedder
        except Exception:
            from embedders.magface_embedder import MagFaceEmbedder
        backbone = os.environ.get("MAGFACE_BACKBONE", "iresnet100")
        key = ("magface", device, int(batch_size), backbone)
        if key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[key] = MagFaceEmbedder(device=device, batch_size=batch_size, backbone=backbone)
        return _EMBEDDER_CACHE[key]
    elif backend == "facenet":
        try:
            from .identity_verifier import IdentityVerifier
        except Exception:
            from identity_verifier import IdentityVerifier
        key = ("facenet", device, int(batch_size))
        if key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[key] = IdentityVerifier(device=device, batch_size=batch_size)
        return _EMBEDDER_CACHE[key]
    else:
        raise RuntimeError(f"Unsupported IDENTITY_EMBEDDER backend: {backend}")


def _frames_to_bbox_map(track: Dict) -> Dict[int, Tuple[float, float, float, float]]:
    frames = track["track"]["frame"]
    bboxes = track["track"]["bbox"]
    # frames/bboxes can be numpy arrays; ensure Python types
    frames_list = frames.tolist() if hasattr(frames, "tolist") else list(frames)
    bboxes_list = bboxes.tolist() if hasattr(bboxes, "tolist") else list(bboxes)
    return {int(f): tuple(map(float, bb)) for f, bb in zip(frames_list, bboxes_list)}

def _sample_indices(total: int, active_indices: Optional[List[int]], max_samples: int = 15, min_active: int = 3) -> List[int]:
    if total <= 0:
        return []
    if active_indices and len(active_indices) > 0:
        idx = sorted(set(int(min(max(0, i), total - 1)) for i in active_indices))
        if len(idx) >= int(min_active):
            # Augment with spread indices for stability
            spread = [0, total // 4, total // 2, (3 * total) // 4, max(0, total - 1)]
            idx = sorted(set(idx + spread))
            if len(idx) > max_samples:
                step = max(1, len(idx) // max_samples)
                idx = idx[::step][:max_samples]
            return idx
    # default spread
    return [0, total // 4, total // 2, (3 * total) // 4, max(0, total - 1)]

def _crop_face_bgr(image_bgr: np.ndarray, x: float, y: float, s: float, cs: float) -> Optional[np.ndarray]:
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        return None
    H, W = image_bgr.shape[:2]
    bsi = int(s * (1 + 2 * cs))
    pad_val = 110
    frame = np.pad(image_bgr, ((bsi,bsi),(bsi,bsi),(0,0)), mode='constant', constant_values=pad_val)
    my = y + bsi
    mx = x + bsi
    y1 = int(my - s)
    y2 = int(my + s * (1 + 2 * cs))
    x1 = int(mx - s * (1 + cs))
    x2 = int(mx + s * (1 + cs))
    if y2 <= y1 or x2 <= x1:
        return None
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (224,224))


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


@torch.no_grad()
def cluster_visual_identities(
    vidTracks: List[Dict],
    device: str = "cuda",
    batch_size: int = 16,
    face_sim_thresh: float = 0.60,
    must_iou_thresh: float = 0.70,
    cannot_iou_thresh: float = 0.10,
    min_overlap_frames: int = 3,
    must_center_thresh: float = 0.12,
    cannot_center_thresh: float = 0.40,
    cannot_sim_thresh: float = 0.35,
    overlap_force_sim: float = 0.72,
    overlap_force_iou: float = 0.35,
    overlap_force_center: float = 0.18,
    pairwise_link_thresh: Optional[float] = None,
    pairwise_overlap_thresh: Optional[float] = None,
    scores_list: Optional[List[List[float]]] = None,
    # ASD gating parameters (frame-level score threshold, min consecutive frames, min ratio over track)
    asd_score_thresh: float = 0.20,
    asd_min_consec: int = 8,
    asd_min_ratio: float = 0.10,
    save_avatars_path: Optional[str] = None,
    temporal_merge_thresh: Optional[float] = None,
    temporal_merge_max_gap: Optional[int] = None,
    embed_topk: int = 5,
    temporal_merge_long_thresh: Optional[float] = None,
    temporal_merge_long_gap: Optional[int] = None,
) -> List[Dict]:
    """Cluster face tracks into stable episode-level identities using only visual cues.

    - Embeddings: facenet via IdentityVerifier (averaged key-frames per cropFile)
    - Must-link: overlapping-in-time with high bbox IoU (same face duplicated), force union
    - Cannot-link: co-visible with low IoU (different persons), forbid merging
    - Greedy agglomerative merging by cosine similarity with threshold and constraints

    Returns a new list mirroring vidTracks but 'identity' set to a stable label: 'Person_1', ...
    """

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 1) Per-track embedding (single prototype per track)
    embedder = _build_embedder(device=device, batch_size=batch_size)
    embs: List[Optional[torch.Tensor]] = []  # (1,D) per track or None
    # Track-level best aligned face and its quality (mag)
    track_best: Dict[int, Dict[str, object]] = {}
    valid_idx: List[int] = []
    # Scheme A: skip tracks with no positive ASD frames when scores_list provided
    include_track = [True] * len(vidTracks)
    if scores_list is not None:
        include_track = []
        for i in range(len(vidTracks)):
            sc = scores_list[i] if i < len(scores_list) else None
            # If no ASD available, do not exclude the track; just skip gating
            if not (isinstance(sc, (list, tuple)) and len(sc) > 0):
                include_track.append(True)
                continue
            # Apply stricter speaking criteria:
            #  - frame-level threshold: score > asd_score_thresh
            #  - require BOTH: consecutive positives >= asd_min_consec AND ratio >= asd_min_ratio
            pos = [1 if (float(v) > asd_score_thresh) else 0 for v in sc]
            if not any(pos):
                include_track.append(False)
                continue
            # longest consecutive ones
            max_run = 0
            run = 0
            for b in pos:
                if b:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 0
            ratio = float(sum(pos)) / max(1, len(pos))
            inc = (max_run >= asd_min_consec) and (ratio >= asd_min_ratio)
            include_track.append(inc)
    try:
        embed_topk = int(embed_topk)
    except Exception:
        embed_topk = 0
    if embed_topk <= 0:
        embed_topk = 0
    for i, tr in enumerate(vidTracks):
        if not include_track[i]:
            embs.append(None)
            continue
        crop_file = tr.get("cropFile")
        emb = None
        best_img = None
        best_mag = None
        # Active indices from ASD gating
        active_idx = None
        if scores_list is not None and i < len(scores_list):
            sc = scores_list[i]
            if isinstance(sc, (list, tuple)) and len(sc) > 0:
                idx = [k for k, v in enumerate(sc) if float(v) > asd_score_thresh]
                if len(idx) > 0:
                    active_idx = idx
        # 1) If crop file exists, open video and compute aligned frames for quality + embedding
        if crop_file:
            avi_path = crop_file + ".avi"
            if os.path.isfile(avi_path):
                import cv2 as _cv
                cap = _cv.VideoCapture(avi_path)
                total = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
                if total > 0:
                    # select sample indices
                    positions = _sample_indices(total, active_idx, max_samples=15)
                    tensors = []
                    align_src = []
                    for pos in positions:
                        cap.set(_cv.CAP_PROP_POS_FRAMES, int(pos))
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        t = embedder._align_and_preprocess(frame)
                        if t is not None:
                            tensors.append(t)
                            align_src.append(t)  # store aligned tensor for thumbnail
                    cap.release()
                    if tensors:
                        batch = torch.stack(tensors, dim=0).to(embedder.device)
                        out = embedder.model(batch)
                        mags = torch.norm(out, p=2, dim=1)
                        out_n = F.normalize(out, p=2, dim=1)
                        # weighted average of top-k embeddings by quality
                        if embed_topk and mags.numel() > embed_topk:
                            vals, idx = torch.topk(mags, k=int(embed_topk), largest=True)
                            out_n = out_n[idx]
                            mags = vals
                        w = mags.view(-1, 1)
                        emb = F.normalize((out_n * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8), p=2, dim=1)
                        # best aligned image by mag
                        j = int(torch.argmax(mags).item())
                        best_mag = float(mags[j].item())
                        at = align_src[j].cpu().numpy()  # CHW float
                        best_img = (at.transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
        # 2) Otherwise, build frames from original video + proc_track and embed in-memory
        if emb is None:
            video_path = tr.get('video_path') or tr.get('videoFilePath')
            proc = tr.get('proc_track')
            track_obj = tr.get('track', {})
            frames = track_obj.get('frame')
            if (not video_path) or (proc is None) or (frames is None):
                embs.append(None)
                continue
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                embs.append(None)
                continue
            frames_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
            sel = _sample_indices(len(frames_list), active_idx, max_samples=15)
            frames_bgr = []
            cs = float(tr.get('cropScale', 0.40))
            for k in sel:
                fidx = int(frames_list[k])
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, img = cap.read()
                if not ret:
                    continue
                x = float(proc['x'][k]); y = float(proc['y'][k]); s = float(proc['s'][k])
                face = _crop_face_bgr(img, x, y, s, cs)
                if face is not None:
                    frames_bgr.append(face)
            cap.release()
            if frames_bgr:
                # Align & forward to get mags and embedding
                tensors = []
                for fb in frames_bgr:
                    t = embedder._align_and_preprocess(fb)
                    if t is not None:
                        tensors.append(t)
                if tensors:
                    batch = torch.stack(tensors, dim=0).to(embedder.device)
                    out = embedder.model(batch)
                    mags = torch.norm(out, p=2, dim=1)
                    out_n = F.normalize(out, p=2, dim=1)
                    # weighted average of top-k embeddings by quality
                    if embed_topk and mags.numel() > embed_topk:
                        vals, idx = torch.topk(mags, k=int(embed_topk), largest=True)
                        out_n = out_n[idx]
                        mags = vals
                    w = mags.view(-1, 1)
                    emb = F.normalize((out_n * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8), p=2, dim=1)
                    j = int(torch.argmax(mags).item())
                    best_mag = float(mags[j].item())
                    at = tensors[j].cpu().numpy()
                    best_img = (at.transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
        if emb is None:
            embs.append(None)
            continue
        # Save best for this track if available
        if isinstance(best_img, np.ndarray) and best_img.size > 0 and (best_mag is not None):
            track_best[i] = { 'img': best_img, 'mag': float(best_mag) }
        embs.append(emb)
        valid_idx.append(i)

    if not valid_idx:
        raise RuntimeError("No valid embeddings computed for any track; cannot cluster identities.")

    # Precompute pairwise embedding similarity cache
    sim_cache = {}
    if valid_idx and len(valid_idx) > 1:
        try:
            E = torch.cat([embs[i] for i in valid_idx], dim=0)
            E = F.normalize(E, p=2, dim=1)
            S = torch.matmul(E, E.t()).cpu().numpy()
            for ai, i in enumerate(valid_idx):
                for bi in range(ai + 1, len(valid_idx)):
                    j = valid_idx[bi]
                    sim_cache[(i, j)] = float(S[ai, bi])
        except Exception:
            sim_cache = {}

    def _sim_ij(i: int, j: int) -> Optional[float]:
        if (i, j) in sim_cache:
            return sim_cache[(i, j)]
        if (j, i) in sim_cache:
            return sim_cache[(j, i)]
        return None

    # 2) Build time overlap + IoU maps
    frame_maps = [_frames_to_bbox_map(tr) for tr in vidTracks]
    frames_sets = [set(m.keys()) for m in frame_maps]

    n = len(vidTracks)
    dsu = _DSU(n)
    cannot = set()  # set of frozenset({i,j})

    for i in range(n):
        if embs[i] is None or not include_track[i]:
            continue
        for j in range(i + 1, n):
            if embs[j] is None or not include_track[j]:
                continue
            # time-overlap frames
            overlap_frames = frames_sets[i].intersection(frames_sets[j])
            if len(overlap_frames) < min_overlap_frames:
                continue
            # mean IoU over overlapped frames
            ious = []
            dists = []
            fmap_i = frame_maps[i]
            fmap_j = frame_maps[j]
            for f in overlap_frames:
                iou = _iou(fmap_i[f], fmap_j[f])
                ious.append(iou)
                # center distance normalized by max box size
                bi = fmap_i[f]
                bj = fmap_j[f]
                cx1 = 0.5 * (bi[0] + bi[2]); cy1 = 0.5 * (bi[1] + bi[3])
                cx2 = 0.5 * (bj[0] + bj[2]); cy2 = 0.5 * (bj[1] + bj[3])
                w1 = max(1.0, float(bi[2] - bi[0])); h1 = max(1.0, float(bi[3] - bi[1]))
                w2 = max(1.0, float(bj[2] - bj[0])); h2 = max(1.0, float(bj[3] - bj[1]))
                denom = max(w1, h1, w2, h2, 1.0)
                d = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 / denom
                dists.append(float(d))
            if not ious:
                continue
            miou = float(np.mean(ious))
            mdist = float(np.mean(dists)) if dists else 1e9
            sim = _sim_ij(i, j)
            if miou >= must_iou_thresh or mdist <= float(must_center_thresh):
                dsu.union(i, j)
            elif sim is not None and sim >= float(overlap_force_sim) and (miou >= float(overlap_force_iou) or mdist <= float(overlap_force_center)):
                dsu.union(i, j)
            elif miou <= cannot_iou_thresh and mdist >= float(cannot_center_thresh):
                # CRITICAL FIX: If two tracks appear at SAME frame but different locations,
                # they MUST be different people - force cannot-link regardless of face similarity
                # This prevents merging two visually similar but distinct persons
                cannot.add(frozenset((i, j)))

    # 3) Initialize groups by must-link unions
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        if embs[i] is None or not include_track[i]:
            continue
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)
    group_ids = list(groups.keys())
    # 4) Group embeddings (use sum of unit embeddings to allow O(1) centroid updates)
    grp_embs: Dict[int, torch.Tensor] = {}
    grp_sums: Dict[int, torch.Tensor] = {}
    grp_counts: Dict[int, int] = {}
    for gid, members in groups.items():
        vecs = [embs[m] for m in members if embs[m] is not None]
        if not vecs:
            raise RuntimeError("Encountered a group without embeddings; aborting to avoid fake data.")
        V = torch.cat(vecs, dim=0)  # (k, D)
        V = F.normalize(V, p=2, dim=1)  # normalize each member embedding
        S = V.sum(dim=0, keepdim=True)  # sum of unit vectors
        grp_sums[gid] = S
        grp_counts[gid] = V.size(0)
        grp_embs[gid] = F.normalize(S.clone(), p=2, dim=1)  # centroid direction

    # Helper: can we merge two groups under cannot-link constraints?
    def can_merge(a_gid: int, b_gid: int) -> bool:
        A = groups[a_gid]
        B = groups[b_gid]
        for i in A:
            for j in B:
                if frozenset((i, j)) in cannot:
                    return False
        return True

    # Helper: can we merge two groups under cannot-link constraints?
    def can_merge(a_gid: int, b_gid: int) -> bool:
        A = groups[a_gid]
        B = groups[b_gid]
        for i in A:
            for j in B:
                if frozenset((i, j)) in cannot:
                    return False
        return True

    # 5) Greedy agglomerative merging with similarity threshold & constraints (vectorized sims)
    active = set(group_ids)
    while True:
        act_list = list(active)
        L = len(act_list)
        if L <= 1:
            break
        # Build centroid matrix [L, D]
        C = torch.cat([grp_embs[g] for g in act_list], dim=0)  # rows are unit centroids
        # Cosine similarity matrix via dot product
        S = torch.matmul(C, C.t())  # [L, L]
        # Mask diagonal
        idx = torch.arange(L)
        S[idx, idx] = -1e9

        merged = False
        while True:
            max_val = torch.max(S)
            best_sim = float(max_val.item()) if torch.is_tensor(max_val) else float(max_val)
            if best_sim < face_sim_thresh:
                break
            pos = torch.nonzero(S == max_val, as_tuple=False)
            if pos.numel() == 0:
                break
            a_i, b_i = int(pos[0,0].item()), int(pos[0,1].item())
            ga, gb = act_list[a_i], act_list[b_i]
            if can_merge(ga, gb):
                groups[ga].extend(groups[gb])
                grp_sums[ga] = grp_sums[ga] + grp_sums[gb]
                grp_counts[ga] = grp_counts[ga] + grp_counts[gb]
                grp_embs[ga] = F.normalize(grp_sums[ga], p=2, dim=1)
                active.discard(gb)
                del groups[gb]
                del grp_embs[gb]
                del grp_sums[gb]
                del grp_counts[gb]
                merged = True
                break
            else:
                S[a_i, b_i] = -1e9
                S[b_i, a_i] = -1e9
        if not merged:
            break

    # 5a) Pairwise single-linkage merge to catch centroid drift (max pair similarity)
    if pairwise_link_thresh is None:
        pairwise_link_thresh = float(face_sim_thresh)
    if pairwise_overlap_thresh is None:
        pairwise_overlap_thresh = max(float(face_sim_thresh) + 0.20, 0.70)
    try:
        pairwise_link_thresh = float(pairwise_link_thresh)
    except Exception:
        pairwise_link_thresh = float(face_sim_thresh)
    try:
        pairwise_overlap_thresh = float(pairwise_overlap_thresh)
    except Exception:
        pairwise_overlap_thresh = max(float(face_sim_thresh) + 0.20, 0.70)

    # Precompute per-track spans for overlap checks
    track_span = {}
    for idx in range(n):
        frames = vidTracks[idx].get("track", {}).get("frame")
        if frames is None:
            continue
        fr_list = frames.tolist() if hasattr(frames, "tolist") else list(frames)
        if not fr_list:
            continue
        track_span[idx] = (int(fr_list[0]), int(fr_list[-1]))

    def _track_overlap(a: int, b: int) -> bool:
        sa = track_span.get(a); sb = track_span.get(b)
        if not sa or not sb:
            return False
        return not (sa[1] < sb[0] or sb[1] < sa[0])

    def _pairwise_max_sim(gA: List[int], gB: List[int]) -> float:
        best = -1.0
        for i in gA:
            for j in gB:
                s = _sim_ij(i, j)
                if s is None:
                    continue
                if s > best:
                    best = s
        return best

    def _overlap_spatial_ok(gA: List[int], gB: List[int]) -> bool:
        # Allow if any overlapping pair is spatially close
        for i in gA:
            for j in gB:
                if not _track_overlap(i, j):
                    continue
                overlap_frames = frames_sets[i].intersection(frames_sets[j])
                if len(overlap_frames) < min_overlap_frames:
                    continue
                ious = []
                dists = []
                fmap_i = frame_maps[i]
                fmap_j = frame_maps[j]
                for f in overlap_frames:
                    bi = fmap_i.get(f); bj = fmap_j.get(f)
                    if bi is None or bj is None:
                        continue
                    ious.append(_iou(bi, bj))
                    cx1 = 0.5 * (bi[0] + bi[2]); cy1 = 0.5 * (bi[1] + bi[3])
                    cx2 = 0.5 * (bj[0] + bj[2]); cy2 = 0.5 * (bj[1] + bj[3])
                    w1 = max(1.0, float(bi[2] - bi[0])); h1 = max(1.0, float(bi[3] - bi[1]))
                    w2 = max(1.0, float(bj[2] - bj[0])); h2 = max(1.0, float(bj[3] - bj[1]))
                    denom = max(w1, h1, w2, h2, 1.0)
                    d = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 / denom
                    dists.append(float(d))
                if ious and dists:
                    if float(np.median(ious)) >= float(overlap_force_iou) or float(np.median(dists)) <= float(overlap_force_center):
                        return True
        return False

    while True:
        merged = False
        act_list = list(active)
        L = len(act_list)
        if L <= 1:
            break
        best_pair = None
        best_sim = -1.0
        for ii in range(L):
            ga = act_list[ii]
            for jj in range(ii + 1, L):
                gb = act_list[jj]
                if not can_merge(ga, gb):
                    continue
                sim = _pairwise_max_sim(groups[ga], groups[gb])
                if sim <= 0:
                    continue
                # If any track overlaps in time, require stricter similarity and spatial match
                overlaps = False
                for a in groups[ga]:
                    for b in groups[gb]:
                        if _track_overlap(a, b):
                            overlaps = True
                            break
                    if overlaps:
                        break
                if overlaps:
                    if sim < pairwise_overlap_thresh:
                        continue
                    if not _overlap_spatial_ok(groups[ga], groups[gb]):
                        continue
                else:
                    if sim < pairwise_link_thresh:
                        continue
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (ga, gb)
        if best_pair is None:
            break
        ga, gb = best_pair
        groups[ga].extend(groups[gb])
        grp_sums[ga] = grp_sums[ga] + grp_sums[gb]
        grp_counts[ga] = grp_counts[ga] + grp_counts[gb]
        grp_embs[ga] = F.normalize(grp_sums[ga], p=2, dim=1)
        active.discard(gb)
        del groups[gb]
        del grp_embs[gb]
        del grp_sums[gb]
        del grp_counts[gb]
        merged = True

    # 5b) Optional temporal-exclusive merge with relaxed threshold
    if temporal_merge_thresh is not None:
        try:
            temporal_merge_thresh = float(temporal_merge_thresh)
        except Exception:
            temporal_merge_thresh = None
    # Only run if relaxed threshold is explicitly set and lower than main threshold
    if temporal_merge_thresh is not None and temporal_merge_thresh < face_sim_thresh and len(active) > 1:
        # Build group time spans (min/max frame index)
        group_span: Dict[int, Tuple[int, int]] = {}
        for gid, members in groups.items():
            min_f = None
            max_f = None
            for m in members:
                frames = vidTracks[m].get("track", {}).get("frame")
                if frames is None:
                    continue
                fr_list = frames.tolist() if hasattr(frames, "tolist") else list(frames)
                if not fr_list:
                    continue
                f0 = int(fr_list[0])
                f1 = int(fr_list[-1])
                if min_f is None or f0 < min_f:
                    min_f = f0
                if max_f is None or f1 > max_f:
                    max_f = f1
            if min_f is None or max_f is None:
                # invalid span; skip from temporal merging
                continue
            group_span[gid] = (min_f, max_f)

        def _non_overlap_gap(a_span: Tuple[int, int], b_span: Tuple[int, int]) -> Optional[int]:
            if a_span[1] < b_span[0]:
                return b_span[0] - a_span[1]
            if b_span[1] < a_span[0]:
                return a_span[0] - b_span[1]
            return None

        while True:
            act_list = [g for g in active if g in group_span]
            L = len(act_list)
            if L <= 1:
                break
            C = torch.cat([grp_embs[g] for g in act_list], dim=0)
            S = torch.matmul(C, C.t())
            idx = torch.arange(L)
            S[idx, idx] = -1e9

            best_pair = None
            best_sim = -1.0
            for i in range(L):
                for j in range(i + 1, L):
                    sim = float(S[i, j].item())
                    ga = act_list[i]
                    gb = act_list[j]
                    if not can_merge(ga, gb):
                        continue
                    gap = _non_overlap_gap(group_span[ga], group_span[gb])
                    if gap is None:
                        continue
                    # Dynamic threshold by gap length
                    thr = temporal_merge_thresh
                    if temporal_merge_long_thresh is not None and temporal_merge_max_gap is not None:
                        if gap > int(temporal_merge_max_gap):
                            if temporal_merge_long_gap is not None and gap > int(temporal_merge_long_gap):
                                continue
                            thr = float(temporal_merge_long_thresh)
                    if sim < float(thr):
                        continue
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (ga, gb)
            if best_pair is None:
                break
            ga, gb = best_pair
            groups[ga].extend(groups[gb])
            grp_sums[ga] = grp_sums[ga] + grp_sums[gb]
            grp_counts[ga] = grp_counts[ga] + grp_counts[gb]
            grp_embs[ga] = F.normalize(grp_sums[ga], p=2, dim=1)
            active.discard(gb)
            del groups[gb]
            del grp_embs[gb]
            del grp_sums[gb]
            del grp_counts[gb]
            # update span
            span_a = group_span.get(ga)
            span_b = group_span.get(gb)
            if span_a and span_b:
                group_span[ga] = (min(span_a[0], span_b[0]), max(span_a[1], span_b[1]))
            if gb in group_span:
                del group_span[gb]

    # 6) Assign stable identity labels
    stable_ids = {}
    for idx, gid in enumerate(sorted(active)):
        # Use Person_* naming in place of prior VID_*
        label = f"Person_{idx+1}"
        for m in groups[gid]:
            stable_ids[m] = label

    # 7) Build output annotated tracks
    annotated = []
    for i, tr in enumerate(vidTracks):
        t2 = dict(tr)
        ident = stable_ids.get(i)
        if ident is not None:
            t2["identity"] = ident
        else:
            t2["identity"] = None
        annotated.append(t2)

    # 8) Build per-identity best avatar (reuse MagFace quality) and optionally persist
    if save_avatars_path:
        id_best = {}
        for gid in active:
            members = groups[gid]
            # find member with highest best_mag
            best = None
            best_m = -1.0
            for m in members:
                rec = track_best.get(m)
                if not rec:
                    continue
                if float(rec['mag']) > best_m:
                    best_m = float(rec['mag'])
                    best = rec['img']
            if best is None:
                continue
            label = None
            # find label for this group
            for k,v in stable_ids.items():
                if k in members:
                    label = v; break
            if isinstance(label, str):
                id_best[label] = best
        if id_best:
            try:
                import pickle
                with open(save_avatars_path, 'wb') as f:
                    pickle.dump(id_best, f)
            except Exception:
                pass

    return annotated
