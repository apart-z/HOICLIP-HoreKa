#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
my_dataset_eval.py

- Triplet HOI mAP (VOC07 11-point): requires subject&object bbox IoU match + action match.
- Outputs:
    * Triplet Full / Rare / Non-Rare mAP (rare split from TRAIN frequency like HICO/RLIPv2)
    * Pair(BBox) mAP: bbox-only at human-object pair level (ignore action), STRICT category match
    * Action mAP: action-only (key=action), bbox matching WITHOUT category constraint
    * (Optional) Group HOI metrics for higher-order interactions:
      - strict group AP (full-match)
      - soft group precision/recall/F1
      - role-aware precision/recall/F1
      - group Recall@K
      - cardinality accuracy

This file is intended to be imported and used during RLIPv2 training/eval.
"""

import os
import json
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import torch
from datasets.myds_meta import load_myds_meta

# Cache to avoid re-reading train.json every eval during training
_TRAIN_FREQ_CACHE: Dict[Tuple[Any, ...], Tuple[Set[Tuple[str, str, Any]], Set[Tuple[str, str, Any]]]] = {}

# Fallback metadata paths (user-provided deployment paths).
_DEFAULT_VERB_ROLES_PATH = "/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds/metadata/verb_roles.txt"
_DEFAULT_OBJECTS_PATH = "/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds/metadata/objects.txt"


def _load_lines(path: str) -> List[str]:
    if not path or (not os.path.isfile(path)):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
    return out


Triplet = Tuple[str, str, Any]  # (sub_cat, obj_cat, action)
PairKey = Tuple[str, str]       # (sub_cat, obj_cat)
ActionKey = Any                 # normalized action (int or str)


def parse_action_token(token):
    """Parse action token like 'verb:role' -> (verb, role)."""
    t = str(token).strip().lower()
    if ":" not in t:
        return t, None
    v, r = t.split(":", 1)
    return v.strip(), r.strip() if r is not None else None


GroupMember = Dict[str, Any]    # {"subject_id","object_id","verb","role"}
GroupDict = Dict[str, Any]      # {"group_id","members","score",...}


def _is_main_process() -> bool:
    """Safe main-process check for both single-GPU and torch.distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _label_to_int(lbl: Any) -> int:
    """Convert various label shapes (tensor/ndarray/list/scalar) to int safely."""
    arr = np.asarray(lbl)
    if arr.size == 1:
        return int(arr.item())
    if arr.ndim == 1:
        if arr.shape[0] > 1:
            return int(arr[-1])  # e.g. [is_person, obj_cls] -> take last
        return int(arr[0])
    return int(arr.flat[0])


def _as_xyxy(box: Any, bbox_format: str) -> np.ndarray:
    """Convert a bbox to xyxy float32."""
    b = np.asarray(box, dtype=np.float32).reshape(-1)
    if b.size != 4:
        raise ValueError(f"Invalid bbox length: {b}")
    if bbox_format == "xyxy":
        return b
    if bbox_format == "xywh":
        x, y, w, h = b
        return np.asarray([x, y, x + w, y + h], dtype=np.float32)
    raise ValueError(f"Unknown bbox_format={bbox_format}")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 0 else 0.0


class GroupHOIMetrics:
    """
    Higher-order group HOI evaluator.
    GT group:
      G = {(subject_id, verb, object_id, role)}
    Pred groups can be:
      1) explicit: pred["group_prediction"]
      2) built from pairwise HOIs by connected components.
    """

    def __init__(self, args, overlap_iou: float, norm_cat_fn):
        self.overlap_iou = overlap_iou
        self._norm_cat = norm_cat_fn
        # Group metrics are significantly more expensive than triplet/pair/action mAP.
        # Keep disabled by default unless explicitly requested.
        self.enabled = bool(getattr(args, "enable_group_eval", False)) if args is not None else False
        self.soft_match_thresh = float(getattr(args, "group_soft_match_thresh", 0.5)) if args is not None else 0.5
        self.duplicate_iou = float(getattr(args, "group_duplicate_iou", 0.85)) if args is not None else 0.85
        self.group_build_mode = str(getattr(args, "group_build_mode", "heuristic_graph")) if args is not None else "heuristic_graph"
        self.recall_ks = list(getattr(args, "group_recall_ks", [20, 50, 100])) if args is not None else [20, 50, 100]
        self.max_groups_per_image = int(getattr(args, "group_max_per_image", 100)) if args is not None else 100
        self.verbose = bool(getattr(args, "group_eval_verbose", False)) if args is not None else False
        self.group_edge_score_thresh = float(getattr(args, "group_edge_score_thresh", 0.2)) if args is not None else 0.2
        self.group_pred_score_thr = float(getattr(args, "group_pred_score_thr", 0.05)) if args is not None else 0.05
        self.group_obj_cluster_iou = float(getattr(args, "group_obj_cluster_iou", 0.7)) if args is not None else 0.7
        self.group_dump_pr_curve = bool(getattr(args, "group_dump_pr_curve", False)) if args is not None else False
        self.group_log_per_image = bool(getattr(args, "group_log_per_image", False)) if args is not None else False
        self.group_metric_size_gamma = float(getattr(args, "group_metric_size_gamma", 0.5)) if args is not None else 0.5
        self.group_metric_score_pow = float(getattr(args, "group_metric_score_pow", 1.0)) if args is not None else 1.0
        # Optional lenient controls (defaults preserve previous strict behavior)
        self.group_strict_match_thresh = float(getattr(args, "group_strict_match_thresh", 1.0)) if args is not None else 1.0
        self.group_strict_role_aware = bool(getattr(args, "group_strict_role_aware", True)) if args is not None else True
        self.group_member_verb_required = bool(getattr(args, "group_member_verb_required", True)) if args is not None else True

    @staticmethod
    def _box_iou_xyxy(b1: np.ndarray, b2: np.ndarray) -> float:
        x1_1, y1_1, x2_1, y2_1 = b1
        x1_2, y1_2, x2_2, y2_2 = b2
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        a1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
        a2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _entity_match_ok(self, gt_ent: Dict[str, Any], pred_ent: Dict[str, Any]) -> bool:
        if self._norm_cat(gt_ent["category"]) != self._norm_cat(pred_ent["category"]):
            return False
        return self._box_iou_xyxy(np.asarray(gt_ent["bbox"], dtype=np.float32), np.asarray(pred_ent["bbox"], dtype=np.float32)) >= self.overlap_iou

    @staticmethod
    def _norm_action(a: Any) -> Any:
        if isinstance(a, str):
            a = a.strip()
        try:
            return int(a)
        except Exception:
            return a

    def build_gt_groups(self, gt_img: Dict[str, Any]) -> List[GroupDict]:
        groups = []
        def _synthesize_from_hois() -> List[GroupDict]:
            out = []
            hois = gt_img.get("hoi_annotation", []) or []
            if not hois:
                return out
            nodes = []
            for h in hois:
                try:
                    nodes.append((int(h["subject_id"]), int(h["object_id"]), self._norm_action(h["action"])))
                except Exception:
                    continue
            if not nodes:
                return out
            p = list(range(len(nodes)))
            def _fd(x):
                while p[x] != x:
                    p[x] = p[p[x]]
                    x = p[x]
                return x
            def _un(a, b):
                ra, rb = _fd(a), _fd(b)
                if ra != rb:
                    p[rb] = ra
            for i in range(len(nodes)):
                si, oi, _ = nodes[i]
                for j in range(i + 1, len(nodes)):
                    sj, oj, _ = nodes[j]
                    if (si == sj) or (oi == oj):
                        _un(i, j)
            comps: Dict[int, List[int]] = {}
            for i in range(len(nodes)):
                comps.setdefault(_fd(i), []).append(i)
            gid = 0
            for mem in comps.values():
                if len(mem) < 2:
                    continue
                mlist = []
                for mi in mem:
                    s, o, v = nodes[mi]
                    mlist.append({"subject_id": s, "object_id": o, "verb": v, "role": "target"})
                out.append({"group_id": f"auto_{gid}", "members": mlist, "score": 1.0})
                gid += 1
            return out
        raw_groups = (
            gt_img.get("group_interactions")
            or gt_img.get("group_annotations")
            or gt_img.get("groups")
            or gt_img.get("group_instances")
            or []
        )
        entities = gt_img.get("entities", [])
        ent_id_to_idx = {e.get("entity_id"): i for i, e in enumerate(entities)}
        # Fallback: synthesize GT groups from HOIs when explicit group annotations
        # are unavailable, reusing training-time group builder logic.
        if not raw_groups:
            syn = _synthesize_from_hois()
            if syn:
                return syn
        # 兼容 string/int id 混用（例如 entity_id=1，但成员里写成"1"）
        ent_id_to_idx.update({str(k): v for k, v in list(ent_id_to_idx.items()) if k is not None})

        parse_stats = defaultdict(int)

        def _iter_member_like_items(group_item: Dict[str, Any]):
            if not isinstance(group_item, dict):
                return []
            if isinstance(group_item.get("members"), list):
                return group_item.get("members", [])
            # 兼容“扁平 group”：group 本身就是一个成员
            if any(k in group_item for k in ("subject_id", "subj_id", "human_id")) and \
               any(k in group_item for k in ("object_id", "obj_id", "target_id")):
                return [group_item]
            # 兼容不同命名
            for k in ("interactions", "items", "edges", "relations", "triplets", "hois"):
                if isinstance(group_item.get(k), list):
                    return group_item.get(k, [])
            return []

        for i, g in enumerate(raw_groups):
            members = []
            raw_member_items = _iter_member_like_items(g)
            if not raw_member_items:
                parse_stats["group_without_member_list"] += 1
            for m in raw_member_items:
                sid_raw = m.get("subject_id", m.get("subj_id", m.get("human_id")))
                oid_raw = m.get("object_id", m.get("obj_id", m.get("target_id")))
                sid = ent_id_to_idx.get(sid_raw, sid_raw)
                oid = ent_id_to_idx.get(oid_raw, oid_raw)
                try:
                    sid = int(sid)
                    oid = int(oid)
                except Exception:
                    parse_stats["member_bad_subject_or_object_id"] += 1
                    continue
                role_default = m.get("role", "target")
                raw_verbs = []
                if "verb" in m:
                    raw_verbs.append(m["verb"])
                if "action" in m:
                    raw_verbs.append(m["action"])
                if "action_id" in m:
                    raw_verbs.append(m["action_id"])
                # 兼容更常见的多标签字段
                raw_verbs.extend(list(m.get("action_tokens", [])))
                raw_verbs.extend(list(m.get("verbs", [])))
                raw_verbs.extend(list(m.get("actions", [])))
                raw_verbs.extend(list(m.get("verb_ids", [])))
                raw_verbs.extend(list(m.get("action_ids", [])))

                if not raw_verbs:
                    parse_stats["member_without_action_token"] += 1

                for raw_verb in raw_verbs:
                    if raw_verb is None:
                        parse_stats["member_action_is_none"] += 1
                        continue
                    role = role_default
                    verb = raw_verb
                    if isinstance(raw_verb, str) and ":" in raw_verb:
                        verb_part, role_part = raw_verb.split(":", 1)
                        verb = verb_part
                        if "role" not in m:
                            role = role_part
                    members.append({
                        "subject_id": sid,
                        "object_id": oid,
                        "verb": verb,
                        "role": role,
                    })
            if members:
                groups.append({"group_id": g.get("group_id", f"gt_{i}"), "members": members, "score": 1.0})
            else:
                parse_stats["group_no_valid_members"] += 1
        if _is_main_process() and raw_groups and not groups:
            print(
                "[GroupHOIMetrics][GT parse debug] all raw groups filtered out. "
                f"num_raw_groups={len(raw_groups)}, parse_stats={dict(parse_stats)}"
            )
        # 如果数据集中没有显式 group 标注，则从 HOI 标注自动推断 group
        # 规则：把 (subject_id, object_id, action) 看作图上的边，按连通分量聚合为 group。
        if not groups:
            hoi_ann = gt_img.get("hoi_annotation", []) or []
            inferred = self._infer_gt_groups_from_hoi(hoi_ann)
            if inferred and self.verbose and _is_main_process():
                print(f"[GroupHOIMetrics] Info: inferred {len(inferred)} GT groups from hoi_annotation fallback.")
            groups = inferred
        if not groups:
            syn = _synthesize_from_hois()
            if syn:
                return syn
        return groups

    @staticmethod
    def _infer_gt_groups_from_hoi(hoi_ann: List[Dict[str, Any]]) -> List[GroupDict]:
        if not hoi_ann:
            return []
        adj = defaultdict(set)
        edge_items = []
        for h in hoi_ann:
            try:
                s = int(h.get("subject_id"))
                o = int(h.get("object_id"))
            except Exception:
                continue
            a = h.get("action")
            role = h.get("role", "target")
            adj[s].add(o)
            adj[o].add(s)
            edge_items.append((s, o, a, role))
        if not edge_items:
            return []

        visited = set()
        groups = []
        for n in list(adj.keys()):
            if n in visited:
                continue
            stack = [n]
            comp_nodes = set([n])
            visited.add(n)
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        comp_nodes.add(nxt)
                        stack.append(nxt)
            members = []
            for s, o, a, role in edge_items:
                if s in comp_nodes and o in comp_nodes:
                    members.append({
                        "subject_id": s,
                        "object_id": o,
                        "verb": a,
                        "role": role,
                    })
            if members:
                groups.append({
                    "group_id": f"gt_infer_{len(groups)}",
                    "members": members,
                    "score": 1.0,
                    "inferred": True,
                })
        return groups

    def _build_pred_groups_heuristic_graph(self, pred_hois: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]]) -> List[GroupDict]:
        if not pred_hois:
            self._last_graph_stats = {"num_edges_in": 0, "num_edges_kept": 0, "num_components": 0, "avg_component_size": 0.0}
            return []
        filtered_hois = [h for h in pred_hois if float(h.get("score", 0.0)) >= self.group_edge_score_thresh]
        if not filtered_hois:
            self._last_graph_stats = {
                "num_edges_in": len(pred_hois),
                "num_edges_kept": 0,
                "num_components": 0,
                "avg_component_size": 0.0,
            }
            return []
        edge_list = []
        for i, h in enumerate(filtered_hois):
            edge_list.append((int(h["subject_id"]), int(h["object_id"]), i))
        adj = defaultdict(list)
        for u, v, ei in edge_list:
            adj[u].append((v, ei))
            adj[v].append((u, ei))
        visited = set()
        groups = []
        for node in list(adj.keys()):
            if node in visited:
                continue
            stack = [node]
            comp_nodes = set()
            comp_edge_ids = set()
            visited.add(node)
            while stack:
                cur = stack.pop()
                comp_nodes.add(cur)
                for nxt, ei in adj[cur]:
                    comp_edge_ids.add(ei)
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            members = []
            scores = []
            for ei in comp_edge_ids:
                h = filtered_hois[ei]
                members.append({
                    "subject_id": int(h["subject_id"]),
                    "object_id": int(h["object_id"]),
                    "verb": h["action"],
                    "role": h.get("role", "target"),
                })
                scores.append(float(h.get("score", 0.0)))
            if not members:
                continue
            # geometric mean controls one very-low-quality member
            s = float(math.exp(np.mean(np.log(np.maximum(np.asarray(scores, dtype=np.float32), 1e-8)))))
            groups.append({"group_id": f"pred_cc_{len(groups)}", "members": members, "score": s, "num_nodes": len(comp_nodes)})

        # 为避免“连通分量过大导致 strict match 极低”，补充单边 group 候选。
        # 这对 GT 中大量 size=1/2 的 group 有明显帮助，也能让 Recall@K 真正随 K 增长。
        singleton_groups = []
        singleton_best = {}
        for h in filtered_hois:
            key = (
                int(h["subject_id"]),
                int(h["object_id"]),
                self._norm_action(h["action"]),
                str(h.get("role", "target")),
            )
            sc = float(h.get("score", 0.0))
            prev = singleton_best.get(key)
            if prev is None or sc > prev:
                singleton_best[key] = sc
        for (s_id, o_id, act, role), sc in singleton_best.items():
            singleton_groups.append({
                "group_id": f"pred_edge_{len(singleton_groups)}",
                "members": [{
                    "subject_id": s_id,
                    "object_id": o_id,
                    "verb": act,
                    "role": role,
                }],
                "score": sc,
                "num_nodes": 2,
                "singleton": True,
            })

        groups.extend(singleton_groups)
        comp_sizes = [len(g.get("members", [])) for g in groups]
        self._last_graph_stats = {
            "num_edges_in": len(pred_hois),
            "num_edges_kept": len(filtered_hois),
            "num_components": len(groups),
            "avg_component_size": float(np.mean(comp_sizes)) if comp_sizes else 0.0,
        }
        return groups

    def build_pred_groups(self, pred_img: Dict[str, Any]) -> List[GroupDict]:
        if self.group_build_mode == "explicit" and "group_prediction" in pred_img:
            groups = []
            hois_for_group = pred_img.get("hoi_prediction_for_group", pred_img.get("hoi_prediction", []))
            for i, g in enumerate(pred_img.get("group_prediction", [])):
                members = []
                # Format A: explicit members with subject/object/verb.
                for m in g.get("members", []):
                    try:
                        members.append({
                            "subject_id": int(m["subject_id"]),
                            "object_id": int(m["object_id"]),
                            "verb": m["verb"],
                            "role": m.get("role", "target"),
                        })
                    except Exception:
                        continue
                # Format B: compact member index list from group_hoi_utils.build_pred_groups
                # e.g. {"member_pair_indices":[...], "verb_id":..., "score":...}
                if not members and isinstance(g.get("member_pair_indices"), list):
                    for idx in g.get("member_pair_indices", []):
                        try:
                            h = hois_for_group[int(idx)]
                            members.append({
                                "subject_id": int(h["subject_id"]),
                                "object_id": int(h["object_id"]),
                                "verb": h.get("action", g.get("verb_id", -1)),
                                "role": h.get("role", "target"),
                            })
                        except Exception:
                            continue
                if members:
                    groups.append({
                        "group_id": g.get("group_id", f"pred_{i}"),
                        "members": members,
                        "score": float(g.get("score", 0.0)),
                    })
            # Debug + safety fallback: if explicit payload exists but cannot be decoded
            # into members, fall back to heuristic graph builder.
            if not groups:
                if _is_main_process() and self.verbose:
                    print("[GroupHOIMetrics] explicit group payload decoded to 0 groups, fallback to heuristic_graph.")
                return self._build_pred_groups_heuristic_graph(
                    pred_img.get("hoi_prediction_for_group", pred_img.get("hoi_prediction", [])),
                    pred_img.get("predictions", []),
                )
            return groups
        return self._build_pred_groups_heuristic_graph(
            pred_img.get("hoi_prediction_for_group", pred_img.get("hoi_prediction", [])),
            pred_img.get("predictions", []),
        )

    def _member_soft_match(self, gm: GroupMember, pm: GroupMember, gt_boxes, pred_boxes, role_aware: bool) -> float:
        if self.group_member_verb_required and (self._norm_action(gm["verb"]) != self._norm_action(pm["verb"])):
            return 0.0
        if role_aware and str(gm.get("role", "target")) != str(pm.get("role", "target")):
            return 0.0
        gs, go = int(gm["subject_id"]), int(gm["object_id"])
        ps, po = int(pm["subject_id"]), int(pm["object_id"])
        if gs >= len(gt_boxes) or go >= len(gt_boxes) or ps >= len(pred_boxes) or po >= len(pred_boxes):
            return 0.0
        sub_ok = self._entity_match_ok(gt_boxes[gs], pred_boxes[ps])
        obj_ok = self._entity_match_ok(gt_boxes[go], pred_boxes[po])
        if not (sub_ok and obj_ok):
            return 0.0
        return 1.0

    def _group_soft_score(self, gt_g: GroupDict, pred_g: GroupDict, gt_boxes, pred_boxes, role_aware: bool) -> float:
        gm = gt_g["members"]
        pm = pred_g["members"]
        if not gm or not pm:
            return 0.0
        score_mat = np.zeros((len(gm), len(pm)), dtype=np.float32)
        for i, gmi in enumerate(gm):
            for j, pmj in enumerate(pm):
                score_mat[i, j] = self._member_soft_match(gmi, pmj, gt_boxes, pred_boxes, role_aware)
        # bipartite matching by greedy over binary scores (sufficient for 0/1 matrix)
        pairs = [(float(score_mat[i, j]), i, j) for i in range(score_mat.shape[0]) for j in range(score_mat.shape[1])]
        pairs.sort(key=lambda x: x[0], reverse=True)
        used_i, used_j = set(), set()
        matched = 0.0
        for s, i, j in pairs:
            if s <= 0:
                break
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            matched += s
        union = len(gm) + len(pm) - matched
        return _safe_div(matched, union)

    def _group_exact_match(self, gt_g, pred_g, gt_boxes, pred_boxes) -> bool:
        if len(gt_g["members"]) != len(pred_g["members"]):
            return False
        return self._group_soft_score(
            gt_g, pred_g, gt_boxes, pred_boxes, role_aware=self.group_strict_role_aware
        ) >= self.group_strict_match_thresh

    def _group_weight(self, g: GroupDict) -> float:
        size = max(len(g.get("members", [])), 1)
        return float(size ** self.group_metric_size_gamma)

    def _image_level_match(self, gt_groups, pred_groups, gt_boxes, pred_boxes, strict: bool, role_aware: bool):
        if not gt_groups or not pred_groups:
            return [], set(), set()
        score_mat = np.zeros((len(gt_groups), len(pred_groups)), dtype=np.float32)
        for i, g in enumerate(gt_groups):
            for j, p in enumerate(pred_groups):
                if strict:
                    score_mat[i, j] = 1.0 if self._group_exact_match(g, p, gt_boxes, pred_boxes) else 0.0
                else:
                    score_mat[i, j] = self._group_soft_score(g, p, gt_boxes, pred_boxes, role_aware=role_aware)
        pairs = [(float(score_mat[i, j]), i, j) for i in range(score_mat.shape[0]) for j in range(score_mat.shape[1])]
        pairs.sort(key=lambda x: x[0], reverse=True)
        assigned, used_gt, used_pred = [], set(), set()
        for s, i, j in pairs:
            if i in used_gt or j in used_pred:
                continue
            if strict and s < 1.0:
                continue
            if (not strict) and s < self.soft_match_thresh:
                continue
            assigned.append((i, j, s))
            used_gt.add(i)
            used_pred.add(j)
        return assigned, used_gt, used_pred

    def evaluate(self, preds: List[Dict[str, Any]], gts: List[Dict[str, Any]]) -> Dict[str, float]:
        if not self.enabled:
            return {}
        wm_scores, wm_tp_w, wm_fp_w = [], [], []
        sum_gt_weight = 0.0
        total_gt_groups = 0
        total_pred_groups = 0
        total_gt_hois_seen = 0
        total_raw_group_items = 0
        graph_stats_acc = defaultdict(float)
        graph_stats_cnt = 0
        per_image_debug = []

        for pred_img, gt_img in zip(preds, gts):
            gt_boxes = gt_img.get("annotations", [])
            pred_boxes = pred_img.get("predictions", [])
            raw_groups = (
                gt_img.get("group_interactions")
                or gt_img.get("group_annotations")
                or gt_img.get("groups")
                or gt_img.get("group_instances")
                or []
            )
            total_raw_group_items += len(raw_groups)
            gt_groups = self.build_gt_groups(gt_img)
            pred_groups = self.build_pred_groups(pred_img)
            total_gt_hois_seen += int(len(gt_img.get("hoi_annotation", []) or []))
            raw_pred_cnt = len(pred_groups)
            total_pred_groups += raw_pred_cnt
            if isinstance(getattr(self, "_last_graph_stats", None), dict):
                for k, v in self._last_graph_stats.items():
                    graph_stats_acc[k] += float(v)
                graph_stats_cnt += 1
            pred_groups = sorted(pred_groups, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:self.max_groups_per_image]
            kept_pred_cnt = len(pred_groups)

            total_gt_groups += len(gt_groups)
            gt_weights = [self._group_weight(g) for g in gt_groups]
            sum_gt_weight += float(sum(gt_weights))
            if not gt_groups and not pred_groups:
                continue

            # role-aware soft one-to-one matching, gives partial credit to near-correct group detections
            assigned_soft, _, _ = self._image_level_match(
                gt_groups, pred_groups, gt_boxes, pred_boxes, strict=False, role_aware=True
            )
            kept_scores = [float(pg.get("score", 0.0)) for pg in pred_groups]
            per_image_debug.append({
                "image_id": gt_img.get("image_id", pred_img.get("image_id", "unknown")),
                "gt_groups": len(gt_groups),
                "pred_raw": raw_pred_cnt,
                "pred_kept": kept_pred_cnt,
                "pred_matched": len(assigned_soft),
                "avg_score_kept": float(np.mean(kept_scores)) if kept_scores else 0.0,
            })
            best_by_pred = {pj: (gi, s) for gi, pj, s in assigned_soft}
            for j, pg in enumerate(pred_groups):
                base_score = max(float(pg.get("score", 0.0)), 0.0)
                wm_scores.append(float(base_score ** self.group_metric_score_pow))
                pred_w = self._group_weight(pg)
                if j in best_by_pred:
                    gi, soft_s = best_by_pred[j]
                    tp_w = float(soft_s) * gt_weights[gi]
                    fp_w = max(pred_w - tp_w, 0.0)
                else:
                    tp_w = 0.0
                    fp_w = pred_w
                wm_tp_w.append(tp_w)
                wm_fp_w.append(fp_w)

        group_wmap = 0.0
        group_mmr = 0.0
        if wm_scores and sum_gt_weight > 0:
            order = np.argsort(-np.asarray(wm_scores, dtype=np.float32))
            tp = np.asarray(wm_tp_w, dtype=np.float32)[order]
            fp = np.asarray(wm_fp_w, dtype=np.float32)[order]
            tp_c = np.cumsum(tp)
            fp_c = np.cumsum(fp)
            rec = tp_c / float(sum_gt_weight)
            prec = tp_c / np.maximum(tp_c + fp_c, 1e-8)
            for t in np.arange(0.0, 1.1, 0.1):
                p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
                group_wmap += float(p) / 11.0
            group_mmr = float(np.max(rec)) if rec.size > 0 else 0.0
            if self.group_dump_pr_curve and _is_main_process():
                points = []
                for t in np.arange(0.0, 1.1, 0.1):
                    p = float(np.max(prec[rec >= t])) if np.any(rec >= t) else 0.0
                    points.append(f"R@{t:.1f}:P={p:.3f}")
                print("[GroupHOIMetrics][PRCurve] " + ", ".join(points))

        out = {
            "group_wmAP": float(group_wmap),
            "group_mean_max_recall": float(group_mmr),
            "group_total_gt_groups": float(total_gt_groups),
            "group_total_pred_groups": float(total_pred_groups),
            # 兼容日志检索习惯：同时暴露 mAP / mean max recall 命名
            "group_mAP": float(group_wmap),
            "group_mean max recall": float(group_mmr),
        }
        if _is_main_process():
            print(
                "[GroupHOIMetrics][Counts] "
                f"total_gt_groups={total_gt_groups}, total_pred_groups={total_pred_groups}, "
                f"sum_gt_weight={sum_gt_weight:.4f}, total_gt_hois_seen={total_gt_hois_seen}"
            )
            if self.group_log_per_image and per_image_debug:
                for it in per_image_debug:
                    print(
                        "[GroupHOIMetrics][PerImage] "
                        f"image_id={it['image_id']} gt={it['gt_groups']} pred_raw={it['pred_raw']} "
                        f"pred_kept={it['pred_kept']} pred_matched={it['pred_matched']} "
                        f"avg_score_kept={it['avg_score_kept']:.4f}"
                    )
                print(
                    "[GroupHOIMetrics][PerImageSummary] "
                    f"mean_pred_raw={float(np.mean([x['pred_raw'] for x in per_image_debug])):.2f} "
                    f"mean_pred_kept={float(np.mean([x['pred_kept'] for x in per_image_debug])):.2f} "
                    f"mean_pred_matched={float(np.mean([x['pred_matched'] for x in per_image_debug])):.2f} "
                    f"mean_avg_score_kept={float(np.mean([x['avg_score_kept'] for x in per_image_debug])):.4f}"
                )
        if _is_main_process() and total_gt_groups == 0:
            print(
                "[GroupHOIMetrics] No valid GT groups in evaluated set; "
                "group metrics are expected to be 0. Check whether val split contains multi-member group HOIs."
            )
        if _is_main_process() and total_gt_groups > 0 and total_pred_groups == 0:
            print(
                "[GroupHOIMetrics] GT groups exist but predicted groups are empty; "
                "check group_build_mode/pred_group payload and score thresholds."
            )
        if self.verbose and _is_main_process() and graph_stats_cnt > 0:
            print(
                "[GroupHOIMetrics][GraphStats] "
                f"edge_score_thresh={self.group_edge_score_thresh:.3f}, "
                f"pred_score_thr={self.group_pred_score_thr:.3f}, obj_cluster_iou={self.group_obj_cluster_iou:.3f}, "
                f"max_groups_per_image={self.max_groups_per_image}, "
                f"avg_edges_in={graph_stats_acc['num_edges_in']/graph_stats_cnt:.2f}, "
                f"avg_edges_kept={graph_stats_acc['num_edges_kept']/graph_stats_cnt:.2f}, "
                f"avg_components={graph_stats_acc['num_components']/graph_stats_cnt:.2f}, "
                f"avg_component_size={graph_stats_acc['avg_component_size']/graph_stats_cnt:.2f}"
            )
        if _is_main_process() and total_gt_groups == 0 and total_raw_group_items > 0:
            print(
                "[GroupHOIMetrics] Warning: raw group annotations exist but none were parsed "
                "into valid members. Check member keys/ids/action fields in your val annotations."
            )
            # 打印一条样例帮助快速定位 schema 问题
            sample = None
            for gt_img in gts:
                rg = (
                    gt_img.get("group_interactions")
                    or gt_img.get("group_annotations")
                    or gt_img.get("groups")
                    or []
                )
                if rg:
                    sample = rg[0]
                    break
            if sample is not None:
                if isinstance(sample, dict):
                    print(f"[GroupHOIMetrics][GT sample keys] {sorted(sample.keys())}")
                    if isinstance(sample.get('members'), list) and sample.get('members'):
                        m0 = sample['members'][0]
                        if isinstance(m0, dict):
                            print(f"[GroupHOIMetrics][GT sample member keys] {sorted(m0.keys())}")
                else:
                    print(f"[GroupHOIMetrics][GT sample type] {type(sample)}")
        return out


class MyDatasetEvaluator:
    """
    RLIPv2-style evaluator entry point.

    Expected call pattern inside training:
        evaluator = MyDatasetEvaluator(preds, gts, subject_category_id, args)
        stats = evaluator.evaluate()
    """

    def __init__(self, preds, gts, subject_category_id=None, args=None):
        t0_ctor = time.time()
        # ---------------- Basic config ----------------
        self.overlap_iou = float(getattr(args, "iou_thresh", 0.5)) if args is not None else 0.5
        self.subject_category_id = subject_category_id
        verb_roles_path = getattr(args, "verb_roles_path", _DEFAULT_VERB_ROLES_PATH) if args is not None else _DEFAULT_VERB_ROLES_PATH
        objects_path = getattr(args, "objects_path", _DEFAULT_OBJECTS_PATH) if args is not None else _DEFAULT_OBJECTS_PATH
        verb_tokens = list(getattr(args, "verb_id_to_token", []) if args is not None else [])
        obj_tokens = list(getattr(args, "object_id_to_category", []) if args is not None else [])
        if not verb_tokens:
            verb_tokens = _load_lines(verb_roles_path)
        if not obj_tokens:
            obj_tokens = _load_lines(objects_path)
        self.verb_id_to_token = verb_tokens
        self.object_id_to_category = obj_tokens
        self.max_hois = int(getattr(args, "max_hois", 100)) if args is not None else 100
        self.group_max_hois = int(getattr(args, "group_max_hois", 2000)) if args is not None else 2000
        self.eval_topk_verbs_per_query = int(getattr(args, "eval_topk_verbs_per_query", 20)) if args is not None else 20
        # IMPORTANT:
        # For MYDS val/test built by datasets_gen/myds.py, actions are integer verb IDs.
        # Keep ID-space matching by default to avoid accidental remapping through an
        # external verb token file with different ordering.
        self.eval_action_by_id = bool(getattr(args, "eval_action_by_id", True)) if args is not None else True
        self.num_verb_classes = int(getattr(args, "num_verb_classes", 0)) if args is not None else 0
        self.hoi_id_to_verb_id = {}
        self.verb_token_to_id = {}
        self.verb_id_to_base = {}
        if args is not None and getattr(args, "hoi_path", None):
            try:
                _meta = load_myds_meta(getattr(args, "hoi_path"))
                self.verb_token_to_id = {str(k).strip().lower(): int(v) for k, v in _meta["verb2id"].items()}
                self.verb_id_to_base = {int(i): str(v).strip().lower() for i, v in _meta["id2verb"].items()}
                # HOI classifier index -> base verb index
                for hid, (v_tok, _o_tok) in _meta["id2hoi"].items():
                    vid = _meta["verb2id"].get(str(v_tok).strip().lower(), None)
                    if vid is not None:
                        self.hoi_id_to_verb_id[int(hid)] = int(vid)
            except Exception:
                self.hoi_id_to_verb_id = {}
                self.verb_id_to_base = {}

        # NMS
        self.use_nms_filter = bool(getattr(args, "use_nms_filter", False)) if args is not None else False
        self.thres_nms = float(getattr(args, "thres_nms", 0.7)) if args is not None else 0.7
        self.nms_alpha = float(getattr(args, "nms_alpha", 1.0)) if args is not None else 1.0
        self.nms_beta = float(getattr(args, "nms_beta", 0.5)) if args is not None else 0.5

        # Rare/non-rare split params (triplet-level, computed from TRAIN frequency)
        self.eval_train_json = getattr(args, "eval_train_json", None) if args is not None else None
        self.eval_debug = bool(getattr(args, "eval_debug", False)) if args is not None else False
        self.enable_role_prior_eval = bool(getattr(args, "enable_role_prior_eval", False)) if args is not None else False
        self.enable_subset_metrics = bool(getattr(args, "enable_subset_metrics", False)) if args is not None else False
        self.rare_thresh = int(getattr(args, "eval_rare_thresh", 10)) if args is not None else 10
        self.bbox_format = getattr(args, "eval_bbox_format", "xyxy") if args is not None else "xyxy"

        # Ignore prefixes (e.g., no_interaction)
        ignore = getattr(args, "ignore_prefixes", ["no_interaction"]) if args is not None else ["no_interaction"]
        self.ignore_action_prefixes = tuple(ignore)

        # Subset-HOI metrics config (pair-level edge evaluation; overlapping subsets enabled).
        # Non-contact is a semantic overlay subset, not an exclusive structural subset.
        self.non_contact_base_verbs = set(getattr(args, "non_contact_base_verbs", [
            "look", "look_at", "gaze", "watch", "point", "point_to", "read",
            "talk", "talk_to", "listen", "listen_to", "smile", "wave", "gesture"
        ]) if args is not None else [
            "look", "look_at", "gaze", "watch", "point", "point_to", "read",
            "talk", "talk_to", "listen", "listen_to", "smile", "wave", "gesture"
        ])
        self.non_contact_full_tokens = set(getattr(args, "non_contact_full_tokens", []) if args is not None else [])
        self.subset_names = [
            "multi_person_single_object",
            "single_person_multi_object",
            "multi_person_multi_object",
            "person_person",
            "non_contact",
        ]

        # Debug options for validating non-contact mapping and verb-id vocabulary alignment.
        self.debug_non_contact = bool(getattr(args, "debug_non_contact", False)) if args is not None else False
        self.debug_non_contact_samples = int(getattr(args, "debug_non_contact_samples", 20)) if args is not None else 20

        # ---------------- Triplet registries ----------------
        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(int)
        self.gt_triplets: List[Triplet] = []

        # ---------------- Pair(BBox-only) registries ----------------
        self.fp_pair = defaultdict(list)
        self.tp_pair = defaultdict(list)
        self.score_pair = defaultdict(list)
        self.sum_gts_pair = defaultdict(int)
        self.gt_pairs: List[PairKey] = []

        # ---------------- Action-only registries ----------------
        self.fp_act = defaultdict(list)
        self.tp_act = defaultdict(list)
        self.score_act = defaultdict(list)
        self.sum_gts_act = defaultdict(int)
        self.gt_actions: List[ActionKey] = []

        # Internal aligned structures
        self.preds: List[Dict[str, Any]] = []
        self.gts: List[Dict[str, Any]] = []
        self.group_evaluator = GroupHOIMetrics(args=args, overlap_iou=self.overlap_iou, norm_cat_fn=self._norm_cat)

        # Build internal structures from raw preds & gts
        self._build_from_preds_gts(preds, gts)
        if self.eval_debug and _is_main_process():
            print(f"[EvalDebug][MyDatasetEvaluator::__init__] _build_from_preds_gts took {time.time() - t0_ctor:.3f}s")

        # Optional: triplet NMS per image
        if self.use_nms_filter:
            t_nms = time.time()
            self.preds = [self.triplet_nms_filter_single(p) for p in self.preds]
            if self.eval_debug and _is_main_process():
                print(f"[EvalDebug][MyDatasetEvaluator::__init__] triplet_nms_filter_single(all images) took {time.time() - t_nms:.3f}s")

        # Build rare/non-rare sets (projected to eval triplets)
        t_rare = time.time()
        self.rare_triplets, self.nonrare_triplets = self._build_rare_nonrare_sets(eval_gts_raw=gts)
        if self.eval_debug and _is_main_process():
            print(f"[EvalDebug][MyDatasetEvaluator::__init__] _build_rare_nonrare_sets took {time.time() - t_rare:.3f}s")

        total_gts = sum(self.sum_gts.values())
        if _is_main_process():
            # extra debug: how many eval triplets have train counts
            # (computed inside _build_rare_nonrare_sets, but we keep message here concise)
            print(
                f"[MyDatasetEvaluator] num_images={len(self.gts)}, "
                f"num_gt_triplets={len(self.gt_triplets)}, total_gt_hois={total_gts}, "
                f"rare_triplets={len(self.rare_triplets)}, nonrare_triplets={len(self.nonrare_triplets)}, "
                f"max_hois={self.max_hois}, group_max_hois={self.group_max_hois}"
            )
        if self.eval_debug and _is_main_process():
            print(f"[EvalDebug][MyDatasetEvaluator::__init__] total ctor time {time.time() - t0_ctor:.3f}s")

    def _action_to_base_verb(self, a: Any) -> str:
        """Return base verb token in MYDS verb vocabulary space."""
        na = self._norm_action(a)
        if isinstance(na, (int, np.integer)):
            vid = int(na)
            if vid in self.verb_id_to_base:
                return self.verb_id_to_base[vid]
            return str(vid)
        return self.get_base_verb(na)

    # ------------------------------------------------------------------
    # Normalizers / helpers
    # ------------------------------------------------------------------

    def _norm_cat(self, cat: Any) -> str:
        """
        Canonicalize category keys so train-json triplets and eval GT triplets
        stay in the same key space.

        Priority:
          1) numeric id -> object token (if vocabulary available)
          2) fallback to normalized string form
        """
        if isinstance(cat, (int, np.integer)):
            idx = int(cat)
            if 0 <= idx < len(self.object_id_to_category):
                return str(self.object_id_to_category[idx]).strip()
            return str(idx)

        if isinstance(cat, str):
            s = cat.strip()
            # try id-like string first so "1" and 1 map identically
            if s.isdigit():
                idx = int(s)
                if 0 <= idx < len(self.object_id_to_category):
                    return str(self.object_id_to_category[idx]).strip()
            return s

        try:
            idx = int(cat)
            if 0 <= idx < len(self.object_id_to_category):
                return str(self.object_id_to_category[idx]).strip()
            return str(idx)
        except Exception:
            return str(cat).strip()

    def _norm_action(self, a: Any) -> Any:
        if self.eval_action_by_id:
            if isinstance(a, str):
                s = a.strip()
                if s.isdigit():
                    ai = int(s)
                    if self.num_verb_classes > 0 and ai >= self.num_verb_classes and ai in self.hoi_id_to_verb_id:
                        return self.hoi_id_to_verb_id[ai]
                    return ai
                base = self.get_base_verb(s)
                if base in self.verb_token_to_id:
                    return self.verb_token_to_id[base]
                return self.normalize_action_token(s)
            try:
                ai = int(a)
                if self.num_verb_classes > 0 and ai >= self.num_verb_classes and ai in self.hoi_id_to_verb_id:
                    return self.hoi_id_to_verb_id[ai]
                return ai
            except Exception:
                return a

        if isinstance(a, str):
            s = a.strip()
            if s.isdigit():
                idx = int(s)
                if 0 <= idx < len(self.verb_id_to_token):
                    return self.normalize_action_token(self.verb_id_to_token[idx])
                return idx
            return self.normalize_action_token(s)
        try:
            idx = int(a)
            if 0 <= idx < len(self.verb_id_to_token):
                return self.normalize_action_token(self.verb_id_to_token[idx])
            return idx
        except Exception:
            return a

    @staticmethod
    def normalize_action_token(action_token: Any) -> str:
        token = str(action_token).strip().lower()
        if ":" in token:
            left, right = token.split(":", 1)
            token = left.strip() + ":" + right.strip()
        return token

    @classmethod
    def get_base_verb(cls, action_token: Any) -> str:
        return cls.normalize_action_token(action_token).split(":", 1)[0].strip()

    def _action_token_for_subset(self, action_token: Any) -> str:
        if isinstance(action_token, (int, np.integer)):
            idx = int(action_token)
            if 0 <= idx < len(self.verb_id_to_token):
                return self.normalize_action_token(self.verb_id_to_token[idx])
        return self.normalize_action_token(action_token)

    def is_non_contact(self, action_token: Any) -> bool:
        token = self._action_token_for_subset(action_token)
        return (self.get_base_verb(token) in self.non_contact_base_verbs) or (token in self.non_contact_full_tokens)

    def _is_ignored_action(self, action_token: Any) -> bool:
        if not isinstance(action_token, str):
            return False
        for prefix in self.ignore_action_prefixes:
            if action_token.startswith(prefix):
                return True
        return False

    def _get_image_id(self, ann: Dict[str, Any], fallback_idx: int) -> str:
        return str(
            ann.get("filename", None)
            or ann.get("file_name", None)
            or ann.get("image_id", None)
            or ann.get("id", None)
            or fallback_idx
        )

    # ------------------------------------------------------------------
    # Build internal structures
    # ------------------------------------------------------------------

    def _build_from_preds_gts(self, preds, gts) -> None:
        """
        Convert raw preds/gts passed from RLIPv2 into internal structures:
          pred: {image_id, predictions[bboxes], hoi_prediction[hois]}
          gt  : {image_id, annotations[bboxes], hoi_annotation[hois]}
        and register triplets/pairs/actions in GT.
        """

        # 1) preds
        for idx, img_preds in enumerate(preds):
            img_preds_np = {}
            for k, v in img_preds.items():
                if hasattr(v, "cpu"):
                    img_preds_np[k] = v.cpu().numpy()
                else:
                    img_preds_np[k] = np.asarray(v)

            boxes = img_preds_np["boxes"]            # [N,4]
            labels = img_preds_np["labels"]          # [N] or [N,D]
            verb_scores = img_preds_np["verb_scores"]  # [Q,V]
            sub_ids = img_preds_np["sub_ids"]        # [Q]
            obj_ids = img_preds_np["obj_ids"]        # [Q]

            pred_bboxes = []
            for box, lbl in zip(boxes, labels):
                pred_bboxes.append({"bbox": _as_xyxy(box, self.bbox_format), "category": _label_to_int(lbl)})

            num_queries, num_verbs = verb_scores.shape
            keep_k = max(
                int(self.max_hois) if self.max_hois > 0 else 0,
                int(self.group_max_hois) if self.group_max_hois > 0 else 0
            )

            # Fast path: per-query top-k verbs first, then global top-k over reduced pool.
            kq = max(1, min(int(self.eval_topk_verbs_per_query), num_verbs))
            if kq < num_verbs:
                local_part = np.argpartition(verb_scores, -kq, axis=1)[:, -kq:]  # [Q, kq]
            else:
                local_part = np.tile(np.arange(num_verbs, dtype=np.int32), (num_queries, 1))

            q_ids = np.repeat(np.arange(num_queries, dtype=np.int32), local_part.shape[1])
            v_ids = local_part.reshape(-1).astype(np.int32, copy=False)
            cand_scores = verb_scores[q_ids, v_ids].astype(np.float32, copy=False)

            if keep_k <= 0 or keep_k >= cand_scores.size:
                order = np.argsort(cand_scores)[::-1]
            else:
                part = np.argpartition(cand_scores, -keep_k)[-keep_k:]
                order = part[np.argsort(cand_scores[part])[::-1]]

            pred_hois = [
                {
                    "subject_id": int(sub_ids[q_ids[i]]),
                    "object_id": int(obj_ids[q_ids[i]]),
                    "action": int(v_ids[i]),
                    "score": float(cand_scores[i]),
                }
                for i in order
            ]

            pred_hois_for_group = pred_hois[: self.group_max_hois] if self.group_max_hois > 0 else pred_hois
            pred_hois_for_hoi = pred_hois[: self.max_hois] if self.max_hois > 0 else pred_hois

            # image_id: prefer gts entry if present
            img_gt_raw = gts[idx]
            image_id = self._get_image_id(img_gt_raw, idx)

            pred_item = {
                "image_id": image_id,
                "predictions": pred_bboxes,
                "hoi_prediction": pred_hois_for_hoi,
                "hoi_prediction_for_group": pred_hois_for_group,
            }
            # Accept both historical "group_prediction" and model-side "pred_groups"
            # produced by PostProcessHOI/PostProcessSGG.
            if "group_prediction" in img_preds or "pred_groups" in img_preds:
                gp = img_preds.get("group_prediction", img_preds.get("pred_groups", []))
                if hasattr(gp, "cpu"):
                    gp = gp.cpu().numpy().tolist()
                pred_item["group_prediction"] = gp
            self.preds.append(pred_item)

        # 2) gts
        for idx, img_gt in enumerate(gts):
            image_id = self._get_image_id(img_gt, idx)
            gt_struct = self._convert_single_gt(img_gt, image_id=image_id)
            self.gts.append(gt_struct)

    def _register_gt_triplet(self, sub_cat: Any, obj_cat: Any, action: Any) -> None:
        t: Triplet = (self._norm_cat(sub_cat), self._norm_cat(obj_cat), self._norm_action(action))
        if t not in self.gt_triplets:
            self.gt_triplets.append(t)
        self.sum_gts[t] += 1

    def _register_gt_pair(self, sub_cat: Any, obj_cat: Any) -> None:
        p: PairKey = (self._norm_cat(sub_cat), self._norm_cat(obj_cat))
        if p not in self.gt_pairs:
            self.gt_pairs.append(p)
        self.sum_gts_pair[p] += 1

    def _register_gt_action(self, action: Any) -> None:
        a = self._norm_action(action)
        if a not in self.gt_actions:
            self.gt_actions.append(a)
        self.sum_gts_act[a] += 1

    def _convert_single_gt(self, img_gt: Dict[str, Any], *, image_id: str) -> Dict[str, Any]:
        """
        Support GT styles:
          A) RLIPv2/HICO-like tensors: boxes + labels + hois
          B) boxes + labels + interactions (action tokens)
          C) entities + interactions (pure JSON)
        Output internal dict:
          {image_id, annotations[bboxes], hoi_annotation[hois]}
        and register triplets/pairs/actions.
        """
        # Case A/B: boxes+labels exist
        if "boxes" in img_gt and "labels" in img_gt:
            boxes = img_gt["boxes"]
            labels = img_gt["labels"]
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
            else:
                boxes = np.asarray(boxes)
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            else:
                labels = np.asarray(labels)

            annotations = []
            for box, lbl in zip(boxes, labels):
                annotations.append({"bbox": _as_xyxy(box, self.bbox_format), "category": _label_to_int(lbl)})

            hoi_ann: List[Dict[str, Any]] = []

            # A) hois present (sub_idx, obj_idx, verb_id)
            if "hois" in img_gt and img_gt["hois"] is not None:
                hois_arr = img_gt["hois"]
                if hasattr(hois_arr, "cpu"):
                    hois_arr = hois_arr.cpu().numpy()
                else:
                    hois_arr = np.asarray(hois_arr)
                for sub_idx, obj_idx, verb_idx in hois_arr:
                    s = int(sub_idx)
                    o = int(obj_idx)
                    a = int(verb_idx)
                    hoi_ann.append({"subject_id": s, "object_id": o, "action": a})

                    # register triplet & action
                    self._register_gt_triplet(annotations[s]["category"], annotations[o]["category"], a)
                    self._register_gt_action(a)

                # register pairs (dedup by (s,o) per image)
                seen_so = set()
                for h in hoi_ann:
                    so = (int(h["subject_id"]), int(h["object_id"]))
                    if so in seen_so:
                        continue
                    seen_so.add(so)
                    s, o = so
                    self._register_gt_pair(annotations[s]["category"], annotations[o]["category"])

                return {
                    "image_id": image_id,
                    "annotations": annotations,
                    "hoi_annotation": hoi_ann,
                    "entities": img_gt.get("entities", []),
                    "group_interactions": img_gt.get("group_interactions", []),
                    "group_annotations": img_gt.get("group_annotations", []),
                    "groups": img_gt.get("groups", []),
                    "group_instances": img_gt.get("group_instances", []),
                }

            # B) interactions present (string tokens)
            if "interactions" in img_gt:
                entities = img_gt.get("entities", [])
                ent_id_to_idx = {e["entity_id"]: i for i, e in enumerate(entities)} if entities else None

                for inter in img_gt.get("interactions", []):
                    s_id, o_id = inter["subject_id"], inter["object_id"]
                    if ent_id_to_idx is not None and not isinstance(s_id, int):
                        if s_id not in ent_id_to_idx or o_id not in ent_id_to_idx:
                            continue
                        s_idx, o_idx = ent_id_to_idx[s_id], ent_id_to_idx[o_id]
                    else:
                        s_idx, o_idx = int(s_id), int(o_id)

                    tokens = list(inter.get("action_tokens", []))
                    if "action" in inter:
                        tokens.append(inter["action"])

                    for tok in tokens:
                        if self._is_ignored_action(tok):
                            continue
                        a = self._norm_action(tok)
                        hoi_ann.append({"subject_id": s_idx, "object_id": o_idx, "action": a})
                        self._register_gt_triplet(annotations[s_idx]["category"], annotations[o_idx]["category"], a)
                        self._register_gt_action(a)

                # register pairs (dedup by (s,o))
                seen_so = set()
                for h in hoi_ann:
                    so = (int(h["subject_id"]), int(h["object_id"]))
                    if so in seen_so:
                        continue
                    seen_so.add(so)
                    s, o = so
                    self._register_gt_pair(annotations[s]["category"], annotations[o]["category"])

                return {
                    "image_id": image_id,
                    "annotations": annotations,
                    "hoi_annotation": hoi_ann,
                    "entities": img_gt.get("entities", []),
                    "group_interactions": img_gt.get("group_interactions", []),
                    "group_annotations": img_gt.get("group_annotations", []),
                    "groups": img_gt.get("groups", []),
                    "group_instances": img_gt.get("group_instances", []),
                }

            # If neither hois nor interactions, still return empty HOIs
            return {
                "image_id": image_id,
                "annotations": annotations,
                "hoi_annotation": [],
                "entities": img_gt.get("entities", []),
                "group_interactions": img_gt.get("group_interactions", []),
                "group_annotations": img_gt.get("group_annotations", []),
                "groups": img_gt.get("groups", []),
                "group_instances": img_gt.get("group_instances", []),
            }

        # Case C: entities + interactions
        entities = img_gt.get("entities", [])
        annotations = [{"bbox": _as_xyxy(e["bbox"], self.bbox_format), "category": e["category"]} for e in entities]
        ent_id_to_idx = {e["entity_id"]: i for i, e in enumerate(entities)}

        hoi_ann: List[Dict[str, Any]] = []
        for inter in img_gt.get("interactions", []):
            s_id, o_id = inter["subject_id"], inter["object_id"]
            if s_id not in ent_id_to_idx or o_id not in ent_id_to_idx:
                continue
            s_idx, o_idx = ent_id_to_idx[s_id], ent_id_to_idx[o_id]

            tokens = list(inter.get("action_tokens", []))
            if "action" in inter:
                tokens.append(inter["action"])

            for tok in tokens:
                if self._is_ignored_action(tok):
                    continue
                a = self._norm_action(tok)
                hoi_ann.append({"subject_id": s_idx, "object_id": o_idx, "action": a})
                self._register_gt_triplet(annotations[s_idx]["category"], annotations[o_idx]["category"], a)
                self._register_gt_action(a)

        # register pairs (dedup by (s,o))
        seen_so = set()
        for h in hoi_ann:
            so = (int(h["subject_id"]), int(h["object_id"]))
            if so in seen_so:
                continue
            seen_so.add(so)
            s, o = so
            self._register_gt_pair(annotations[s]["category"], annotations[o]["category"])

        return {
            "image_id": image_id,
            "annotations": annotations,
            "hoi_annotation": hoi_ann,
            "entities": img_gt.get("entities", []),
            "group_interactions": img_gt.get("group_interactions", []),
            "group_annotations": img_gt.get("group_annotations", []),
            "groups": img_gt.get("groups", []),
            "group_instances": img_gt.get("group_instances", []),
        }

    # ------------------------------------------------------------------
    # Rare / non-rare
    # ------------------------------------------------------------------

    def _extract_triplets_from_single_gt_for_count(self, img_gts: Dict[str, Any]) -> List[Triplet]:
        """
        Extract triplets from a single raw GT entry (train_json or fallback eval GT),
        used for frequency counting only.
        """
        triplets: List[Triplet] = []

        # boxes+labels+hois
        if "boxes" in img_gts and "labels" in img_gts:
            labels = img_gts["labels"]
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            else:
                labels = np.asarray(labels)
            cats = [self._norm_cat(_label_to_int(l)) for l in labels]

            if "hois" in img_gts and img_gts["hois"] is not None:
                hois = img_gts["hois"]
                if hasattr(hois, "cpu"):
                    hois = hois.cpu().numpy()
                else:
                    hois = np.asarray(hois)
                for sub_idx, obj_idx, verb_idx in hois:
                    s = int(sub_idx)
                    o = int(obj_idx)
                    a = int(verb_idx)
                    triplets.append((cats[s], cats[o], a))
                return triplets

            if "interactions" in img_gts:
                entities = img_gts.get("entities", [])
                ent_map = {e["entity_id"]: i for i, e in enumerate(entities)} if entities else None
                for inter in img_gts.get("interactions", []):
                    s_id, o_id = inter["subject_id"], inter["object_id"]
                    if ent_map is not None and not isinstance(s_id, int):
                        if s_id not in ent_map or o_id not in ent_map:
                            continue
                        s = ent_map[s_id]
                        o = ent_map[o_id]
                    else:
                        s = int(s_id)
                        o = int(o_id)
                    tokens = list(inter.get("action_tokens", []))
                    if "action" in inter:
                        tokens.append(inter["action"])
                    for tok in tokens:
                        if self._is_ignored_action(tok):
                            continue
                        triplets.append((cats[s], cats[o], self._norm_action(tok)))
                return triplets

        # entities+interactions
        entities = img_gts.get("entities", [])
        cats = [self._norm_cat(e["category"]) for e in entities]
        ent_map = {e["entity_id"]: i for i, e in enumerate(entities)}
        for inter in img_gts.get("interactions", []):
            s_id, o_id = inter["subject_id"], inter["object_id"]
            if s_id not in ent_map or o_id not in ent_map:
                continue
            s = ent_map[s_id]
            o = ent_map[o_id]
            tokens = list(inter.get("action_tokens", []))
            if "action" in inter:
                tokens.append(inter["action"])
            for tok in tokens:
                if self._is_ignored_action(tok):
                    continue
                triplets.append((cats[s], cats[o], self._norm_action(tok)))

        return triplets

    def _build_rare_nonrare_sets(self, eval_gts_raw) -> Tuple[Set[Triplet], Set[Triplet]]:
        """
        Compute rare/non-rare from TRAIN triplet frequency (recommended).
        Project to eval_triplets space:
          - if a triplet is missing in TRAIN counts, treat count=0 (rare).
        """
        cache_key = (self.eval_train_json, self.rare_thresh, self.bbox_format, self.ignore_action_prefixes)
        if cache_key in _TRAIN_FREQ_CACHE:
            rare_all, nonrare_all = _TRAIN_FREQ_CACHE[cache_key]
        else:
            # choose source for counting
            if self.eval_train_json and os.path.isfile(self.eval_train_json):
                with open(self.eval_train_json, "r", encoding="utf-8") as f:
                    src = json.load(f)
                split_from_train = True
            else:
                src = eval_gts_raw
                split_from_train = False

            counts = defaultdict(int)
            for raw in src:
                for t in self._extract_triplets_from_single_gt_for_count(raw):
                    counts[t] += 1

            # store counts-derived sets (global space)
            rare_all = {t for t, c in counts.items() if c < self.rare_thresh}
            nonrare_all = {t for t, c in counts.items() if c >= self.rare_thresh}

            if not split_from_train and _is_main_process():
                print(
                    f"[Warning] eval_train_json not set or not found. "
                    f"Rare/non-rare fallback to *eval GT* frequency (threshold={self.rare_thresh})."
                )

            _TRAIN_FREQ_CACHE[cache_key] = (rare_all, nonrare_all)

        # IMPORTANT: project to eval triplets space with count-missing treated as rare
        # If train json is ID-aligned, rare/nonrare_all will already overlap well.
        # If not, projection still yields a valid partition (but nonrare may be empty).
        # We additionally try to detect overlap quality for debugging.
        # For projection we need actual counts; if cached sets only, we cannot recover counts.
        # So we instead use membership in cached sets as a best-effort, plus "missing->rare".
        eval_space = set(self.gt_triplets)

        # If a triplet is in nonrare_all -> nonrare, else -> rare (includes missing & rare_all)
        nonrare = {t for t in eval_space if t in nonrare_all}
        rare = {t for t in eval_space if t not in nonrare}

        if _is_main_process():
            shared = len(nonrare) + len({t for t in eval_space if t in rare_all})
            # shared is a rough indicator; most important is: nonrare computed from overlap
            print(f"[MyDatasetEvaluator] eval_triplets={len(eval_space)}, triplets_with_train_count≈{shared}")

        return rare, nonrare

    # ------------------------------------------------------------------
    # Metrics: VOC07 AP
    # ------------------------------------------------------------------

    @staticmethod
    def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
        """11-point interpolated AP (VOC2007 style)."""
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0.0
            else:
                p = float(np.max(prec[rec >= t]))
            ap += p / 11.0
        return ap

    def compute_map_triplet(self, triplets_subset: List[Triplet]) -> Dict[str, float]:
        if not triplets_subset:
            return {"mAP": 0.0, "mean max recall": 0.0}

        ap = {}
        max_recall = {}

        for triplet in triplets_subset:
            sum_gts = self.sum_gts.get(triplet, 0)
            if sum_gts == 0:
                continue

            tp = np.asarray(self.tp.get(triplet, []), dtype=np.float32)
            fp = np.asarray(self.fp.get(triplet, []), dtype=np.float32)
            if tp.size == 0:
                ap[triplet] = 0.0
                max_recall[triplet] = 0.0
                continue

            scores = np.asarray(self.score.get(triplet, []), dtype=np.float32)
            order = np.argsort(-scores)
            fp = np.cumsum(fp[order])
            tp = np.cumsum(tp[order])

            rec = tp / float(sum_gts)
            prec = tp / np.maximum(fp + tp, 1e-8)

            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = float(np.max(rec)) if rec.size > 0 else 0.0

        m_ap = float(np.mean(list(ap.values()))) if ap else 0.0
        m_mr = float(np.mean(list(max_recall.values()))) if max_recall else 0.0
        return {"mAP": m_ap, "mean max recall": m_mr}

    def compute_map_pair(self, pairs_subset: List[PairKey]) -> Dict[str, float]:
        if not pairs_subset:
            return {"mAP": 0.0, "mean max recall": 0.0}

        ap = {}
        max_recall = {}

        for pair in pairs_subset:
            sum_gts = self.sum_gts_pair.get(pair, 0)
            if sum_gts == 0:
                continue

            tp = np.asarray(self.tp_pair.get(pair, []), dtype=np.float32)
            fp = np.asarray(self.fp_pair.get(pair, []), dtype=np.float32)
            if tp.size == 0:
                ap[pair] = 0.0
                max_recall[pair] = 0.0
                continue

            scores = np.asarray(self.score_pair.get(pair, []), dtype=np.float32)
            order = np.argsort(-scores)
            fp = np.cumsum(fp[order])
            tp = np.cumsum(tp[order])

            rec = tp / float(sum_gts)
            prec = tp / np.maximum(fp + tp, 1e-8)

            ap[pair] = self.voc_ap(rec, prec)
            max_recall[pair] = float(np.max(rec)) if rec.size > 0 else 0.0

        m_ap = float(np.mean(list(ap.values()))) if ap else 0.0
        m_mr = float(np.mean(list(max_recall.values()))) if max_recall else 0.0
        return {"mAP": m_ap, "mean max recall": m_mr}

    def compute_map_action(self, actions_subset: List[ActionKey]) -> Dict[str, float]:
        if not actions_subset:
            return {"mAP": 0.0, "mean max recall": 0.0}

        ap = {}
        max_recall = {}

        for a in actions_subset:
            sum_gts = self.sum_gts_act.get(a, 0)
            if sum_gts == 0:
                continue

            tp = np.asarray(self.tp_act.get(a, []), dtype=np.float32)
            fp = np.asarray(self.fp_act.get(a, []), dtype=np.float32)
            if tp.size == 0:
                ap[a] = 0.0
                max_recall[a] = 0.0
                continue

            scores = np.asarray(self.score_act.get(a, []), dtype=np.float32)
            order = np.argsort(-scores)
            fp = np.cumsum(fp[order])
            tp = np.cumsum(tp[order])

            rec = tp / float(sum_gts)
            prec = tp / np.maximum(fp + tp, 1e-8)

            ap[a] = self.voc_ap(rec, prec)
            max_recall[a] = float(np.max(rec)) if rec.size > 0 else 0.0

        m_ap = float(np.mean(list(ap.values()))) if ap else 0.0
        m_mr = float(np.mean(list(max_recall.values()))) if max_recall else 0.0
        return {"mAP": m_ap, "mean max recall": m_mr}

    # ------------------------------------------------------------------
    # IoU utilities
    # ------------------------------------------------------------------

    def compute_iou(self, gt_box: Dict[str, Any], pred_box: Dict[str, Any]) -> float:
        """Strict IoU: categories must match, otherwise IoU=0."""
        if self._norm_cat(gt_box["category"]) != self._norm_cat(pred_box["category"]):
            return 0.0

        rec1 = np.asarray(gt_box["bbox"], dtype=np.float32).reshape(-1)
        rec2 = np.asarray(pred_box["bbox"], dtype=np.float32).reshape(-1)
        if rec1.size != 4 or rec2.size != 4:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = rec1
        x1_2, y1_2, x2_2, y2_2 = rec2

        w1 = max(0.0, x2_1 - x1_1 + 1.0)
        h1 = max(0.0, y2_1 - y1_1 + 1.0)
        w2 = max(0.0, x2_2 - x1_2 + 1.0)
        h2 = max(0.0, y2_2 - y1_2 + 1.0)
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
        inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
        inter_area = inter_w * inter_h

        union = area1 + area2 - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def compute_iou_nocat(self, gt_box: Dict[str, Any], pred_box: Dict[str, Any]) -> float:
        """IoU without category constraint (for action-only evaluation)."""
        rec1 = np.asarray(gt_box["bbox"], dtype=np.float32).reshape(-1)
        rec2 = np.asarray(pred_box["bbox"], dtype=np.float32).reshape(-1)
        if rec1.size != 4 or rec2.size != 4:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = rec1
        x1_2, y1_2, x2_2, y2_2 = rec2

        w1 = max(0.0, x2_1 - x1_1 + 1.0)
        h1 = max(0.0, y2_1 - y1_1 + 1.0)
        w2 = max(0.0, x2_2 - x1_2 + 1.0)
        h2 = max(0.0, y2_2 - y1_2 + 1.0)
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
        inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
        inter_area = inter_w * inter_h

        union = area1 + area2 - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def compute_iou_mat(self, gt_bboxes: List[Dict[str, Any]], pred_bboxes: List[Dict[str, Any]]):
        """
        IoU matrix with strict category match.
        Return:
          match_pairs: pred_idx -> [gt_idx...]
          match_overlaps: pred_idx -> [iou... aligned with list above]
        """
        if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
            return {}, {}

        iou_mat = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float32)
        for gi, g in enumerate(gt_bboxes):
            for pi, p in enumerate(pred_bboxes):
                iou_mat[gi, pi] = self.compute_iou(g, p)

        mask = iou_mat >= self.overlap_iou
        match_pairs = {}
        match_overlaps = {}
        if mask.max() > 0:
            gt_inds, pred_inds = np.nonzero(mask)
            for gt_i, pred_i in zip(gt_inds, pred_inds):
                pred_i = int(pred_i)
                gt_i = int(gt_i)
                match_pairs.setdefault(pred_i, []).append(gt_i)
                match_overlaps.setdefault(pred_i, []).append(float(iou_mat[gt_i, pred_i]))
        return match_pairs, match_overlaps

    def compute_iou_mat_nocat(self, gt_bboxes: List[Dict[str, Any]], pred_bboxes: List[Dict[str, Any]]):
        """IoU matrix without category match (for action-only)."""
        if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
            return {}, {}

        iou_mat = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float32)
        for gi, g in enumerate(gt_bboxes):
            for pi, p in enumerate(pred_bboxes):
                iou_mat[gi, pi] = self.compute_iou_nocat(g, p)

        mask = iou_mat >= self.overlap_iou
        match_pairs = {}
        match_overlaps = {}
        if mask.max() > 0:
            gt_inds, pred_inds = np.nonzero(mask)
            for gt_i, pred_i in zip(gt_inds, pred_inds):
                pred_i = int(pred_i)
                gt_i = int(gt_i)
                match_pairs.setdefault(pred_i, []).append(gt_i)
                match_overlaps.setdefault(pred_i, []).append(float(iou_mat[gt_i, pred_i]))
        return match_pairs, match_overlaps

    # ------------------------------------------------------------------
    # TP/FP matching: Triplet / Pair / Action
    # ------------------------------------------------------------------

    def compute_fptp(
        self,
        pred_hois: List[Dict[str, Any]],
        gt_hois: List[Dict[str, Any]],
        match_pairs: Dict[int, List[int]],
        pred_bboxes: List[Dict[str, Any]],
        bbox_overlaps: Dict[int, List[float]],
        gt_bboxes: List[Dict[str, Any]],
    ) -> None:
        """Triplet-level TP/FP (strict bbox category match via match_pairs)."""
        pos_pred_ids = set(match_pairs.keys())
        vis_tag = np.zeros(len(gt_hois), dtype=np.int32)

        pred_hois = sorted(pred_hois, key=lambda k: float(k.get("score", 0.0)), reverse=True)

        for pred in pred_hois:
            is_match = 0
            max_gt_idx = None
            max_overlap = 0.0

            s_id = int(pred["subject_id"])
            o_id = int(pred["object_id"])
            a_pred = self._norm_action(pred["action"])

            if match_pairs and s_id in pos_pred_ids and o_id in pos_pred_ids:
                sub_cands = match_pairs[s_id]
                obj_cands = match_pairs[o_id]
                sub_ovs = bbox_overlaps[s_id]
                obj_ovs = bbox_overlaps[o_id]

                for gt_idx, gt in enumerate(gt_hois):
                    if gt["subject_id"] in sub_cands and gt["object_id"] in obj_cands:
                        if a_pred != self._norm_action(gt["action"]):
                            continue
                        sub_iou = sub_ovs[sub_cands.index(gt["subject_id"])]
                        obj_iou = obj_ovs[obj_cands.index(gt["object_id"])]
                        ov = min(sub_iou, obj_iou)
                        if ov > max_overlap:
                            max_overlap = ov
                            max_gt_idx = gt_idx
                            is_match = 1

            pred_triplet: Triplet = (
                self._norm_cat(pred_bboxes[s_id]["category"]),
                self._norm_cat(pred_bboxes[o_id]["category"]),
                a_pred,
            )

            if is_match == 1 and max_gt_idx is not None and vis_tag[max_gt_idx] == 0:
                gt = gt_hois[max_gt_idx]
                gt_triplet: Triplet = (
                    self._norm_cat(gt_bboxes[gt["subject_id"]]["category"]),
                    self._norm_cat(gt_bboxes[gt["object_id"]]["category"]),
                    self._norm_action(gt["action"]),
                )
                if gt_triplet not in self.gt_triplets:
                    self.gt_triplets.append(gt_triplet)
                    self.sum_gts[gt_triplet] += 0

                self.fp[gt_triplet].append(0)
                self.tp[gt_triplet].append(1)
                self.score[gt_triplet].append(float(pred["score"]))
                vis_tag[max_gt_idx] = 1
            else:
                if pred_triplet in self.gt_triplets:
                    self.fp[pred_triplet].append(1)
                    self.tp[pred_triplet].append(0)
                    self.score[pred_triplet].append(float(pred["score"]))

    def compute_fptp_pair(
        self,
        pred_pairs: List[Dict[str, Any]],     # {subject_id, object_id, score}
        gt_pairs_inst: List[Dict[str, Any]],  # {subject_id, object_id}
        match_pairs: Dict[int, List[int]],
        pred_bboxes: List[Dict[str, Any]],
        bbox_overlaps: Dict[int, List[float]],
        gt_bboxes: List[Dict[str, Any]],
    ) -> None:
        """Pair-level TP/FP: bbox-only, strict category match via match_pairs."""
        pos_pred_ids = set(match_pairs.keys())
        vis_tag = np.zeros(len(gt_pairs_inst), dtype=np.int32)

        pred_pairs = sorted(pred_pairs, key=lambda k: float(k.get("score", 0.0)), reverse=True)

        for p in pred_pairs:
            s_id = int(p["subject_id"])
            o_id = int(p["object_id"])

            is_match = 0
            max_gt_idx = None
            max_overlap = 0.0

            if match_pairs and s_id in pos_pred_ids and o_id in pos_pred_ids:
                sub_cands = match_pairs[s_id]
                obj_cands = match_pairs[o_id]
                sub_ovs = bbox_overlaps[s_id]
                obj_ovs = bbox_overlaps[o_id]

                for gi, gt in enumerate(gt_pairs_inst):
                    gs = int(gt["subject_id"])
                    go = int(gt["object_id"])
                    if gs in sub_cands and go in obj_cands:
                        sub_iou = sub_ovs[sub_cands.index(gs)]
                        obj_iou = obj_ovs[obj_cands.index(go)]
                        ov = min(sub_iou, obj_iou)
                        if ov > max_overlap:
                            max_overlap = ov
                            max_gt_idx = gi
                            is_match = 1

            pred_pair_key: PairKey = (
                self._norm_cat(pred_bboxes[s_id]["category"]),
                self._norm_cat(pred_bboxes[o_id]["category"]),
            )

            if is_match == 1 and max_gt_idx is not None and vis_tag[max_gt_idx] == 0:
                gt = gt_pairs_inst[max_gt_idx]
                gs = int(gt["subject_id"])
                go = int(gt["object_id"])
                gt_pair_key: PairKey = (
                    self._norm_cat(gt_bboxes[gs]["category"]),
                    self._norm_cat(gt_bboxes[go]["category"]),
                )
                if gt_pair_key not in self.gt_pairs:
                    self.gt_pairs.append(gt_pair_key)
                    self.sum_gts_pair[gt_pair_key] += 0

                self.fp_pair[gt_pair_key].append(0)
                self.tp_pair[gt_pair_key].append(1)
                self.score_pair[gt_pair_key].append(float(p["score"]))
                vis_tag[max_gt_idx] = 1
            else:
                if pred_pair_key in self.gt_pairs:
                    self.fp_pair[pred_pair_key].append(1)
                    self.tp_pair[pred_pair_key].append(0)
                    self.score_pair[pred_pair_key].append(float(p["score"]))

    def compute_fptp_action(
        self,
        pred_hois: List[Dict[str, Any]],
        gt_hois: List[Dict[str, Any]],
        match_pairs_nocat: Dict[int, List[int]],
        pred_bboxes: List[Dict[str, Any]],
        bbox_overlaps_nocat: Dict[int, List[float]],
        gt_bboxes: List[Dict[str, Any]],
    ) -> None:
        """Action-level TP/FP: key=action, bbox matching WITHOUT category constraint."""
        pos_pred_ids = set(match_pairs_nocat.keys())
        vis_tag = np.zeros(len(gt_hois), dtype=np.int32)

        pred_hois = sorted(pred_hois, key=lambda k: float(k.get("score", 0.0)), reverse=True)

        for pred in pred_hois:
            s_id = int(pred["subject_id"])
            o_id = int(pred["object_id"])
            a_pred = self._norm_action(pred["action"])

            is_match = 0
            max_gt_idx = None
            max_overlap = 0.0

            if match_pairs_nocat and s_id in pos_pred_ids and o_id in pos_pred_ids:
                sub_cands = match_pairs_nocat[s_id]
                obj_cands = match_pairs_nocat[o_id]
                sub_ovs = bbox_overlaps_nocat[s_id]
                obj_ovs = bbox_overlaps_nocat[o_id]

                for gt_idx, gt in enumerate(gt_hois):
                    if vis_tag[gt_idx] == 1:
                        continue
                    if a_pred != self._norm_action(gt["action"]):
                        continue

                    gs = int(gt["subject_id"])
                    go = int(gt["object_id"])
                    if gs in sub_cands and go in obj_cands:
                        sub_iou = sub_ovs[sub_cands.index(gs)]
                        obj_iou = obj_ovs[obj_cands.index(go)]
                        ov = min(sub_iou, obj_iou)
                        if ov > max_overlap:
                            max_overlap = ov
                            max_gt_idx = gt_idx
                            is_match = 1

            if is_match == 1 and max_gt_idx is not None:
                self.fp_act[a_pred].append(0)
                self.tp_act[a_pred].append(1)
                self.score_act[a_pred].append(float(pred["score"]))
                vis_tag[max_gt_idx] = 1
            else:
                # only count FP for actions that exist in eval GT space
                if a_pred in self.gt_actions:
                    self.fp_act[a_pred].append(1)
                    self.tp_act[a_pred].append(0)
                    self.score_act[a_pred].append(float(pred["score"]))

    # ------------------------------------------------------------------
    # Triplet NMS (optional)
    # ------------------------------------------------------------------

    def triplet_nms_filter_single(self, img_preds: Dict[str, Any]) -> Dict[str, Any]:
        pred_bboxes = img_preds["predictions"]
        pred_hois = img_preds["hoi_prediction"]

        all_triplets: Dict[str, Dict[str, Any]] = {}
        for index, pred_hoi in enumerate(pred_hois):
            triplet_key = "{}_{}_{}".format(
                self._norm_cat(pred_bboxes[pred_hoi["subject_id"]]["category"]),
                self._norm_cat(pred_bboxes[pred_hoi["object_id"]]["category"]),
                self._norm_action(pred_hoi["action"]),
            )
            if triplet_key not in all_triplets:
                all_triplets[triplet_key] = {"subs": [], "objs": [], "scores": [], "indexes": []}

            all_triplets[triplet_key]["subs"].append(pred_bboxes[pred_hoi["subject_id"]]["bbox"])
            all_triplets[triplet_key]["objs"].append(pred_bboxes[pred_hoi["object_id"]]["bbox"])
            all_triplets[triplet_key]["scores"].append(float(pred_hoi["score"]))
            all_triplets[triplet_key]["indexes"].append(index)

        all_keep_inds: List[int] = []
        for _, v in all_triplets.items():
            subs = np.asarray(v["subs"], dtype=np.float32)
            objs = np.asarray(v["objs"], dtype=np.float32)
            scores = np.asarray(v["scores"], dtype=np.float32)
            keep_local = self.pairwise_nms(subs, objs, scores)
            keep_global = list(np.asarray(v["indexes"], dtype=np.int64)[keep_local])
            all_keep_inds.extend(keep_global)

        return {
            "image_id": img_preds["image_id"],
            "predictions": pred_bboxes,
            "hoi_prediction": list(np.asarray(pred_hois, dtype=object)[all_keep_inds]),
        }

    def pairwise_nms(self, subs: np.ndarray, objs: np.ndarray, scores: np.ndarray) -> List[int]:
        if subs.size == 0:
            return []

        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1.0) * (sy2 - sy1 + 1.0)
        obj_areas = (ox2 - ox1 + 1.0) * (oy2 - oy1 + 1.0)

        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1.0)
            sh = np.maximum(0.0, syy2 - syy1 + 1.0)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1.0)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1.0)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter / np.maximum(sub_union, 1e-8), self.nms_alpha) * \
                  np.power(obj_inter / np.maximum(obj_union, 1e-8), self.nms_beta)

            inds = np.where(ovr <= self.thres_nms)[0]
            order = order[inds + 1]

        return keep

    def _is_person_category(self, cat: Any) -> bool:
        c = self._norm_cat(cat).strip().lower()
        if c == "person":
            return True
        try:
            cid = int(float(c))
            if 0 <= cid < len(self.object_id_to_category):
                if self.object_id_to_category[cid].strip().lower() == "person":
                    return True
            if self.subject_category_id is not None and cid == int(self.subject_category_id):
                return True
        except Exception:
            pass
        return False

    def _assign_subset_membership(self, img_gt: Dict[str, Any]) -> List[Set[str]]:
        subsets = [set() for _ in img_gt["hoi_annotation"]]
        ann = img_gt["annotations"]
        # person-person + non-contact + build structural graph over person-object only
        parent = {}
        def find(x):
            parent.setdefault(x,x)
            while parent[x]!=x:
                parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: parent[rb]=ra

        po_edges=[]
        for i,h in enumerate(img_gt["hoi_annotation"]):
            s,o=int(h['subject_id']), int(h['object_id'])
            raw_action = h['action']
            sc=ann[s]['category'] if s < len(ann) else ''
            oc=ann[o]['category'] if o < len(ann) else ''
            s_is_person = self._is_person_category(sc)
            o_is_person = self._is_person_category(oc)
            if s_is_person and o_is_person: subsets[i].add('person_person')
            if self.is_non_contact(raw_action): subsets[i].add('non_contact')
            if s_is_person and (not o_is_person):
                po_edges.append((i,s,o))
                union(('p',s),('o',o))

        comp_nodes=defaultdict(set); comp_edges=defaultdict(list)
        for i,s,o in po_edges:
            r=find(('p',s)); comp_edges[r].append(i)
            comp_nodes[r].add(('p',s)); comp_nodes[r].add(('o',o))
        for r, nodes in comp_nodes.items():
            nperson=sum(1 for t,_ in nodes if t=='p'); nobj=sum(1 for t,_ in nodes if t=='o')
            tag=None
            if nperson>1 and nobj==1: tag='multi_person_single_object'
            elif nperson==1 and nobj>1: tag='single_person_multi_object'
            elif nperson>1 and nobj>1: tag='multi_person_multi_object'
            if tag:
                for ei in comp_edges[r]: subsets[ei].add(tag)
        return subsets

    def _compute_subset_metrics(self) -> Dict[str, Dict[str, float]]:
        # Reuse triplet-level class key and matching; add subset mask + ignore_gt mechanism.
        gt_records=[]; pred_records=[]
        for img_preds, img_gts in zip(self.preds, self.gts):
            subsets = self._assign_subset_membership(img_gts)
            ann=img_gts['annotations']
            for i,h in enumerate(img_gts['hoi_annotation']):
                s,o=int(h['subject_id']),int(h['object_id'])
                gt_records.append({'image_id':img_gts['image_id'],'s':s,'o':o,'a':self._norm_action(h['action']),
                                   'class_key':(self._norm_cat(ann[s]['category']), self._norm_cat(ann[o]['category']), self._norm_action(h['action'])),
                                   'subsets':subsets[i], 'matched':{}})
            pann=img_preds['predictions']
            for p in img_preds['hoi_prediction']:
                s,o=int(p['subject_id']),int(p['object_id'])
                if s>=len(pann) or o>=len(pann):
                    continue
                pred_records.append({'image_id':img_preds['image_id'],'sbox':pann[s],'obox':pann[o],'a':self._norm_action(p['action']),
                                     'class_key':(self._norm_cat(pann[s]['category']), self._norm_cat(pann[o]['category']), self._norm_action(p['action'])),
                                     'score':float(p.get('score',0.0))})
        # speed: cache image_id->annotations and class-key partitions once
        imgid_to_ann = {g["image_id"]: g["annotations"] for g in self.gts}
        gt_by_ck = defaultdict(list)
        pred_by_ck = defaultdict(list)
        for i, g in enumerate(gt_records):
            gt_by_ck[g["class_key"]].append((i, g))
        for p in pred_records:
            pred_by_ck[p["class_key"]].append(p)

        out={}
        for subset in self.subset_names:
            class_keys=sorted({g['class_key'] for g in gt_records if subset in g['subsets']})
            gt_count=sum(1 for g in gt_records if subset in g['subsets'])
            pred_count=0
            ap_list=[]; mr_list=[]
            for ck in class_keys:
                pos=[(i,g) for i,g in gt_by_ck[ck] if subset in g['subsets']]
                ign=[(i,g) for i,g in gt_by_ck[ck] if subset not in g['subsets']]
                preds=sorted(pred_by_ck[ck], key=lambda x:-x['score'])
                pred_count += len(preds)
                if not pos: continue
                pos_by_img = defaultdict(list)
                ign_by_img = defaultdict(list)
                for i, g in pos:
                    pos_by_img[g['image_id']].append((i, g))
                for i, g in ign:
                    ign_by_img[g['image_id']].append((i, g))
                used=set(); tp=[]; fp=[]; sc=[]
                for p in preds:
                    pos_img = pos_by_img.get(p['image_id'], [])
                    cand=[(i,g) for i,g in pos_img if i not in used and self.compute_iou(imgid_to_ann[g['image_id']][g['s']], p['sbox'])>=self.overlap_iou and self.compute_iou(imgid_to_ann[g['image_id']][g['o']], p['obox'])>=self.overlap_iou]
                    if cand:
                        used.add(cand[0][0]); tp.append(1); fp.append(0); sc.append(p['score']); continue
                    dup=[(i,g) for i,g in pos_img if self.compute_iou(imgid_to_ann[g['image_id']][g['s']], p['sbox'])>=self.overlap_iou and self.compute_iou(imgid_to_ann[g['image_id']][g['o']], p['obox'])>=self.overlap_iou]
                    if dup: tp.append(0); fp.append(1); sc.append(p['score']); continue
                    ignm=False
                    for _,g in ign_by_img.get(p['image_id'], []):
                        ann2=imgid_to_ann[g['image_id']]
                        if self.compute_iou(ann2[g['s']],p['sbox'])>=self.overlap_iou and self.compute_iou(ann2[g['o']],p['obox'])>=self.overlap_iou:
                            ignm=True; break
                    if ignm: continue
                    tp.append(0); fp.append(1); sc.append(p['score'])
                if not tp:
                    ap_list.append(0.0); mr_list.append(0.0); continue
                tp=np.asarray(tp); fp=np.asarray(fp); order=np.argsort(-np.asarray(sc)); tp=np.cumsum(tp[order]); fp=np.cumsum(fp[order])
                rec=tp/float(len(pos)); prec=tp/np.maximum(tp+fp,1e-8)
                ap_list.append(self.voc_ap(rec,prec)); mr_list.append(float(np.max(rec)) if rec.size else 0.0)
            out[subset]={'mAP':float(np.mean(ap_list)) if ap_list else -1.0,'mean max recall':float(np.mean(mr_list)) if mr_list else -1.0,
                         'gt_count':int(gt_count),'pred_count':int(pred_count),'valid_classes':int(len(class_keys))}
        return out

    def _print_non_contact_debug_summary(self) -> None:
        if not (_is_main_process() and self.debug_non_contact):
            return
        print("[NonContactDebug] Enabled non-contact mapping debug.")
        print(f"[NonContactDebug] len(verb_id_to_token)={len(self.verb_id_to_token)}")
        if self.verb_id_to_token:
            head = self.verb_id_to_token[:20]
            tail = self.verb_id_to_token[-5:] if len(self.verb_id_to_token) > 5 else self.verb_id_to_token
            print(f"[NonContactDebug] verb_id_to_token[:20]={head}")
            print(f"[NonContactDebug] verb_id_to_token[-5:]={tail}")
        print(f"[NonContactDebug] non_contact_base_verbs={sorted(list(self.non_contact_base_verbs))}")
        print(f"[NonContactDebug] non_contact_full_tokens={sorted(list(self.non_contact_full_tokens))}")

    def _print_non_contact_debug_samples(self) -> None:
        if not (_is_main_process() and self.debug_non_contact):
            return
        max_samples = max(0, int(self.debug_non_contact_samples))
        sampled = 0
        total_gt_hois = 0
        total_gt_non_contact = 0
        unmapped_int_actions = 0
        for img in self.gts:
            for h in img.get("hoi_annotation", []):
                total_gt_hois += 1
                raw_a = h.get("action")
                mapped = self._action_token_for_subset(raw_a)
                is_nc = self.is_non_contact(raw_a)
                if is_nc:
                    total_gt_non_contact += 1
                if isinstance(raw_a, (int, np.integer)) and (int(raw_a) < 0 or int(raw_a) >= len(self.verb_id_to_token)):
                    unmapped_int_actions += 1
                if sampled < max_samples:
                    print(
                        "[NonContactDebug][Sample] "
                        f"image_id={img.get('image_id','unknown')} raw_action={raw_a} mapped_token={mapped} is_non_contact={is_nc}"
                    )
                    sampled += 1
        ratio = _safe_div(total_gt_non_contact, total_gt_hois)
        print(
            "[NonContactDebug][Summary] "
            f"total_gt_hois={total_gt_hois} total_gt_non_contact={total_gt_non_contact} "
            f"ratio={ratio:.4f} unmapped_int_actions={unmapped_int_actions}"
        )


    def _find_prior_annotation_file(self) -> Optional[str]:
        cands = []
        if self.eval_train_json and os.path.isfile(self.eval_train_json):
            cands.append(self.eval_train_json)
        roots = []
        for p in cands:
            roots.append(os.path.dirname(os.path.dirname(p)))
        if self.eval_train_json:
            roots.append(os.path.dirname(os.path.dirname(self.eval_train_json)))
        for r in roots:
            for rel in ("annotations/train.json", "annotations/train_20k.json", "annotations/train.full.json"):
                fp = os.path.join(r, rel)
                if os.path.isfile(fp):
                    return fp
        return cands[0] if cands else None

    def _compute_role_metrics_prior(self) -> Dict[str, float]:
        valid_roles = ("target", "instrument", "support", "location")
        role_set = set(valid_roles)
        ann_path = self._find_prior_annotation_file()
        prior_records = None
        if ann_path and os.path.isfile(ann_path):
            with open(ann_path, 'r', encoding='utf-8') as f:
                prior_records = json.load(f)
        else:
            prior_records = self.gts
            if _is_main_process():
                print("[MyDatasetEvaluator][WARN] train annotation for role prior not found; fallback to eval-split GT prior.")

        c_vro = defaultdict(int); c_vo = defaultdict(int); c_vr = defaultdict(int); c_v = defaultdict(int)
        for rec in prior_records:
            ents = rec.get('entities', []) or rec.get('annotations', [])
            ent_map = {e.get('entity_id', i): e for i,e in enumerate(ents)}
            for inter in rec.get('interactions', []) or rec.get('hoi_annotation', []):
                sid = inter.get('subject_id'); oid = inter.get('object_id')
                toks = inter.get('action_tokens', [inter.get('action')])
                if toks is None: continue
                obj = ent_map.get(oid)
                if obj is None and isinstance(oid, int) and oid < len(ents): obj = ents[oid]
                ocat = self._norm_cat(obj.get('category')) if isinstance(obj, dict) else self._norm_cat('')
                for tk in toks:
                    if tk is None: continue
                    # support both token actions ("verb:role") and numeric ids
                    if isinstance(tk, (int, np.integer)) or (isinstance(tk, str) and str(tk).strip().isdigit()):
                        v = self._action_to_base_verb(tk)
                        r = "target"
                    else:
                        v, r = parse_action_token(tk)
                        v = str(v).strip().lower()
                    if r not in role_set: continue
                    c_vro[(v,r,ocat)] += 1; c_vo[(v,ocat)] += 1; c_vr[(v,r)] += 1; c_v[v] += 1

        gt_counts = {r:0 for r in valid_roles}
        gt_by_class = defaultdict(int)
        pred_by_class = defaultdict(list)
        std_tp = 0; role_ok_tp = 0
        nic_gt = 0; nic_fp = 0
        for imgp, imgg in zip(self.preds, self.gts):
            gtb = imgg['annotations']; ph = imgp['hoi_prediction']; pb = imgp['predictions']; gh = imgg['hoi_annotation']
            # role GT tuples
            role_gt = []
            std_gt = []
            nic_pairs = []
            for g in gh:
                s,o = int(g['subject_id']), int(g['object_id'])
                ga = self._action_token_for_subset(g.get('action'))
                gv,gr = parse_action_token(ga)
                oc = self._norm_cat(gtb[o]['category']) if o < len(gtb) else ''
                std_gt.append((s,o,gv,oc))
                if gv == 'no_interaction' and gr == 'location':
                    nic_gt += 1; nic_pairs.append((s,o,oc))
                if gr in role_set:
                    gt_counts[gr] += 1
                    role_gt.append((s,o,gv,gr,oc))
                    gt_by_class[(gv,gr,oc)] += 1
            # preds expanded to roles
            role_preds = []
            for p in ph:
                s,o = int(p['subject_id']), int(p['object_id'])
                if s>=len(pb) or o>=len(pb): continue
                v = self._action_to_base_verb(p['action'])
                oc = self._norm_cat(pb[o]['category']); sc=float(p.get('score',0.0))
                den = c_vo.get((v,oc),0)
                probs = {}
                if den>0:
                    for r in valid_roles: probs[r]=c_vro.get((v,r,oc),0)/den
                elif c_v.get(v,0)>0:
                    for r in valid_roles: probs[r]=c_vr.get((v,r),0)/max(c_v[v],1)
                else:
                    for r in valid_roles: probs[r]=1.0/len(valid_roles)
                best_r = max(valid_roles, key=lambda r: probs[r])
                for r in valid_roles:
                    role_preds.append((s,o,v,r,oc,sc*probs[r]))
                # HRER conventional tp check
                best_match=None; best_ov=-1
                for gi,(gs,go,gv,goc) in enumerate(std_gt):
                    if gv!=v or goc!=oc: continue
                    ious=self.compute_iou(gtb[gs], pb[s]), self.compute_iou(gtb[go], pb[o])
                    if min(ious)>=self.overlap_iou and min(ious)>best_ov:
                        best_ov=min(ious); best_match=gi
                if best_match is not None:
                    std_tp += 1
                    _,_,gv,goc = std_gt[best_match]
                    # role correct if any matching role gt
                    ok=False
                    for (gs,go,rv,rr,roc) in role_gt:
                        if rv==v and roc==oc and gs==std_gt[best_match][0] and go==std_gt[best_match][1] and rr==best_r:
                            ok=True; break
                    if ok: role_ok_tp += 1
                # NIC FPR
                if v != 'no_interaction':
                    for gs,go,goc in nic_pairs:
                        if goc!=oc: continue
                        if min(self.compute_iou(gtb[gs], pb[s]), self.compute_iou(gtb[go], pb[o])) >= self.overlap_iou:
                            nic_fp += 1; break
            # AP matching
            used = set()
            role_preds.sort(key=lambda x: -x[5])
            for s,o,v,r,oc,score in role_preds:
                cls=(v,r,oc)
                m=None
                for gi,g in enumerate(role_gt):
                    if gi in used: continue
                    gs,go,gv,gr,goc=g
                    if (gv,gr,goc)!=(v,r,oc): continue
                    if min(self.compute_iou(gtb[gs], pb[s]), self.compute_iou(gtb[go], pb[o]))>=self.overlap_iou:
                        m=gi; break
                pred_by_class[cls].append((score, 1 if m is not None else 0))
                if m is not None: used.add(m)
        ap_by_role = {r:[] for r in valid_roles}; ap_all=[]
        for cls,npos in gt_by_class.items():
            preds = sorted(pred_by_class.get(cls,[]), key=lambda x:-x[0])
            if not preds: ap=0.0
            else:
                tp=np.cumsum(np.array([p[1] for p in preds],dtype=np.float32)); fp=np.cumsum(1-np.array([p[1] for p in preds],dtype=np.float32))
                rec=tp/max(float(npos),1e-8); prec=tp/np.maximum(tp+fp,1e-8); ap=self.voc_ap(rec,prec)
            ap_all.append(ap); ap_by_role[cls[1]].append(ap)
        out={
            'role_mAP_vro_prior': float(np.mean(ap_all)) if ap_all else 0.0,
            'role_AP_target_prior': float(np.mean(ap_by_role['target'])) if ap_by_role['target'] else 0.0,
            'role_AP_instrument_prior': float(np.mean(ap_by_role['instrument'])) if ap_by_role['instrument'] else 0.0,
            'role_AP_support_prior': float(np.mean(ap_by_role['support'])) if ap_by_role['support'] else 0.0,
            'role_AP_location_prior': float(np.mean(ap_by_role['location'])) if ap_by_role['location'] else 0.0,
            'role_HRER_prior': 1.0 - (float(role_ok_tp)/float(std_tp)) if std_tp>0 else 0.0,
            'role_NIC_FPR': float(nic_fp)/float(nic_gt) if nic_gt>0 else 0.0,
            'role_GT_target': float(gt_counts['target']),
            'role_GT_instrument': float(gt_counts['instrument']),
            'role_GT_support': float(gt_counts['support']),
            'role_GT_location': float(gt_counts['location']),
        }
        role_avail=[out['role_AP_target_prior'],out['role_AP_instrument_prior'],out['role_AP_support_prior'],out['role_AP_location_prior']]
        non_empty=[role_avail[i] for i,r in enumerate(valid_roles) if gt_counts[r]>0]
        out['role_RB_mAP_prior']=float(np.mean(non_empty)) if non_empty else 0.0
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        t_eval_all = time.time()
        self._print_non_contact_debug_summary()
        self._print_non_contact_debug_samples()
        if self.eval_debug and _is_main_process():
            print("[EvalDebug][MyDatasetEvaluator] start image-wise TP/FP accumulation")
        # accumulate TP/FP for every image
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds["predictions"]
            gt_bboxes = img_gts["annotations"]
            pred_hois = img_preds["hoi_prediction"]
            gt_hois = img_gts["hoi_annotation"]

            # Triplet matching uses strict category IoU
            if len(gt_bboxes) != 0 and len(pred_hois) != 0:
                match_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(
                    pred_hois=pred_hois,
                    gt_hois=gt_hois,
                    match_pairs=match_pairs,
                    pred_bboxes=pred_bboxes,
                    bbox_overlaps=bbox_overlaps,
                    gt_bboxes=gt_bboxes,
                )

                # ---- Pair(BBox-only) accumulation (ignore action) ----
                # GT pairs: dedup by (subject_id, object_id)
                gt_pairs_inst = []
                seen_gt = set()
                for h in gt_hois:
                    so = (int(h["subject_id"]), int(h["object_id"]))
                    if so in seen_gt:
                        continue
                    seen_gt.add(so)
                    gt_pairs_inst.append({"subject_id": so[0], "object_id": so[1]})

                # Pred pairs: collapse multi-action per (s,o) by max score
                best = {}
                for h in pred_hois:
                    so = (int(h["subject_id"]), int(h["object_id"]))
                    sc = float(h.get("score", 0.0))
                    if so not in best or sc > best[so]:
                        best[so] = sc
                pred_pairs = [{"subject_id": k[0], "object_id": k[1], "score": v} for k, v in best.items()]

                if gt_pairs_inst and pred_pairs:
                    self.compute_fptp_pair(
                        pred_pairs=pred_pairs,
                        gt_pairs_inst=gt_pairs_inst,
                        match_pairs=match_pairs,
                        pred_bboxes=pred_bboxes,
                        bbox_overlaps=bbox_overlaps,
                        gt_bboxes=gt_bboxes,
                    )

                # ---- Action-only accumulation (nocat IoU) ----
                match_pairs_nc, bbox_overlaps_nc = self.compute_iou_mat_nocat(gt_bboxes, pred_bboxes)
                if match_pairs_nc and gt_hois and pred_hois:
                    self.compute_fptp_action(
                        pred_hois=pred_hois,
                        gt_hois=gt_hois,
                        match_pairs_nocat=match_pairs_nc,
                        pred_bboxes=pred_bboxes,
                        bbox_overlaps_nocat=bbox_overlaps_nc,
                        gt_bboxes=gt_bboxes,
                    )

            else:
                # No GT or no predictions: all predicted HOIs become FP (triplet-level)
                for pred_hoi in pred_hois:
                    triplet = (
                        self._norm_cat(pred_bboxes[pred_hoi["subject_id"]]["category"]),
                        self._norm_cat(pred_bboxes[pred_hoi["object_id"]]["category"]),
                        self._norm_action(pred_hoi["action"]),
                    )
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(float(pred_hoi["score"]))

                # For pair/action metrics in this branch:
                # - If no GT, we do not count FP for unseen keys beyond eval space.
                # - If no preds, nothing to accumulate.

        if self.eval_debug and _is_main_process():
            print("[EvalDebug][MyDatasetEvaluator] done accumulation, start AP summaries")
        # Quick action-space sanity diagnostics (helps identify verb-id permutation issues).
        if self.eval_debug and _is_main_process():
            gt_act_set = set([self._norm_action(a) for a in self.gt_actions])
            pred_act_hist = defaultdict(int)
            for img_preds in self.preds:
                for h in img_preds.get("hoi_prediction", []):
                    pred_act_hist[self._norm_action(h.get("action"))] += 1
            pred_act_set = set(pred_act_hist.keys())
            overlap = len(gt_act_set & pred_act_set)
            print(
                f"[EvalDebug][ActionSpace] gt_unique={len(gt_act_set)} pred_unique={len(pred_act_set)} "
                f"overlap={overlap} overlap_ratio={_safe_div(overlap, len(gt_act_set)):.4f}"
            )
            # print top-20 predicted action ids/tokens by frequency
            top_pred = sorted(pred_act_hist.items(), key=lambda x: -x[1])[:20]
            print(f"[EvalDebug][ActionSpace] top_pred_actions={top_pred}")

        # ---------------- Triplet Full / Rare / Non-rare ----------------
        full = self.compute_map_triplet(self.gt_triplets)
        rare_list = [t for t in self.gt_triplets if t in self.rare_triplets]
        nonrare_list = [t for t in self.gt_triplets if t in self.nonrare_triplets]
        rare = self.compute_map_triplet(rare_list)
        nonrare = self.compute_map_triplet(nonrare_list)

        # ---------------- Pair(BBox-only) ----------------
        pair_full = self.compute_map_pair(self.gt_pairs)

        # ---------------- Action-only ----------------
        action_full = self.compute_map_action([self._norm_action(a) for a in self.gt_actions])

        if _is_main_process():
            print("--------------------------------------------------")
            print(f"Pair(BBox) mAP: {pair_full['mAP']:.4f}  mean max recall: {pair_full['mean max recall']:.4f}")
            print(f"Action     mAP: {action_full['mAP']:.4f}  mean max recall: {action_full['mean max recall']:.4f}")
            print(f"Triplet Full     mAP: {full['mAP']:.4f}  mean max recall: {full['mean max recall']:.4f}")
            print(f"Triplet Rare     mAP: {rare['mAP']:.4f}  mean max recall: {rare['mean max recall']:.4f}")
            print(f"Triplet Non-Rare mAP: {nonrare['mAP']:.4f}  mean max recall: {nonrare['mean max recall']:.4f}")

        # normalize action space for group evaluator as well (verb-id aligned).
        group_preds = []
        for p in self.preds:
            pp = dict(p)
            hp = []
            for h in p.get("hoi_prediction", []):
                hh = dict(h)
                hh["action"] = self._norm_action(hh.get("action"))
                hp.append(hh)
            pp["hoi_prediction"] = hp
            if "hoi_prediction_for_group" in p:
                hpg = []
                for h in p.get("hoi_prediction_for_group", []):
                    hh = dict(h)
                    hh["action"] = self._norm_action(hh.get("action"))
                    hpg.append(hh)
                pp["hoi_prediction_for_group"] = hpg
            group_preds.append(pp)
        group_gts = []
        for g in self.gts:
            gg = dict(g)
            gha = []
            for h in g.get("hoi_annotation", []):
                hh = dict(h)
                hh["action"] = self._norm_action(hh.get("action"))
                gha.append(hh)
            gg["hoi_annotation"] = gha
            group_gts.append(gg)

        t_group = time.time()
        group_out = self.group_evaluator.evaluate(group_preds, group_gts)
        if self.eval_debug and _is_main_process():
            print(f"[EvalDebug][MyDatasetEvaluator] group_evaluator.evaluate() took {time.time()-t_group:.3f}s")
        if _is_main_process() and (not isinstance(group_out, dict) or len(group_out) == 0):
            print("[MyDatasetEvaluator] Group evaluator returned empty output (likely disabled). "
                  "Set --enable_group_eval to force group metric computation.")
        gm = float(group_out.get("group_wmAP", 0.0)) if isinstance(group_out, dict) else 0.0
        gr = float(group_out.get("group_mean_max_recall", 0.0)) if isinstance(group_out, dict) else 0.0

        if _is_main_process():
            print(f"Group mAP: {gm:.4f}  mean max recall: {gr:.4f}")
            if isinstance(group_out, dict):
                gt_cnt = int(group_out.get("group_total_gt_groups", 0))
                pd_cnt = int(group_out.get("group_total_pred_groups", 0))
                print(f"Group counts: total_gt_groups={gt_cnt}  total_pred_groups={pd_cnt}")
            print("--------------------------------------------------")

        subset_out = self._compute_subset_metrics() if self.enable_subset_metrics else {
            k: {'mAP': -1.0, 'mean max recall': -1.0, 'gt_count': 0, 'pred_count': 0, 'valid_classes': 0}
            for k in self.subset_names
        }
        if _is_main_process():
            print("[SubsetHOIMetrics]")
            name_map = {
                "multi_person_single_object": "Multi-person single-object",
                "single_person_multi_object": "Single-person multi-object",
                "multi_person_multi_object": "Multi-person multi-object",
                "person_person": "Person-person",
                "non_contact": "Non-contact",
            }
            for key in self.subset_names:
                m = subset_out[key]
                smap = "N/A" if m['mAP'] < 0 else f"{m['mAP']:.4f}"
                srec = "N/A" if m['mean max recall'] < 0 else f"{m['mean max recall']:.4f}"
                print(f"{name_map[key]} mAP: {smap}  mean max recall: {srec}  GT: {m['gt_count']}  pred: {m.get('pred_count', 0)}  valid_classes: {m['valid_classes']}")

        stats = {
            "mAP": gm, "mean max recall": gr, "group_mAP": gm, "group_mean max recall": gr,
            "pair_mAP": pair_full['mAP'], "pair_mean max recall": pair_full['mean max recall'],
            "action_mAP": action_full['mAP'], "action_mean max recall": action_full['mean max recall'],
            "triplet_full_mAP": full['mAP'], "triplet_full_mean max recall": full['mean max recall'],
            "triplet_rare_mAP": rare['mAP'], "triplet_rare_mean max recall": rare['mean max recall'],
            "triplet_non_rare_mAP": nonrare['mAP'], "triplet_non_rare_mean max recall": nonrare['mean max recall'],
        }
        if self.enable_role_prior_eval:
            t_role = time.time()
            role_stats = self._compute_role_metrics_prior()
            if self.eval_debug and _is_main_process():
                print(f"[EvalDebug][MyDatasetEvaluator] _compute_role_metrics_prior() took {time.time()-t_role:.3f}s")
        else:
            role_stats = {
                'role_mAP_vro_prior': 0.0, 'role_AP_target_prior': 0.0, 'role_AP_instrument_prior': 0.0,
                'role_AP_support_prior': 0.0, 'role_AP_location_prior': 0.0, 'role_HRER_prior': 0.0,
                'role_NIC_FPR': 0.0, 'role_GT_target': 0.0, 'role_GT_instrument': 0.0,
                'role_GT_support': 0.0, 'role_GT_location': 0.0, 'role_RB_mAP_prior': 0.0
            }
        stats.update(role_stats)
        if _is_main_process():
            print("[RoleAwarePriorMetrics]")
            print(
                f"role_mAP_vro_prior: {role_stats.get('role_mAP_vro_prior', 0.0):.4f}  "
                f"role_RB_mAP_prior: {role_stats.get('role_RB_mAP_prior', 0.0):.4f}"
            )
            print(
                f"role_AP_target_prior: {role_stats.get('role_AP_target_prior', 0.0):.4f}  "
                f"role_AP_instrument_prior: {role_stats.get('role_AP_instrument_prior', 0.0):.4f}  "
                f"role_AP_support_prior: {role_stats.get('role_AP_support_prior', 0.0):.4f}  "
                f"role_AP_location_prior: {role_stats.get('role_AP_location_prior', 0.0):.4f}"
            )
            print(
                f"role_HRER_prior: {role_stats.get('role_HRER_prior', 0.0):.4f}  "
                f"role_NIC_FPR: {role_stats.get('role_NIC_FPR', 0.0):.4f}"
            )
        if self.eval_debug and _is_main_process():
            print(f"[EvalDebug][MyDatasetEvaluator] total evaluate() time {time.time()-t_eval_all:.3f}s")
            print(
                f"role_GT_target: {int(role_stats.get('role_GT_target', 0.0))}  "
                f"role_GT_instrument: {int(role_stats.get('role_GT_instrument', 0.0))}  "
                f"role_GT_support: {int(role_stats.get('role_GT_support', 0.0))}  "
                f"role_GT_location: {int(role_stats.get('role_GT_location', 0.0))}"
            )

        for k,v in subset_out.items():
            prefix = f"subset_{k}"
            stats[f"{prefix}_mAP"] = v['mAP']
            stats[f"{prefix}_mean_max_recall"] = v['mean max recall']
            stats[f"{prefix}_gt_count"] = v['gt_count']
            stats[f"{prefix}_pred_count"] = v.get('pred_count', 0)
            stats[f"{prefix}_valid_classes"] = v['valid_classes']
        return stats
