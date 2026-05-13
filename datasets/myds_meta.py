from pathlib import Path
import json
import numpy as np


def _read_lines(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return [x.strip() for x in f if x.strip()]


def normalize_verb_token(tok: str):
    t = str(tok).strip().lower()
    if ':' in t:
        base, role = t.split(':', 1)
        return base.strip(), role.strip()
    return t, None


def load_or_build_base_verbs(meta_dir: Path):
    base_path = meta_dir / 'base_verbs.txt'
    if base_path.exists():
        verbs = _read_lines(base_path)
        return [v.lower().strip() for v in verbs]
    role_path = meta_dir / 'verb_roles.txt'
    if not role_path.exists():
        raise FileNotFoundError(f'missing metadata file: {base_path} and fallback {role_path}')
    seen, verbs = set(), []
    for tok in _read_lines(role_path):
        v, _ = normalize_verb_token(tok)
        if v and v not in seen:
            seen.add(v)
            verbs.append(v)
    with open(base_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(verbs) + '\n')
    return verbs


def load_or_build_hoi_classes(hoi_path: Path, objects, verbs):
    meta_dir = hoi_path / 'metadata'
    cls_path = meta_dir / 'hoi_classes.txt'
    if cls_path.exists():
        pairs = []
        for line in _read_lines(cls_path):
            if '\t' in line:
                v, o = line.split('\t', 1)
            else:
                sp = line.split()
                if len(sp) < 2:
                    continue
                v, o = sp[0], sp[1]
            pairs.append((v.strip().lower(), o.strip().lower()))
        return pairs

    # deterministic full cross-product by default
    pairs = [(v, o) for v in verbs for o in objects]
    with open(cls_path, 'w', encoding='utf-8') as f:
        for v, o in pairs:
            f.write(f'{v}\t{o}\n')
    return pairs


def load_myds_meta(hoi_path):
    hoi_path = Path(hoi_path)
    meta_dir = hoi_path / 'metadata'
    if not meta_dir.exists():
        raise FileNotFoundError(f'missing metadata directory: {meta_dir}')

    obj_path = meta_dir / 'objects.txt'
    if not obj_path.exists():
        raise FileNotFoundError(f'missing metadata file: {obj_path}')
    objects = [x.lower().strip() for x in _read_lines(obj_path)]
    if 'person' not in objects:
        raise ValueError('objects.txt must contain category "person"')

    verbs = load_or_build_base_verbs(meta_dir)
    hoi_pairs = load_or_build_hoi_classes(hoi_path, objects, verbs)

    hoi_text_path = meta_dir / 'hoi_text_labels.txt'
    if not hoi_text_path.exists():
        with open(hoi_text_path, 'w', encoding='utf-8') as f:
            for v, o in hoi_pairs:
                f.write(f'a photo of a person {v} {o}\n')

    obj_text_path = meta_dir / 'object_text_labels.txt'
    if not obj_text_path.exists():
        with open(obj_text_path, 'w', encoding='utf-8') as f:
            for o in objects:
                f.write(f'a photo of a {o}\n')

    hoi_texts = _read_lines(hoi_text_path)
    obj_texts = _read_lines(obj_text_path)

    obj2id = {o: i for i, o in enumerate(objects)}
    id2obj = {i: o for o, i in obj2id.items()}
    verb2id = {v: i for i, v in enumerate(verbs)}
    id2verb = {i: v for v, i in verb2id.items()}
    hoi2id = {p: i for i, p in enumerate(hoi_pairs)}
    id2hoi = {i: p for p, i in hoi2id.items()}

    corre_path = meta_dir / 'corre_myds.npy'
    if not corre_path.exists():
        mat = np.zeros((len(objects), len(verbs)), dtype=np.float32)
        for v, o in hoi_pairs:
            if (v in verb2id) and (o in obj2id):
                mat[obj2id[o], verb2id[v]] = 1.0
        np.save(corre_path, mat)

    return {
        'objects': objects,
        'verbs': verbs,
        'hoi_pairs': hoi_pairs,
        'hoi_texts': hoi_texts,
        'obj_texts': obj_texts,
        'obj2id': obj2id,
        'id2obj': id2obj,
        'verb2id': verb2id,
        'id2verb': id2verb,
        'hoi2id': hoi2id,
        'id2hoi': id2hoi,
        'correct_mat_path': corre_path,
    }
