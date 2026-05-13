from pathlib import Path
import json
import numpy as np
from collections import defaultdict

from PIL import Image
import torch
import torch.utils.data
import clip

import datasets.transforms as T
from datasets.myds_meta import load_myds_meta, normalize_verb_token


class MYDS(torch.utils.data.Dataset):
    def __init__(self, img_set, anno_file, transforms, num_queries, args):
        self.img_set = img_set
        self.root = Path(args.hoi_path)
        self.anno_file = Path(anno_file)
        if not self.anno_file.exists():
            raise FileNotFoundError(f'missing annotation file: {self.anno_file}')
        with open(self.anno_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self._transforms = transforms
        self.num_queries = num_queries
        self.meta = load_myds_meta(args.hoi_path)
        self._valid_obj_ids = list(range(len(self.meta['objects'])))
        self._valid_verb_ids = list(range(len(self.meta['verbs'])))
        self.text_label_ids = list(self.meta['hoi2id'].keys())
        self.correct_mat = np.load(self.meta['correct_mat_path'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, self.clip_preprocess = clip.load(args.clip_model, device)
        self.image_roots = self._build_image_roots()

    def _build_image_roots(self):
        roots = [self.root / 'images' / 'train', self.root / 'images' / 'test', self.root / 'images' / 'val', self.root / 'images']
        extra = self.root / 'metadata' / 'image_roots.txt'
        if extra.exists():
            for p in open(extra, 'r', encoding='utf-8'):
                p = p.strip()
                if p:
                    roots.append(Path(p))
        return roots

    def _resolve_image(self, rec):
        attempted = []
        image_path = rec.get('image_path', None)
        if image_path:
            p = Path(image_path)
            attempted.append(str(p))
            if p.is_file():
                return p
            p2 = self.root / image_path
            attempted.append(str(p2))
            if p2.is_file():
                return p2
        file_name = rec.get('file_name') or rec.get('image_id')
        if file_name:
            for r in self.image_roots:
                c = r / str(file_name)
                attempted.append(str(c))
                if c.is_file():
                    return c
        raise FileNotFoundError('image not found; attempted:\n' + '\n'.join(attempted))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        rec = self.annotations[idx]
        img_path = self._resolve_image(rec)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        ents = rec.get('entities', [])
        if self.img_set == 'train' and len(ents) > self.num_queries:
            ents = ents[:self.num_queries]

        entity_id_to_new = {}
        boxes_raw, labels_raw = [], []
        for i, e in enumerate(ents):
            cat = str(e.get('category', '')).strip().lower()
            if cat not in self.meta['obj2id']:
                continue
            b = e.get('bbox', None)
            if b is None or len(b) != 4:
                continue
            boxes_raw.append(b)
            labels_raw.append((i, self.meta['obj2id'][cat]))
            entity_id_to_new[e.get('entity_id', i)] = i

        boxes = torch.as_tensor(boxes_raw, dtype=torch.float32).reshape(-1, 4)
        classes = torch.tensor(labels_raw if self.img_set == 'train' else [x[1] for x in labels_raw], dtype=torch.int64)

        target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
        if self.img_set == 'train':
            if len(boxes) > 0:
                boxes[:, 0::2].clamp_(min=0, max=w)
                boxes[:, 1::2].clamp_(min=0, max=h)
                keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                boxes = boxes[keep]
                classes = classes[keep]
            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else torch.zeros((0,))
            img_0, target_0 = self._transforms[0](img, target)
            img, target = self._transforms[1](img_0, target_0)
            target['clip_inputs'] = self.clip_preprocess(img_0)
            kept_box_indices = [label[0] for label in target['labels']]
            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, hoi_labels, sub_boxes, obj_boxes, sub_obj_pairs = [], [], [], [], [], []
            for inter in rec.get('interactions', []):
                sid = inter.get('subject_id')
                oid = inter.get('object_id')
                if sid not in entity_id_to_new or oid not in entity_id_to_new:
                    continue
                sid = entity_id_to_new[sid]; oid = entity_id_to_new[oid]
                if sid not in kept_box_indices or oid not in kept_box_indices:
                    continue
                obj_cls = target['labels'][kept_box_indices.index(oid)].item()
                for token in inter.get('action_tokens', []):
                    vb, _ = normalize_verb_token(token)
                    if vb.startswith('no_interaction') or vb not in self.meta['verb2id']:
                        continue
                    pair = (vb, self.meta['id2obj'][obj_cls])
                    if pair not in self.meta['hoi2id']:
                        continue
                    vp = self.meta['verb2id'][vb]
                    sop = (sid, oid)
                    if sop in sub_obj_pairs:
                        j = sub_obj_pairs.index(sop)
                        verb_labels[j][vp] = 1
                        hoi_labels[j][self.meta['hoi2id'][pair]] = 1
                    else:
                        sub_obj_pairs.append(sop)
                        obj_labels.append(target['labels'][kept_box_indices.index(oid)])
                        vl = [0] * len(self._valid_verb_ids)
                        hl = [0] * len(self.text_label_ids)
                        vl[vp] = 1
                        hl[self.meta['hoi2id'][pair]] = 1
                        verb_labels.append(vl); hoi_labels.append(hl)
                        sub_boxes.append(target['boxes'][kept_box_indices.index(sid)])
                        obj_boxes.append(target['boxes'][kept_box_indices.index(oid)])

            target['filename'] = str(img_path)
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['hoi_labels'] = torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['hoi_labels'] = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
        else:
            target['filename'] = str(img_path)
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
            img_0, _ = self._transforms[0](img, None)
            img, _ = self._transforms[1](img_0, None)
            target['clip_inputs'] = self.clip_preprocess(img_0)
            hois = []
            for inter in rec.get('interactions', []):
                sid = inter.get('subject_id'); oid = inter.get('object_id')
                if sid not in entity_id_to_new or oid not in entity_id_to_new:
                    continue
                sid = entity_id_to_new[sid]; oid = entity_id_to_new[oid]
                for token in inter.get('action_tokens', []):
                    vb, _ = normalize_verb_token(token)
                    if vb.startswith('no_interaction') or vb not in self.meta['verb2id']:
                        continue
                    hois.append((sid, oid, self.meta['verb2id'][vb]))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
        return img, target


def make_myds_transforms(image_set):
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return [T.Compose([T.RandomHorizontalFlip(), T.ColorJitter(.4, .4, .4), T.RandomSelect(T.RandomResize(scales, max_size=1333), T.Compose([T.RandomResize([400, 500, 600]), T.RandomSizeCrop(384, 600), T.RandomResize(scales, max_size=1333)]))]), normalize]
    return [T.Compose([T.RandomResize([800], max_size=1333)]), normalize]


def build(image_set, args):
    root = Path(args.hoi_path)
    if image_set == 'train':
        anno = getattr(args, 'myds_train_anno', '') or (root / 'annotations' / 'train_20k.json')
    elif image_set == 'val':
        anno = getattr(args, 'myds_val_anno', '') or (root / 'annotations' / 'val.json')
    else:
        anno = getattr(args, 'myds_test_anno', '') or (root / 'annotations' / 'test.json')
    return MYDS(image_set, anno, make_myds_transforms(image_set), args.num_queries, args)
