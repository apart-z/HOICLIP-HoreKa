#!/usr/bin/env python3
import argparse
from pathlib import Path
from datasets.myds_meta import load_myds_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hoi_path', required=True)
    ap.add_argument('--train_json', default='annotations/train_20k.json')
    ap.add_argument('--val_json', default='annotations/val.json')
    ap.add_argument('--test_json', default='annotations/test.json')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--generate_base_verbs', action='store_true')
    ap.add_argument('--generate_hoi_classes', action='store_true')
    ap.add_argument('--generate_text_labels', action='store_true')
    args = ap.parse_args()

    root = Path(args.hoi_path)
    for rel in [args.train_json, args.val_json, args.test_json]:
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f'missing annotation file: {p}')
    meta = load_myds_meta(root)
    print('objects:', len(meta['objects']))
    print('verbs:', len(meta['verbs']))
    print('hoi classes:', len(meta['hoi_pairs']))
    print('correct_mat:', meta['correct_mat_path'])


if __name__ == '__main__':
    main()
