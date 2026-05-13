from datasets.myds_meta import load_myds_meta


def build_myds_text_label_dicts(hoi_path):
    meta = load_myds_meta(hoi_path)
    hoi_text_label = {}
    for idx, pair in enumerate(meta['hoi_pairs']):
        hoi_text_label[pair] = meta['hoi_texts'][idx]
    obj_text_label = [(i, meta['obj_texts'][i]) for i in range(len(meta['obj_texts']))]
    return hoi_text_label, obj_text_label, meta
