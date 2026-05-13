# Run HOICLIP on myds

## 1) Build/validate metadata
```bash
python tools/build_myds_metadata.py \
  --hoi_path /path/to/myds \
  --train_json annotations/train_20k.json \
  --val_json annotations/val.json \
  --test_json annotations/test.json \
  --generate_base_verbs \
  --generate_hoi_classes \
  --generate_text_labels \
  --validate
```

## 2) Optional verb representation
Use dataset-specific verb feature (do not reuse HICO `tmp/verb.pth`), e.g. `tmp/myds_verb.pth`.

## 3) Train smoke test
```bash
python main.py \
  --model_name HOICLIP \
  --dataset_file myds \
  --hoi_path /path/to/myds \
  --num_obj_classes <N_OBJ> \
  --num_verb_classes <N_VERB> \
  --num_queries 64 --dec_layers 3 \
  --with_clip_label --with_obj_clip_label \
  --batch_size 2 --epochs 1 --output_dir logs/hoiclip_myds_smoke
```

## 4) Eval smoke test
```bash
python main.py \
  --eval \
  --model_name HOICLIP \
  --pretrained logs/hoiclip_myds_smoke/checkpoint_last.pth \
  --dataset_file myds \
  --hoi_path /path/to/myds \
  --num_obj_classes <N_OBJ> \
  --num_verb_classes <N_VERB> \
  --num_queries 64 --dec_layers 3 \
  --with_clip_label --with_obj_clip_label \
  --output_dir logs/hoiclip_myds_eval
```

Notes:
- `verb:role` tokens are reduced to base verb (`verb`) for training/eval.
- `no_interaction*` tokens are ignored.
- Person-person interactions are kept and evaluated as normal pairwise triplets.
