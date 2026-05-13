# myds → HOICLIP Adapter Implementation Brief

Derived from the RLIPv2 myds adaptation and adjusted for the HOICLIP repository.

This document translates the current **myds** integration behavior from the RLIPv2 adaptation into a practical adapter plan for a separate **HOICLIP** repository.

The target is a **plain pairwise HOI baseline on myds using HOICLIP**, not a group-aware, role-aware, QC-CEM, or L2CS version.

---

## 0) Scope and Design Principle

The first HOICLIP baseline should remain close to the original HOICLIP pipeline:

- Use **pairwise human-object interaction detection**.
- Use **base verbs only**.
- Ignore role suffixes such as `:target`, `:instrument`, `:support` during training and standard evaluation.
- Preserve person-person interactions.
- Preserve myds object vocabulary with custom contiguous IDs.
- Add myds-specific CLIP text labels and HOI combination labels.
- Avoid changing HICO/V-COCO behavior.
- Do not add group-aware loss, role-aware head, QC-CEM, or L2CS.

The expected first successful target is:

```text
HOICLIP trains and evaluates on myds with standard pairwise triplet mAP.
```

Not:

```text
HOICLIP supports group-aware, role-aware, subset-aware, QC-CEM, or L2CS evaluation.
```

---

## 1) Dataset Paths

RLIPv2 `datasets/my_dataset.py::build()` currently uses the following dataset root:

- **Dataset root**:  
  `/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds`

- **Image root base**:  
  `<root>/images`

- **Train annotation path**:  
  `<root>/annotations/train_20k.json`

- **Val annotation path**:  
  `<root>/annotations/val.json`

- **Test annotation path**:  
  `<root>/annotations/test.json`

Image resolution search logic in `MyHOIDataset` additionally probes, in order:

1. `<root>/images/train`
2. `<root>/images/test`
3. `<root>/images/val`
4. `<root>/images`
5. optional extra roots from `<root>/metadata/image_roots.txt`, one absolute path per line

For HOICLIP, use the same split paths by default:

```text
<hoi_path>/annotations/train_20k.json
<hoi_path>/annotations/val.json
<hoi_path>/annotations/test.json
```

Also add optional CLI overrides if easy:

```text
--myds_train_anno
--myds_val_anno
--myds_test_anno
```

HOICLIP already uses `--hoi_path` for dataset root in its HICO/V-COCO commands, so the preferred myds interface should keep:

```text
--dataset_file myds
--hoi_path /hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds
```

Do not hard-code myds paths inside HOICLIP if avoidable.

---

## 2) Final Annotation Schema

RLIPv2 consumes annotation JSON as a **top-level list** of per-image records.

### Required Per-Image Fields

- `entities`: list of entity dicts
- `interactions`: list of interaction dicts
- image identity through one of:
  - `image_path`, preferred if present
  - `file_name`
  - `image_id`

### Entity Item Schema

```json
{
  "entity_id": "p1",
  "category": "person",
  "bbox": [x1, y1, x2, y2]
}
```

### Interaction Item Schema

```json
{
  "subject_id": "p1",
  "object_id": "o1",
  "action_tokens": ["ride:support", "straddle:target"]
}
```

### Representative Complete Record

```json
{
  "image_id": "HICO_train2015_00000001.jpg",
  "entities": [
    {
      "entity_id": "p1",
      "category": "person",
      "bbox": [100, 120, 220, 400]
    },
    {
      "entity_id": "o1",
      "category": "motorcycle",
      "bbox": [180, 220, 420, 520]
    }
  ],
  "interactions": [
    {
      "subject_id": "p1",
      "object_id": "o1",
      "action_tokens": ["ride:support"]
    },
    {
      "subject_id": "p1",
      "object_id": "o1",
      "action_tokens": ["straddle:target"]
    }
  ]
}
```

Optional fields used by RLIPv2 evaluator or group paths may exist:

```text
group_interactions
group_annotations
groups
group_instances
```

For the first HOICLIP baseline, these optional group fields should be **preserved if present**, but **ignored during training and standard mAP evaluation**.

---

## 3) Entity Format

- **`entity_id` format**: string-like IDs, for example `"p1"`, `"o1"`.
- **`category` field**: object category string, must exist in myds object vocabulary.
- **`bbox` format**: 4 numbers in **xyxy**.
- **Coordinate convention**: `[x1, y1, x2, y2]`.
- **Absolute vs normalized**: absolute image pixels.
- **Image width/height source**: loaded directly from actual image using `PIL.Image.size`.
- **Clamping / validity during training**:
  - clamp boxes to `[0, W]` and `[0, H]`,
  - filter invalid boxes where `x2 <= x1` or `y2 <= y1`.

For HOICLIP adapter, preserve this exact **absolute xyxy** convention end-to-end.

Do not convert to COCO `xywh` in the myds loader.

---

## 4) Interaction Format

- `subject_id` references an entity by `entity_id`.
- `object_id` references an entity by `entity_id`.
- `action_tokens` is a list.
- Multiple entries in `action_tokens` are supported.
- Multiple interaction records can point to the same `(subject_id, object_id)` pair.

### `verb:role` Parsing

RLIPv2 myds loader uses full `verb:role` tokens as the label space.

For HOICLIP baseline, do **not** use full `verb:role` tokens as training classes.

Instead:

```text
ride:support      -> base verb: ride
cut:instrument    -> base verb: cut
watch:target      -> base verb: watch
```

The role suffix is ignored in this first HOICLIP baseline.

### Role Handling

- Role information is retained in the raw annotation.
- Role information is not used for HOICLIP classification loss.
- Role information is not used for standard triplet mAP.
- Role-aware metrics can be implemented later as a separate evaluator extension.

### Person-Person Interactions

Person-person interactions are allowed.

Example:

```json
{
  "subject_id": "p1",
  "object_id": "p2",
  "action_tokens": ["watch:target"]
}
```

If `object_id` points to a person entity, the object category is simply:

```text
person
```

Do not discard person-person interactions.

---

## 5) HOICLIP-Specific Label Spaces

HOICLIP differs from QPIC because it uses not only object labels and verb labels, but also **HOI combination labels** and **CLIP text-derived HOI classifiers**.

Therefore myds should expose three vocabularies:

```text
objects.txt
base_verbs.txt
hoi_classes.txt
```

Recommended metadata files:

```text
<hoi_path>/metadata/objects.txt
<hoi_path>/metadata/verb_roles.txt
<hoi_path>/metadata/base_verbs.txt
<hoi_path>/metadata/hoi_classes.txt
<hoi_path>/metadata/hoi_text_labels.txt
<hoi_path>/metadata/object_text_labels.txt
```

Only the first two may already exist. The others can be generated deterministically.

---

## 6) Base Verb Vocabulary

RLIPv2 uses full `verb:role` labels.

HOICLIP myds baseline must use **base verbs only**.

### Source

Preferred:

```text
<hoi_path>/metadata/base_verbs.txt
```

If this file does not exist, generate it from:

```text
<hoi_path>/metadata/verb_roles.txt
```

Generation rule:

1. read each non-empty line from `verb_roles.txt`;
2. normalize by `strip().lower()`;
3. take substring before `:`;
4. deduplicate while preserving first occurrence order;
5. write to `base_verbs.txt`.

Example:

```text
ride:support
ride:target
cut:target
cut:instrument
```

becomes:

```text
ride
cut
```

Use the same `base_verbs.txt` for train, val, test, and evaluation.

Do **not** infer a different verb order per split.

### Mapping

```python
verb2id = {verb: idx for idx, verb in enumerate(base_verbs)}
id2verb = {idx: verb for verb, idx in verb2id.items()}
num_verb_classes = len(base_verbs)
```

HOICLIP CLI should receive:

```text
--num_verb_classes <len(base_verbs)>
```

---

## 7) Object Vocabulary

### Source

Use:

```text
<hoi_path>/metadata/objects.txt
```

### Rules

- Object IDs are custom contiguous IDs.
- Do not assume COCO sparse category IDs.
- Do not assume HICO original object IDs.
- Ensure `"person"` is included.
- If `"person"` is missing, prepend it and rebuild the map.

### Mapping

```python
obj2id = {category: idx for idx, category in enumerate(objects)}
id2obj = {idx: category for category, idx in obj2id.items()}
num_obj_classes = len(objects)
```

HOICLIP CLI should receive:

```text
--num_obj_classes <len(objects)>
```

---

## 8) HOI Combination Vocabulary

HOICLIP's HICO implementation is built around HOI categories, not only independent verbs. HICO has 600 HOI categories derived from verb-object combinations. For myds, create an equivalent custom HOI combination space.

### Recommended File

```text
<hoi_path>/metadata/hoi_classes.txt
```

Each line should define one HOI combination:

```text
<base_verb>\t<object_category>
```

Example:

```text
ride	motorcycle
ride	bicycle
cut	apple
cut	knife
watch	person
sit	chair
```

### Recommended Deterministic Generation

For the first baseline, use one of the following two strategies.

#### Option A: Full Cross Product, Safest for Closed Vocabulary

Generate:

```text
base_verbs × objects
```

excluding only clearly invalid combinations if you explicitly maintain a validity filter.

Pros:

- No missing eval class due to unseen split combination.
- Works naturally with CLIP text prompts.
- Avoids split-dependent label space.

Cons:

- Larger `num_hoi_classes`.
- Many invalid combinations become negative labels.

#### Option B: Observed Combinations from Metadata or All Splits

Generate combinations observed in the curated dataset metadata or all released split annotations.

Pros:

- Smaller HOI head.
- Closer to HICO's fixed list of valid classes.

Cons:

- If generated from train only, val/test combinations may be missing.
- If generated from test annotations, document that this is a fixed benchmark vocabulary, not train-label leakage.

For the current thesis baseline, **Option A** is recommended if the number of objects and verbs is manageable. If it becomes too large, use **Option B from a fixed metadata file**, not dynamically from one split.

### Mapping

```python
hoi2id = {(verb_id, obj_id): hoi_id}
id2hoi = {hoi_id: (verb_id, obj_id)}
num_hoi_classes = len(hoi_classes)
```

In target construction, for each valid pair and base verb:

```python
obj_id = obj2id[object_category]
verb_id = verb2id[base_verb]
hoi_id = hoi2id[(verb_id, obj_id)]
```

Then set:

```python
verb_labels[pair_idx, verb_id] = 1
hoi_labels[pair_idx, hoi_id] = 1
```

This is important because HOICLIP uses both verb-level and HOI-combination-level information.

---

## 9) CLIP Text Labels and Prompts

HOICLIP uses CLIP text features for HOI classification. For myds, generate text labels for every HOI class.

### Recommended File

```text
<hoi_path>/metadata/hoi_text_labels.txt
```

Each line corresponds to the same index as `hoi_classes.txt`.

Example:

```text
a photo of a person riding a motorcycle
a photo of a person riding a bicycle
a photo of a person cutting an apple
a photo of a person cutting a knife
a photo of a person watching a person
a photo of a person sitting on a chair
```

### Prompt Generation Rules

For the first baseline, keep prompts simple and deterministic:

```python
prompt = f"a photo of a person {verb_ing} a {object_name}"
```

Need a small verb normalization helper:

```text
ride  -> riding
cut   -> cutting
sit   -> sitting
watch -> watching
hold  -> holding
```

If no reliable gerund conversion is implemented, use a simpler robust template:

```text
a photo of a person {verb} {object}
```

This is less grammatical but deterministic.

### Object CLIP Labels

If `--with_obj_clip_label` is used, also generate object prompts:

```text
a photo of a person
a photo of a bicycle
a photo of a motorcycle
a photo of a chair
```

Recommended file:

```text
<hoi_path>/metadata/object_text_labels.txt
```

---

## 10) HOICLIP Verb Representation Generation

HOICLIP includes a “visual semantic arithmetic” step for generating verb representation. The official workflow exposes this through:

```text
sh ./scripts/generate_verb.sh
```

and provides generated verb representation files such as:

```text
./tmp/verb.pth
./tmp/vcoco_verb.pth
```

For myds, there are two practical choices.

### Option A: Disable or Bypass Generated Verb Representation Initially

This is the simplest path if the current HOICLIP code allows running without precomputed verb representation.

Use this only if the code has a clean fallback.

### Option B: Add myds Verb Representation Generation

Recommended for faithful HOICLIP behavior.

Add:

```text
scripts/generate_verb_myds.sh
```

Expected output:

```text
tmp/myds_verb.pth
```

Generation should use:

```text
<hoi_path>/annotations/train_20k.json
<hoi_path>/metadata/base_verbs.txt
<hoi_path>/metadata/objects.txt
<hoi_path>/metadata/hoi_classes.txt
```

Important myds-specific changes:

- use base verb IDs, not full `verb:role`;
- use custom contiguous object IDs;
- use myds image path resolver;
- use union box of subject and object as HOI region if following HOICLIP's visual semantic arithmetic logic;
- handle person-person interactions normally.

If verb representation generation is too time-consuming or fragile, document it and start with HOI text classifier + supervised HOI labels first.

---

## 11) Pairwise HOI Conversion Logic for HOICLIP

The myds loader should convert `entities + interactions` into HOICLIP-compatible pairwise targets.

### Training Conversion

For each image:

1. Load image.
2. Build entity list in original order.
3. Build `entid2idx`.
4. Convert entity boxes to tensor `[M, 4]`.
5. Convert entity categories to object IDs using `obj2id`.
6. Clamp boxes and filter invalid boxes.
7. Apply HOICLIP transforms.
8. Track which original entity indices survived transform filtering.
9. Iterate over all interactions.
10. Skip interaction if:
    - `subject_id` not found,
    - `object_id` not found,
    - subject box was filtered out,
    - object box was filtered out.
11. For each action token:
    - normalize token with `strip().lower()`;
    - skip if token starts with `no_interaction`;
    - extract base verb before `:`;
    - skip if base verb not in `verb2id`;
    - get object category of `object_id`;
    - map object category to `obj_id`;
    - map `(verb_id, obj_id)` to `hoi_id`;
    - if no such HOI class exists, skip.
12. Aggregate duplicate `(subject, object)` pairs:
    - one pair row per unique subject-object pair;
    - `verb_labels` is multi-hot over base verbs;
    - `hoi_labels` is multi-hot over HOI combinations;
    - `obj_labels` is the object category ID of the object endpoint.
13. If no valid pair remains, return zero-length tensors with correct shapes.

### Important Behavior

- Subject box selection: from transformed boxes using kept subject index.
- Object box selection: from transformed boxes using kept object index.
- Multiple verbs on same pair: represented as multi-hot.
- Duplicate interactions: merged by pair key.
- Missing or invalid entities: skipped.
- No-object interactions: no dedicated null-object baseline.
- Person-person interactions: preserved, object category is `person`.
- Role suffix: ignored for class mapping.
- Group information: ignored in training target.

---

## 12) HOICLIP Training Target Format

The myds dataset `__getitem__` should return:

```python
img, target
```

where `target` contains at least:

```text
orig_size      Tensor[2], int64, [H, W]
size           Tensor[2], int64, [H, W]
filename       str
boxes          Tensor[M, 4], float32, transformed entity boxes when needed
labels         Tensor[M], int64, object class labels for entities when needed

obj_labels     Tensor[N], int64, object class per HOI pair
verb_labels    Tensor[N, V], float32, multi-hot base verb labels
hoi_labels     Tensor[N, H], float32, multi-hot HOI-combination labels
sub_boxes      Tensor[N, 4], float32, xyxy
obj_boxes      Tensor[N, 4], float32, xyxy

clip_inputs    Tensor, CLIP-preprocessed image input if HOICLIP requires it
```

Where:

```text
M = number of valid entities after filtering
N = number of unique valid subject-object pairs
V = number of base verbs
H = number of HOI combination classes
```

Recommended optional debug fields:

```text
sub_entity_ids
obj_entity_ids
obj_classes
verb_classes
hoi_classes
```

For empty targets:

```python
target["obj_labels"] = torch.zeros((0,), dtype=torch.int64)
target["verb_labels"] = torch.zeros((0, num_verb_classes), dtype=torch.float32)
target["hoi_labels"] = torch.zeros((0, num_hoi_classes), dtype=torch.float32)
target["sub_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
target["obj_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
```

---

## 13) Val/Test Target Format

For validation and test, return entity boxes and HOI annotations for evaluator.

Recommended target fields:

```text
filename
orig_size
size
boxes          Tensor[M, 4], xyxy
labels         Tensor[M], object IDs
hois           Tensor[K, 3], each row = [subject_entity_idx, object_entity_idx, verb_id]
hoi_ids        Tensor[K], optional, HOI combination ID
raw_hois       optional list for debugging
```

For each interaction token:

```text
subject_entity_idx
object_entity_idx
base_verb_id
```

If multiple action tokens exist for the same subject-object pair, create multiple rows in `hois`, one per base verb, unless duplicated.

The evaluator should evaluate by triplet:

```text
subject bbox + object bbox + base verb
```

For HOICLIP's HOI-combination output, evaluator may also use:

```text
subject bbox + object bbox + hoi_id
```

but final reported standard mAP should be interpretable as:

```text
<person, base verb, object category>
```

---

## 14) Image Path Handling

Reuse RLIPv2 image search behavior.

Resolution order:

1. If `image_path` exists and is absolute, use it directly.
2. If `image_path` exists and is relative, resolve under `<hoi_path>`.
3. Else use `file_name` or `image_id`.
4. Try:
   - `<hoi_path>/images/train/<name>`
   - `<hoi_path>/images/test/<name>`
   - `<hoi_path>/images/val/<name>`
   - `<hoi_path>/images/<name>`
5. Try extra roots from:
   - `<hoi_path>/metadata/image_roots.txt`

Error message should print all attempted paths.

This is important because myds may contain mixed image names from HICO-style, COCO-style, or custom sources.

---

## 15) Valid Object-Verb / HOI Matrix

RLIPv2 myds evaluator does not require a HICO-style `correct_mat`.

HOICLIP HICO code may expect a HICO-style correction matrix for HICO. For myds, do not force HICO's `corre_hico.npy`.

Instead, implement one of the following.

### Minimal Baseline

Do not use any correct-matrix filtering for myds.

All model outputs are evaluated directly.

### Optional myds Correction Matrix

Generate:

```text
<hoi_path>/metadata/corre_myds.npy
```

Recommended shape:

```text
[num_obj_classes, num_verb_classes]
```

Value:

```text
1 if object-verb pair is valid
0 otherwise
```

Generation options:

- from train annotations only, if used as a training prior;
- from fixed metadata, if used as benchmark validity;
- from all released vocabulary combinations, if using full cross product.

For the first HOICLIP baseline, the safest recommendation is:

```text
Do not require correct_mat integration.
Do not use HICO corre_hico.npy.
Do not mask myds predictions with HICO object-verb validity.
```

---

## 16) Evaluation Format

For myds, add a separate evaluator path instead of reusing HICO-specific assumptions.

### Ground-Truth Structure Consumed by myds Evaluator

Internal GT per image should become:

```python
{
  "filename": str,
  "annotations": [
    {
      "bbox": [x1, y1, x2, y2],
      "category_id": obj_id
    }
  ],
  "hoi_annotation": [
    {
      "subject_id": int,
      "object_id": int,
      "verb_id": int,
      "hoi_id": int,
      "object_id_label": int
    }
  ]
}
```

### Prediction Structure Consumed by Evaluator

Internal prediction per image should become:

```python
{
  "predictions": [
    {
      "bbox": [x1, y1, x2, y2],
      "category_id": obj_id,
      "score": float
    }
  ],
  "hoi_prediction": [
    {
      "subject_id": pred_subject_box_index,
      "object_id": pred_object_box_index,
      "verb_id": int,
      "hoi_id": int,
      "score": float
    }
  ]
}
```

### Matching Rule

A predicted HOI is a true positive if:

```text
subject IoU >= 0.5
object IoU >= 0.5
base verb matches
object category matches
```

For person-person interactions, object category is still:

```text
person
```

### mAP Computation

Implement standard pairwise triplet mAP:

```text
<subject/person, base verb, object category>
```

Default IoU:

```text
0.5
```

Recommended AP calculation:

```text
VOC2007 11-point AP
```

or keep the same AP style as the existing HOICLIP/HICO evaluator if simpler, but document it.

### Report

Minimum report:

```text
Triplet Full mAP
```

Optional:

```text
Rare mAP
Non-rare mAP
Mean max recall
```

Ignore in first baseline:

```text
group metrics
subset-aware metrics
role-aware metrics
QC-CEM metrics
L2CS metrics
```

---

## 17) Rare / Non-Rare Split

Rare/non-rare is optional for the first HOICLIP myds baseline.

If implemented, compute from train annotations:

```text
<hoi_path>/annotations/train_20k.json
```

Count triplets:

```text
(object_category, base_verb)
```

or:

```text
(subject_category=person, base_verb, object_category)
```

Recommended threshold:

```text
rare if count < 10
non-rare otherwise
```

Do not use HICO rare/non-rare lists.

Do not use HICO 600-class rare/non-rare indices.

---

## 18) Checkpoint Loading

HOICLIP pretrained HICO checkpoints may have incompatible classifier head shapes because myds uses different:

```text
num_obj_classes
num_verb_classes
num_hoi_classes
hoi text labels
object text labels
```

When `dataset_file == "myds"`, checkpoint loading should be tolerant.

### Load Compatible Parameters

Load:

```text
backbone parameters
transformer encoder/decoder parameters
query embeddings if shape-compatible
bbox heads if shape-compatible
CLIP-related frozen encoder parameters if present and compatible
interaction decoder parameters if shape-compatible
```

### Skip Incompatible Parameters

Skip:

```text
object classification head
verb classification head
HOI classification head
CLIP text classifier buffers tied to HICO classes
object CLIP label embeddings tied to HICO classes
HICO-specific hoi embedding / text label tensors
HICO-specific correction matrix dependent parameters
```

### Required Behavior

- Do not crash on classifier shape mismatch.
- Print skipped keys.
- Print loaded keys summary.
- Explicitly state that myds classification heads are randomly initialized or reinitialized from myds text prompts.

This is necessary because HOICLIP HICO configurations use HICO-specific object, verb, and HOI class spaces, while myds uses custom vocabulary sizes.

---

## 19) Existing HICO Behavior

Do not break existing HICO/V-COCO behavior.

All myds changes should be isolated behind:

```python
if args.dataset_file == "myds":
    ...
```

The original paths should remain:

```text
dataset_file == "hico"
dataset_file == "vcoco"
```

The official HICO/V-COCO scripts should continue to work.

---

## 20) Files to Inspect in HOICLIP

Before implementation, inspect at least:

```text
datasets/__init__.py
datasets/datasets_gen/hico.py
datasets/datasets_gen/vcoco.py
datasets/hico_text_label.py
datasets/vcoco_text_label.py

models/
models/models_hoiclip/
engine.py
main.py
scripts/train_hico.sh
scripts/train_vcoco.sh
scripts/generate_verb.sh
tools/convert_parameters.py
```

---

## 21) New Files to Add

Recommended minimal additions:

```text
datasets/datasets_gen/myds.py
datasets/myds_text_label.py
datasets/myds_meta.py
datasets/myds_eval.py
scripts/train_myds.sh
scripts/eval_myds.sh
scripts/generate_verb_myds.sh
tools/build_myds_metadata.py
```

### `datasets/datasets_gen/myds.py`

Responsibilities:

- parse myds `entities + interactions`;
- load object/base-verb/HOI vocabularies;
- resolve image paths;
- build HOICLIP-compatible training targets;
- produce `clip_inputs`;
- produce val/test `hois`.

### `datasets/myds_text_label.py`

Responsibilities:

- expose myds HOI text labels;
- expose object text labels if needed;
- replace HICO-specific `hico_text_label`.

Example structure:

```python
myds_text_label = {
    (verb_id, obj_id): "a photo of a person riding a motorcycle",
    ...
}

myds_hoi_text_label = {
    hoi_id: "a photo of a person riding a motorcycle",
    ...
}
```

Choose whichever format matches HOICLIP's internal expectation.

### `datasets/myds_meta.py`

Responsibilities:

- load `objects.txt`;
- load or generate `base_verbs.txt`;
- load or generate `hoi_classes.txt`;
- build `obj2id`, `verb2id`, `hoi2id`;
- normalize tokens.

### `datasets/myds_eval.py`

Responsibilities:

- evaluate standard pairwise triplet mAP;
- no HICO `correct_mat` dependency;
- optional rare/non-rare split.

### `tools/build_myds_metadata.py`

Responsibilities:

- generate missing metadata files;
- validate that all annotation categories exist in `objects.txt`;
- validate all base verbs exist in `base_verbs.txt`;
- generate `hoi_classes.txt`;
- generate `hoi_text_labels.txt`;
- optionally generate `corre_myds.npy`.

---

## 22) Dataset Registry Changes

In `datasets/__init__.py`, add:

```python
from .datasets_gen.myds import build as build_myds_gen
```

Then route:

```python
def build_dataset(image_set, args):
    if args.dataset_root == "GEN":
        if args.dataset_file == "hico":
            return build_hico_gen(image_set, args)
        if args.dataset_file == "vcoco":
            return build_vcoco_gen(image_set, args)
        if args.dataset_file == "myds":
            return build_myds_gen(image_set, args)

    ...
```

If HOICLIP uses a special `dataset_root == "GENERATE_VERB"` path for verb representation generation, add:

```python
from .datasets_generate_feature.myds import build as build_myds_generate_verb
```

and route:

```python
if args.dataset_root == "GENERATE_VERB":
    if args.dataset_file == "myds":
        return build_myds_generate_verb(image_set, args)
```

---

## 23) Main CLI Arguments

Add or reuse:

```text
--dataset_file myds
--hoi_path <path_to_myds>
--num_obj_classes <auto or explicit>
--num_verb_classes <auto or explicit>
--num_hoi_classes <auto or explicit if code needs it>

--myds_train_anno <optional>
--myds_val_anno <optional>
--myds_test_anno <optional>

--objects_path <optional>
--verb_roles_path <optional>
--base_verbs_path <optional>
--hoi_classes_path <optional>
--hoi_text_labels_path <optional>

--eval_train_json <optional>
--myds_no_correct_mat
```

Recommended default paths:

```text
objects_path         = <hoi_path>/metadata/objects.txt
verb_roles_path      = <hoi_path>/metadata/verb_roles.txt
base_verbs_path      = <hoi_path>/metadata/base_verbs.txt
hoi_classes_path     = <hoi_path>/metadata/hoi_classes.txt
hoi_text_labels_path = <hoi_path>/metadata/hoi_text_labels.txt
```

If possible, auto-fill:

```text
num_obj_classes = len(objects.txt)
num_verb_classes = len(base_verbs.txt)
num_hoi_classes = len(hoi_classes.txt)
```

This avoids manual mismatch.

---

## 24) HOICLIP Baseline Implementation Choices

For the first myds baseline, follow these constraints strictly:

1. Plain pairwise HOI only.
2. Use base verbs only.
3. Ignore role suffix in training.
4. Ignore role suffix in standard evaluation.
5. Preserve person-person interactions.
6. If `object_id` resolves to a person entity, map object category to `person`.
7. No group-aware loss.
8. No role-aware head.
9. No QC-CEM.
10. No L2CS.
11. No architecture rewrite beyond class-count / text-label / dataset changes.
12. No HICO-specific `correct_mat` requirement.
13. No HICO 600-class assumption.
14. No COCO sparse category ID assumption.
15. Keep all myds changes behind `dataset_file == "myds"`.

---

## 25) Training Command Pattern

Recommended script:

```bash
#!/bin/bash
set -eo pipefail

HOICLIP_DIR="/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/HOICLIP"
HOI_PATH="/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds"

cd "${HOICLIP_DIR}"

NUM_OBJ_CLASSES=$(awk 'NF{c++} END{print c+0}' "${HOI_PATH}/metadata/objects.txt")
NUM_VERB_CLASSES=$(awk 'NF{c++} END{print c+0}' "${HOI_PATH}/metadata/base_verbs.txt")
NUM_HOI_CLASSES=$(awk 'NF{c++} END{print c+0}' "${HOI_PATH}/metadata/hoi_classes.txt")

python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env \
  main.py \
  --dataset_file myds \
  --hoi_path "${HOI_PATH}" \
  --num_obj_classes "${NUM_OBJ_CLASSES}" \
  --num_verb_classes "${NUM_VERB_CLASSES}" \
  --num_hoi_classes "${NUM_HOI_CLASSES}" \
  --backbone resnet50 \
  --num_queries 64 \
  --dec_layers 3 \
  --with_clip_label \
  --with_obj_clip_label \
  --use_nms_filter \
  --output_dir logs/myds_hoiclip
```

Adjust additional HOICLIP-specific hyperparameters according to the existing `scripts/train_hico.sh`.

---

## 26) Evaluation Command Pattern

```bash
#!/bin/bash
set -eo pipefail

HOICLIP_DIR="/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/HOICLIP"
HOI_PATH="/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds"
CKPT="$1"

cd "${HOICLIP_DIR}"

NUM_OBJ_CLASSES=$(awk 'NF{c++} END{print c+0}' "${HOI_PATH}/metadata/objects.txt")
NUM_VERB_CLASSES=$(awk 'NF{c++} END{print c+0}' "${HOI_PATH}/metadata/base_verbs.txt")
NUM_HOI_CLASSES=$(awk 'NF{c++} END{print c+0}' "${HOI_PATH}/metadata/hoi_classes.txt")

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --use_env \
  main.py \
  --pretrained "${CKPT}" \
  --dataset_file myds \
  --hoi_path "${HOI_PATH}" \
  --num_obj_classes "${NUM_OBJ_CLASSES}" \
  --num_verb_classes "${NUM_VERB_CLASSES}" \
  --num_hoi_classes "${NUM_HOI_CLASSES}" \
  --backbone resnet50 \
  --num_queries 64 \
  --dec_layers 3 \
  --eval \
  --zero_shot_type default \
  --with_clip_label \
  --with_obj_clip_label \
  --use_nms_filter \
  --output_dir logs/myds_hoiclip_eval
```

If `training_free_enhancement_path` is used, ensure it is regenerated for myds HOI text labels, not reused from HICO.

---

## 27) Metadata Generation Checklist

Before training HOICLIP on myds, run a metadata validation step:

```text
[ ] objects.txt exists
[ ] person is included in objects.txt
[ ] verb_roles.txt exists
[ ] base_verbs.txt exists or can be generated
[ ] hoi_classes.txt exists or can be generated
[ ] hoi_text_labels.txt exists or can be generated
[ ] every entity category in train/val/test exists in objects.txt
[ ] every action token base verb exists in base_verbs.txt
[ ] every training verb-object pair maps to a valid HOI class
[ ] image paths resolve successfully
[ ] empty images / empty valid HOI pairs are handled safely
```

Recommended script:

```bash
python tools/build_myds_metadata.py \
  --hoi_path /hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds \
  --train_json annotations/train_20k.json \
  --val_json annotations/val.json \
  --test_json annotations/test.json \
  --generate_base_verbs \
  --generate_hoi_classes \
  --generate_text_labels \
  --validate
```

---

## 28) Expected Implementation Risks

### Risk 1: HOICLIP Assumes HICO 600 HOI Classes

Fix:

- add myds-specific `hoi_classes`;
- replace HICO text label lookup with myds text label lookup;
- do not import `hico_text_label` when `dataset_file == "myds"`.

### Risk 2: HICO `category_id` Uses COCO Sparse IDs

Fix:

- myds uses string categories;
- map strings through `objects.txt`;
- no COCO sparse IDs.

### Risk 3: HICO Correction Matrix Is Required

Fix:

- bypass `correct_mat` for myds;
- or generate `corre_myds.npy`;
- never load `corre_hico.npy` for myds.

### Risk 4: CLIP Text Label Count Mismatches Classifier Head

Fix:

- `hoi_text_labels.txt` length must equal `num_hoi_classes`;
- object text label length must equal `num_obj_classes`;
- checkpoint loader must skip HICO classifier tensors.

### Risk 5: Verb Representation File Is HICO-Specific

Fix:

- generate `tmp/myds_verb.pth`;
- or bypass the verb representation branch only if code supports this cleanly;
- never reuse `tmp/verb.pth` from HICO for myds.

### Risk 6: Test Combinations Missing from HOI Class Vocabulary

Fix:

- generate `hoi_classes.txt` from full fixed metadata, not per split;
- prefer full `base_verbs × objects` if manageable.

---

## 29) Minimal Recommendation for HOICLIP

To get a stable first baseline quickly:

1. Add `datasets/datasets_gen/myds.py`.
2. Add `datasets/myds_meta.py`.
3. Add `datasets/myds_text_label.py`.
4. Add `datasets/myds_eval.py`.
5. Add `dataset_file == "myds"` routing.
6. Generate:
   - `base_verbs.txt`
   - `hoi_classes.txt`
   - `hoi_text_labels.txt`
7. Use base verbs only.
8. Use pairwise triplet mAP only.
9. Ignore group/role/subset metrics.
10. Skip incompatible HICO classifier heads during checkpoint loading.
11. Preserve HICO/V-COCO behavior.

---

## 30) Notes on RLIPv2-Specific Extras Not Required in HOICLIP Baseline

The RLIPv2 myds code includes:

```text
group HOI generation
group evaluator hooks
subset metrics
role-aware normalization
additional analysis dumps
QC-CEM-related project logic
L2CS-related project logic
```

These are not required for the first HOICLIP baseline.

For HOICLIP, the extra implementation burden is instead:

```text
HOI combination vocabulary
CLIP HOI text prompts
object CLIP text prompts
myds-specific verb representation
checkpoint-compatible classifier loading
```

---

## 31) Implementation Priority

Recommended implementation order:

1. Metadata builder:
   - `base_verbs.txt`
   - `hoi_classes.txt`
   - `hoi_text_labels.txt`
   - `object_text_labels.txt`
2. Dataset loader:
   - parse entities/interactions
   - resolve images
   - build pairwise targets
3. Dataset registry:
   - route `dataset_file == "myds"`
4. Checkpoint loader:
   - tolerate HICO classifier mismatch
5. Evaluator:
   - pairwise triplet mAP at IoU 0.5
6. Train script:
   - single-node baseline first
7. Eval script:
   - checkpoint evaluation
8. Optional:
   - rare/non-rare split
   - myds correction matrix
   - myds verb representation generation

---

## 32) Summary

For QPIC, the main adaptation requirement is to convert myds into a pairwise DETR-style HOI target with base verbs.

For HOICLIP, the adaptation is stricter because the model additionally depends on:

- HOI combination classes;
- CLIP HOI text prompts;
- object text prompts;
- optional generated verb representations;
- classifier heads whose shapes depend on the dataset vocabulary.

Therefore, the HOICLIP myds adapter should not simply copy the QPIC adapter. It must add a myds-specific metadata and text-label layer while preserving the same plain pairwise HOI baseline constraints.

The final baseline should answer one question clearly:

```text
How well does standard HOICLIP perform on myds under conventional pairwise triplet mAP, using base verbs only?
```
