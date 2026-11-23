# Review of the supplied implementation

The supplied archive contained 12 CNN training scripts (`covid.py`, `covid1.py`, `covid2.py`, and `covid4.py` through `covid12.py`) plus three augmentation/data-generation utilities (`datasen.py`, `imagegen.py`, and `imagegen1.py`). The experiments vary image size (224, 256, 299), batch size, CNN depth, dropout, augmentation, class weights, and learning-rate callbacks.

## Issues identified

1. **Class-count mismatch.** The supplied directory layout contains four classes (`COVID`, `Lung_Opacity`, `Normal`, `Viral Pneumonia`), but some scripts end with three or two output units. Those models cannot correctly represent the supplied four-way task.
2. **Inconsistent file assumptions.** One script searches for `*.jpg`, while all 4,032 supplied images are `.png`.
3. **Label/path mismatch.** One script uses `Viral_Pneumonia` while the folder is named `Viral Pneumonia`, and also relies on `os.path` without importing `os`.
4. **Unstable batch-level split.** Several scripts create train/validation/test sets using `take()` and `skip()` after shuffling a batched dataset. This does not provide a persisted file-level split manifest and can undermine reproducibility when reshuffling occurs.
5. **Validation-only evaluation in one variant.** `covid12.py` uses the validation subset for its final evaluation and does not create an independent test split.
6. **Augmentation policy.** The original augmentation helpers include horizontal flipping. The cleaned pipeline deliberately does not apply flips by default because they reverse anatomical laterality.
7. **Class-weight leakage risk.** `covid5.py` calculates class weights using labels gathered from the full dataset before splitting. The cleaned implementation derives weights from training records only.
8. **No saved model evidence.** The archive contains scripts and images but no trained checkpoint, training history, or final evaluation report.

## New implementation choices

- Uses the four class folders discovered in the supplied archive, with class names stored in metadata.
- Scans image file paths and removes exact duplicate content before splitting; identical content under conflicting labels raises an error.
- Performs deterministic, stratified, file-level train/validation/test splitting and saves a manifest.
- Applies modest rotation/zoom only, with no image flips.
- Calculates class weights using the training split only.
- Saves the best model, final model, training history, split manifest, and class-name mapping.
- Keeps prediction output explicitly experimental and non-diagnostic.

## Dataset inventory from the supplied archive

| Class folder | Raw PNG files | Exact duplicates removed | Unique files |
|---|---:|---:|---:|
| COVID | 1,008 | 12 | 996 |
| Lung_Opacity | 1,008 | 0 | 1,008 |
| Normal | 1,008 | 0 | 1,008 |
| Viral Pneumonia | 1,008 | 5 | 1,003 |
| **Total** | **4,032** | **17** | **4,015** |

The images sampled during inspection were 299×299 pixels; nearly all were grayscale, with a small number of RGB images in the Viral Pneumonia folder. The training pipeline converts all images consistently to three-channel tensors.

## Important limitations

The supplied archive does not include patient IDs or reliable study-level grouping metadata. The split is therefore stratified at the image-file level, not patient level. If the source provider supplies patient/study identifiers, use group-wise splitting before making generalization claims. Exact duplicate removal does not prevent leakage between different images from the same patient.
