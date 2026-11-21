# Review of the supplied implementations

The uploaded source contained 12 files representing six distinct scripts. The other six uploads were exact byte-for-byte duplicates; see [`experiments/ORIGINAL_FILE_MAP.md`](../experiments/ORIGINAL_FILE_MAP.md).

## Experimental families

| Script family | Input pipeline | Input shape | Training setup | Notes |
|---|---|---|---|---|
| `banglamnistall.py`, `banglamnistp2.py` | Keras directory loader | 28×28 RGB | Batch 64, 30 epochs | Includes a 70/10/20 split by dataset batches; one version adds weighted precision, recall, F1 and a confusion matrix |
| `bengalimnist.py` | Keras directory loader | 28×28 RGB | Batch 64, 30 epochs | Earlier directory-based CNN variant |
| `bengalimnistp.py` | PIL TIFF loader | 28×28 grayscale | Batch 64, 100 epochs | Rotation/zoom augmentation and training-curve plots |
| `banglamnist2.py` | PIL TIFF loader | 28×28 grayscale | Batch 64, 100 epochs | Similar augmented TIFF experiment; test split logic is faulty |
| `banglamnist3.py` | PIL TIFF loader | 28×28 grayscale | Batch 64, 30 epochs | Augmentation includes horizontal and vertical flips; validation split is faulty |

## Issues corrected in the maintained implementation

1. **Machine-specific paths:** the original scripts point to absolute Windows directories. The maintained code uses command-line paths and a documented data layout.
2. **Mixed channel formats:** the directory-loader scripts use RGB while the TIFF scripts use grayscale. The maintained pipeline standardizes every image to 28×28 grayscale.
3. **Unstable split boundaries:** the directory-based scripts shuffle one dataset pipeline and then use `take()`/`skip()` to create train/validation/test datasets. Separate iterators over a shuffled upstream pipeline can produce inconsistent partitions. The new preparation step creates fixed, stratified file-level splits once and records them in a manifest.
4. **Ignored test data:** in `bengalimnistp.py` and `banglamnist2.py`, the return value of `load_images_and_labels(test_path)` is assigned to `test_size`, then the later `test_images`/`test_labels` are taken from the training array. Because the 90% and 10% slices consume the full training set, those resulting test arrays are empty. See the preserved original files for the exact code.
5. **Empty validation array:** in `banglamnist3.py`, `train_images` is truncated before `val_images` is sliced from it using the original `train_size` offset. This can leave validation data empty.
6. **Digit-changing augmentation:** horizontal and vertical flips appear in one experiment. They are not enabled in the maintained pipeline because flipping a handwritten digit can change or invalidate its class. The new model uses mild random rotation and zoom only.
7. **No saved model:** none of the uploaded scripts contains a model checkpoint save call. The maintained training script now saves the best and final Keras models, logs, and training curves.

## Scope of the new implementation

The new pipeline intentionally standardizes the data format and uses a reproducible stratified 80/10/10 train/validation/test split. This is a cleaned implementation inspired by the uploaded CNN experiments; it is not a claim that the original scripts used this exact split or that previous results have been reproduced.

No dataset, trained checkpoint, or verified numerical evaluation result was included with the uploaded scripts. Run the new training and evaluation commands before reporting metrics.
