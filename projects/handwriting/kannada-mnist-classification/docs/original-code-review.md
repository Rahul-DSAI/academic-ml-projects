# Review of the supplied Kannada-MNIST files

## Files received

- `kmnist2.py`: 20 epochs, batch size 128, 10 classes; compact CNN with Adam.
- `Kmnist.py`: 50 epochs, batch size 128; CNN with 32/64/128/256 convolutional channels and an exponential learning-rate scheduler.
- `kmnist1.py`: 30 epochs, batch size 32; deeper CNN experiment with 128/256/512 channels and a 1,024-unit dense layer; scheduler callback is commented out.
- `KMNIST.ipynb`: notebook containing several iterations, data augmentation experiments, an NPZ download cell, plus unrelated GUI/LLM examples.

## Dataset facts visible in the notebook

The notebook output reports 60,000 training samples and 10,000 test samples. It downloads four NPZ files:
`X_kannada_MNIST_train.npz`, `X_kannada_MNIST_test.npz`, `y_kannada_MNIST_train.npz`, and `y_kannada_MNIST_test.npz`. The scripts load the single array stored under `arr_0`.

## Methodology issues to be aware of

1. The supplied notebook and scripts pass the official test split directly as `validation_data` during training. Therefore, the logged `val_accuracy` values are not untouched final test metrics; repeated model/augmentation tuning on that split can bias the reported performance.
2. Several notebook experiments use horizontal and/or vertical flips. Flipping handwritten numerals can alter class identity or create unrealistic examples. The cleaned pipeline intentionally omits flips and does not add aggressive augmentation by default.
3. The notebook contains multiple experimental runs. Their configurations and outputs should not be combined into a single claimed benchmark.
4. Notebook cell 6 is a separate PyQt image-generation utility and uses a `COVID` filename prefix; it is not part of Kannada digit recognition.
5. Notebook cell 8 is a separate Streamlit/Gemini chat example, with an incomplete API-key configuration (`os.getenv` is passed without reading an environment variable). It is not part of the digit classifier.
6. Notebook cell 9 shows a failed installation attempt for the `dotenv` package. The separate chatbot example is not packaged here.

## What the cleaned implementation changes

- Removes local Windows-only file paths.
- Loads the four NPZ arrays from a configurable data directory and supports `arr_0` archives.
- Validates image shapes, labels and image/label counts.
- Splits only the training data into train and validation partitions using a stratified, seeded split.
- Keeps the official test set out of training and validation; evaluate it once after selecting the model.
- Uses a compact CNN inspired by the supplied convolutional experiments.
- Saves the best checkpoint and training history, and generates a test classification report and confusion matrix.

No new training run was performed while creating this package. The supplied notebook output is retained as historical evidence, not treated as a clean held-out benchmark.
