# Original Code Review

## `urdhumnist.py` — TensorFlow character classifier

Observed in the supplied source:
- Loads `characters_train_set` and `characters_test_set`.
- Uses `image_dataset_from_directory`.
- Resizes images to 28 × 28; default directory-loader color mode is RGB.
- Uses 40 output classes, batch size 32, and 20 epochs.
- Scales image values by dividing by 255.
- Uses a CNN with 32, 64, 128, and 256 convolution filters, batch normalization, pooling, dropout, and a 512-unit dense layer.

Issues addressed in the refactor:
- Replaces machine-specific absolute paths with CLI arguments.
- Creates a fixed validation split from the training directory.
- Enforces the same class-name order for the test directory.
- Saves model and class metadata.
- Adds early stopping and best-model checkpointing.

Note: the original script does not explicitly record the class-name list or save model weights.

## `urdu_pytorch.py` — PyTorch digit classifier

Observed in the supplied source:
- Loads `digits_train_set` and `digits_test_set` using `torchvision.datasets.ImageFolder`.
- Resizes images to 28 × 28 and converts them to tensors.
- Defines 10 output classes, batch size 64, and 30 epochs.
- Uses a CNN with 32, 64, 128, and 256 convolution channels.
- Calculates accuracy, confusion matrix, precision, recall, and F1-score.

Important architecture issue:
- The forward method applies max pooling after each of four convolutional layers. With 28 × 28 input, the spatial size becomes 14 → 7 → 3 → 1.
- However, the fully connected layer expects `256 * 7 * 7` features.
- The refactored version performs pooling after the second and fourth convolutional layers, leaving a 7 × 7 feature map before flattening.

Other changes:
- Adds a stratified validation split.
- Verifies training and test class mappings.
- Saves the best checkpoint and class names.
- Adds independent evaluation and prediction scripts.

## Performance and dataset limitations

The two supplied source files do not include model checkpoints, complete training logs, or the actual datasets. Therefore, this review does not claim any test accuracy or benchmark result.
