import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define constants
batch_size = 64
num_classes = 10
epochs = 30

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Load dataset
trainset = torchvision.datasets.ImageFolder(root="D:\\Streamlit\\data\\digits_train_set", transform=transform)
testset = torchvision.datasets.ImageFolder(root="D:\\Streamlit\\data\\digits_test_set", transform=transform)

# Data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.batchnorm2(x)
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.batchnorm3(x)
        x = self.dropout2(x)
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.batchnorm4(x)
        x = self.dropout2(x)
        x = x.view(-1, 256 * 7 * 7)  # Corrected view size
        x = nn.functional.relu(self.fc1(x))
        x = self.batchnorm5(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# Instantiate the model
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters())

# Training the model
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

print('Finished Training')

# Evaluating the model
net.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.numpy())
        true_labels.extend(labels.numpy())

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate precision, recall, and F1-score
report = classification_report(true_labels, predicted_labels, target_names=[str(i) for i in range(num_classes)])

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Print precision, recall, and F1-score
print("\nClassification Report:")
print(report)
