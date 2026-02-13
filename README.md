# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
1. Load and Preprocess Data
2. Get the shape of the first image in the training dataset
3. Get the shape of the first image in the test dataset
4. Train the Model
5. Test the Model
6. Predict on a Single Image
7. Display the image  
## PROGRAM

### Name: Jeevika R
### Register Number: 212224040137

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return
  



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        
        
        
        print('Name:  Jeevika R      ')
        print('Register Number:   212224040137    ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="461" height="213" alt="image" src="https://github.com/user-attachments/assets/ed42c094-ee9f-4645-8b05-d3bfbbffd98a" />


## Confusion Matrix

<img width="893" height="765" alt="image" src="https://github.com/user-attachments/assets/4de487f1-8346-4a3c-9bb5-6519d1d2ed87" />


## Classification Report
<img width="637" height="428" alt="image" src="https://github.com/user-attachments/assets/5736255a-5c1b-4a3a-b70c-863fec49dd60" />


### New Sample Data Prediction
<img width="560" height="611" alt="image" src="https://github.com/user-attachments/assets/7e90680c-eac9-4761-a3d2-5fc65743b13f" />

## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
