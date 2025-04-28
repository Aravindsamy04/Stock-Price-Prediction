# Stock-Price-Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.

## Design Steps

### Step 1:
Import necessary libraries.

### Step 2:
Load and preprocess the data.

### Step 3:
Create input-output sequences.

### Step 4: 
Convert data to PyTorch tensors.

### Step 5:
Define the RNN model.

### Step 6:
Train the model using the training data.

### Step 7:
Evaluate the model and plot predictions.

## Program
#### Name:
#### Register Number:
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1,hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  def forward(self, x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out





model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the Model

criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=20
model.train()
train_losses=[]
for epoch in range(epochs):
  epoch_loss=0
  for x_batch,y_batch in train_loader:
    x_batch,y_batch=x_batch.to(device),y_batch.to(device)
    optimizer.zero_grad()
    outputs=model(x_batch)
    loss=criterion(outputs,y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()
  train_losses.append(epoch_loss/len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_losses[-1]:.4f}")






```

## Output

### True Stock Price, Predicted Stock Price vs time

![Screenshot 2025-04-28 114919](https://github.com/user-attachments/assets/34dbba3e-4cf5-4ccb-ad4f-bb2d71477e57)

### Predictions 

![Screenshot 2025-04-28 114955](https://github.com/user-attachments/assets/0a8c3a16-8f17-4061-8128-e1137c9ca6c9)

## Result
The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.


