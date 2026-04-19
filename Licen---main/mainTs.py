import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
 
data = pd.read_csv("Date.txt")
data['dt'] = pd.to_datetime(data['dt'])
 
cutoff_date = data['dt'].max() - pd.DateOffset(years=100)
data = data[data['dt'] >= cutoff_date]
 
data['AverageTemperature'].fillna(method='ffill', inplace=True)
 
scaler = MinMaxScaler()
data['AverageTemperature'] = scaler.fit_transform(data[['AverageTemperature']])
 
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

sequence_length = 12
data_values = data['AverageTemperature'].values
X, y = create_sequences(data_values, sequence_length)
 
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
 
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (batch, seq_len, 1)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test)
 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
 
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.004
num_epochs = 400

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
 
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
    train_predictions = model(X_train).squeeze()
    train_loss = criterion(train_predictions, y_train)
    print(f'Train Loss (MSE): {train_loss.item():.4f}')
 
predictions_np = scaler.inverse_transform(predictions.numpy().reshape(-1, 1)).flatten()
y_test_np = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Valori reale')
plt.plot(predictions_np, label='Predicții')
plt.legend()
plt.show()
 
def predict_future(model, data, seq_length, num_months, scaler):
    model.eval()
    predictions = []
    input_seq = data[-seq_length:]

    for _ in range(num_months):
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(input_tensor).squeeze().item()
        predictions.append(pred)
        input_seq = np.append(input_seq[1:], pred)
 
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions

future_predictions = predict_future(model, data_values, sequence_length, 12, scaler)

months = ["Ianuarie", "Februarie", "Martie", "Aprilie", "Mai", "Iunie",
          "Iulie", "August", "Septembrie", "Octombrie", "Noiembrie", "Decembrie"]

prediction_table = pd.DataFrame({
    "Luna": months,
    "Temperatura Medie Prezisă (°C)": future_predictions
})

print(prediction_table)
