import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import Tk, Text, Scrollbar, END, Frame, Label, Entry, Button, StringVar, IntVar, DoubleVar
import tkinter.messagebox as messagebox
import os

data = pd.read_csv("Date.txt")
data_i = pd.read_csv("Date.txt")

data['year'] = pd.to_datetime(data['dt']).dt.year
 
max_year = data['year'].max()
data = data[data['year'] >= max_year - 100]

scaler = MinMaxScaler()
data['AverageTemperature'] = scaler.fit_transform(data[['AverageTemperature']])

epochs_list = []
rmse_list = []

stop_training = False

def stop_training_function():
    global stop_training
    stop_training = True
 
def start():
 global stop_training
 stop_training = False
 global epochs_list, rmse_list
 epochs_list = []
 rmse_list = []
 global model
 
 def create_sequences(data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            sequences.append(seq)
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)



 def sigmoid(x):
    return 1 / (1 + np.exp(-x))

 def tanh(x):
    return np.tanh(x)

 def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

 def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

 def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))


 class LSTM:
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
 
        self.W_f, self.W_i, self.W_c, self.W_o = [], [], [], []
        self.b_f, self.b_i, self.b_c, self.b_o = [], [], [], []

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.W_f.append(np.random.randn(hidden_size, layer_input_size + hidden_size) * 0.01)
            self.b_f.append(np.zeros((hidden_size, 1)))

            self.W_i.append(np.random.randn(hidden_size, layer_input_size + hidden_size) * 0.01)
            self.b_i.append(np.zeros((hidden_size, 1)))

            self.W_c.append(np.random.randn(hidden_size, layer_input_size + hidden_size) * 0.01)
            self.b_c.append(np.zeros((hidden_size, 1)))

            self.W_o.append(np.random.randn(hidden_size, layer_input_size + hidden_size) * 0.01)
            self.b_o.append(np.zeros((hidden_size, 1)))
 
        self.h_prev = [np.zeros((hidden_size, 1)) for _ in range(num_layers)]
        self.C_prev = [np.zeros((hidden_size, 1)) for _ in range(num_layers)]
        self.cache = []

    def forward(self, x_t):
     if len(x_t.shape) == 1:
        x_t = x_t.reshape(-1, 1)
     if x_t.shape != (self.input_size, 1):
        raise ValueError(f"x_t shape mismatch: Expected ({self.input_size}, 1), got {x_t.shape}")

     step_cache = []

     h_t_list = []
     C_t_list = []

     input_layer = x_t
     for layer in range(self.num_layers): 
        combined = np.vstack((self.h_prev[layer], input_layer))
 
        f_t = sigmoid(np.dot(self.W_f[layer], combined) + self.b_f[layer])
        i_t = sigmoid(np.dot(self.W_i[layer], combined) + self.b_i[layer])
        C_tilde = tanh(np.dot(self.W_c[layer], combined) + self.b_c[layer])
        o_t = sigmoid(np.dot(self.W_o[layer], combined) + self.b_o[layer])
 
        C_t = f_t * self.C_prev[layer] + i_t * C_tilde
        h_t = o_t * tanh(C_t)
 
        self.h_prev[layer] = h_t
        self.C_prev[layer] = C_t
 
        input_layer = h_t
        h_t_list.append(h_t)
        C_t_list.append(C_t)
 
        layer_cache = {
            "f_t": f_t,
            "i_t": i_t,
            "C_tilde": C_tilde,
            "o_t": o_t,
            "combined": combined,
            "C_t": C_t,
            "h_t": h_t,
            "h_prev": self.h_prev[layer].copy(),
            "C_prev": self.C_prev[layer].copy()
        }
        step_cache.append(layer_cache)

     self.cache = step_cache

     return h_t_list[-1], C_t_list[-1]





    def backward(self, X_seq, Y_seq, sequence_cache):
     d_h_next = np.zeros((self.hidden_size, 1))
     self.gradients = [ 
        {
            "W_f": np.zeros_like(self.W_f[layer]),
            "b_f": np.zeros_like(self.b_f[layer]),
            "W_i": np.zeros_like(self.W_i[layer]),
            "b_i": np.zeros_like(self.b_i[layer]),
            "W_c": np.zeros_like(self.W_c[layer]),
            "b_c": np.zeros_like(self.b_c[layer]),
            "W_o": np.zeros_like(self.W_o[layer]),
            "b_o": np.zeros_like(self.b_o[layer]),
        }
        for layer in range(self.num_layers)
      ]

     for t in reversed(range(len(X_seq))):
        x_t = X_seq[t].reshape(-1, 1)
        y_t = Y_seq[t].reshape(-1, 1)
        cache_t = sequence_cache[t]
        d_loss = 2 * (cache_t[-1]["h_t"] - y_t)
        
        input_layer = x_t
        d_h = d_loss + d_h_next

        for layer in reversed(range(self.num_layers)):
            cache = sequence_cache[t][layer]

            h_t = cache["h_t"]
            C_t = cache["C_t"]
            h_prev = cache["h_prev"]
            C_prev = cache["C_prev"]
            f_t = cache["f_t"]
            i_t = cache["i_t"]
            C_tilde = cache["C_tilde"]
            o_t = cache["o_t"]
            combined = cache["combined"]

            d_o = d_h * tanh(C_t) * sigmoid_derivative(o_t)
            d_C = d_h * o_t * tanh_derivative(C_t)
            d_f = d_C * C_prev * sigmoid_derivative(f_t)
            d_i = d_C * C_tilde * sigmoid_derivative(i_t)
            d_C_tilde = d_C * i_t * tanh_derivative(C_tilde)

            dW_o = np.dot(d_o, combined.T)
            db_o = d_o
            dW_f = np.dot(d_f, combined.T)
            db_f = d_f
            dW_i = np.dot(d_i, combined.T)
            db_i = d_i
            dW_c = np.dot(d_C_tilde, combined.T)
            db_c = d_C_tilde

            self.gradients[layer]["W_f"] += dW_f
            self.gradients[layer]["b_f"] += db_f
            self.gradients[layer]["W_i"] += dW_i
            self.gradients[layer]["b_i"] += db_i
            self.gradients[layer]["W_c"] += dW_c
            self.gradients[layer]["b_c"] += db_c
            self.gradients[layer]["W_o"] += dW_o
            self.gradients[layer]["b_o"] += db_o

            d_combined = (
                np.dot(self.W_o[layer].T, d_o) +
                np.dot(self.W_f[layer].T, d_f) +
                np.dot(self.W_i[layer].T, d_i) +
                np.dot(self.W_c[layer].T, d_C_tilde)
            )

            d_h = d_combined[:self.hidden_size, :]
            input_layer = d_combined[self.hidden_size:, :]
        
        d_h_next = d_h 



    def update_weights(self):
     for layer in range(self.num_layers):
        grads = self.gradients[layer]
         
        self.W_f[layer] -= self.learning_rate * grads["W_f"]
        self.b_f[layer] -= self.learning_rate * grads["b_f"]
        
        self.W_i[layer] -= self.learning_rate * grads["W_i"]
        self.b_i[layer] -= self.learning_rate * grads["b_i"]
        
        self.W_c[layer] -= self.learning_rate * grads["W_c"]
        self.b_c[layer] -= self.learning_rate * grads["b_c"]
        
        self.W_o[layer] -= self.learning_rate * grads["W_o"]
        self.b_o[layer] -= self.learning_rate * grads["b_o"]



 input_size = int(input_entry.get())
 hidden_size = int(hidden_entry.get())
 num_layers = int(nr_hidden_entry.get())
 output_size = int(output_entry.get())
 learning_rate = float(learn_entry.get())
 num_epochs = int(ephoch_entry.get())
 admited_error = float(error_entry.get())

 data_values = data[['AverageTemperature']].values
 X, y = create_sequences(data[['AverageTemperature']].values, input_size)
 train_size = int(len(X) * 0.8)
 X_train, y_train = X[:train_size], y[:train_size]
 X_test, y_test = X[train_size:], y[train_size:]
 
 #X_train = np.array(X_train, dtype=np.float32)
 #y_train = np.array(y_train, dtype=np.float32)
 #X_test = np.array(X_test, dtype=np.float32)
 #y_test = np.array(y_test, dtype=np.float32)
 
 X_train = [x.reshape(-1, 1).astype(np.float32) for x in X_train]
 y_train = [y.reshape(-1, 1).astype(np.float32) for y in y_train]
 X_test = [x.reshape(-1, 1).astype(np.float32) for x in X_test]
 y_test = [y.reshape(-1, 1).astype(np.float32) for y in y_test]
 
 model = LSTM(input_size, hidden_size, num_layers, 1, learning_rate)
 model.h_prev = [np.zeros((hidden_size, 1)) for _ in range(num_layers)]
 model.C_prev = [np.zeros((hidden_size, 1)) for _ in range(num_layers)]

 def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

 for epoch in range(num_epochs):
 
    sequence_cache = []
    loss = 0
    for t in range(len(X_train)):
       x_t = X_train[t]
       y_t = y_train[t]
       h_t, C_t = model.forward(x_t)
       sequence_cache.append(model.cache)
       loss += mse_loss(y_t, h_t)
    model.backward(X_train, y_train, sequence_cache)
    model.update_weights()

    rmse = np.sqrt(loss / len(X_train))
    print(f"Epoch {epoch + 1}, RMSE: {rmse}")


    epochs_list.append(epoch + 1)
    rmse_list.append(rmse)

    ax.clear()
    ax.set_title("Grafic eroare")
    ax.set_xlabel("Epoci")
    ax.set_ylabel("RMSE")
    ax.grid()
    ax.plot(epochs_list, rmse_list, label="RMSE", color="blue", marker="o")
    ax.legend()

    canvas.draw() 
    root.update_idletasks()

    if rmse <= admited_error:
        messagebox.showinfo("Training Stopped", "The admitted error was reached!") 
        break
    
 current_seq = X_test[0].copy()
 predictions = []

 for _ in range(len(X_test)):
    y_pred, _ = model.forward(current_seq)

    # Extrage scalarul din vector/matrice
    if isinstance(y_pred, np.ndarray):
        if y_pred.shape == (1, 1):
            y_scalar = y_pred[0, 0]
        else:
            y_scalar = y_pred.flatten()[0]
    else:
        y_scalar = float(y_pred)

    predictions.append(y_scalar)

    # Actualizează secvența
    current_seq = np.vstack((current_seq[1:], [[y_scalar]]))

 # Transformă în vectori și inversează scalarea
 predictions = np.array(predictions).reshape(-1, 1)
 y_test = np.array(y_test).reshape(-1, 1)

 predictions = scaler.inverse_transform(predictions).flatten()
 y_test = scaler.inverse_transform(y_test).flatten()

 
 plt.figure(figsize=(10, 6))
 plt.plot(y_test, label="Valori reale")
 plt.plot(predictions, label="Predicții")
 plt.legend()
 plt.title("Predicții prezise vs Valori reale")
 plt.xlabel("Timp")
 plt.ylabel("Temperatura medie (°C)")
 plt.show()

 
 def predict_future(model, data, seq_length, num_months, scaler):
    predictions = []
    input_seq = data[-seq_length:]

    for _ in range(num_months):
        input_tensor = np.array(input_seq).reshape(-1, 1)
        h_t, C_t = model.forward(input_tensor)

        
        pred = h_t.flatten()[0]
        predictions.append(pred)
        
        next_input = np.array([pred]).reshape(-1, 1)
        input_seq = np.vstack((input_seq[1:], next_input))

    predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros_like(predictions))))[:, 0]
    #predictions += 4
    return predictions
 
 num_future_months = 12
 future_predictions = predict_future(
    model=model,
    data=data_values,
    seq_length=input_size,
    num_months=num_future_months,
    scaler=scaler
 )
 
 months = ["Ianuarie", "Februarie", "Martie", "Aprilie", "Mai", "Iunie",
          "Iulie", "August", "Septembrie", "Octombrie", "Noiembrie", "Decembrie"]
 
 fig, axy = plt.subplots(figsize=(8, 4))
  
 
 table_data = list(zip(months, future_predictions))
 table = axy.table(cellText=table_data, colLabels=["Luna", "Temperatura Medie (°C)"], loc='center')
 table.auto_set_font_size(False)
 table.set_fontsize(12)
 table.scale(1, 1.5)


 plt.title("Predicții pentru Temperatura Medie (2025)", fontsize=14)
 plt.show()
 
 plt.figure(figsize=(10, 5))
 plt.bar(months, future_predictions, color='skyblue')
 plt.xlabel('Luna')
 plt.ylabel('Temperatura Medie Prezisă (°C)')
 plt.title('Temperatura Medie Prezisă pe Lună (2025)')
 plt.xticks(rotation=45)
 plt.grid(axis='y')
 plt.show()

def save_weights(model, filename="lstm_weights"):
    weights = {
        "W_f": model.W_f, "b_f": model.b_f,
        "W_i": model.W_i, "b_i": model.b_i,
        "W_c": model.W_c, "b_c": model.b_c,
        "W_o": model.W_o, "b_o": model.b_o
    }
    np.savez(filename, **weights)
    messagebox.showinfo("Saved", "Model weights saved successfully!")

def load_weights(model, filename="lstm_weights"):
    if not os.path.exists(filename):
        messagebox.showwarning("No Weights", "No saved weights found!")
        return
    
    weights = np.load(filename)
    model.W_f, model.b_f = weights["W_f"], weights["b_f"]
    model.W_i, model.b_i = weights["W_i"], weights["b_i"]
    model.W_c, model.b_c = weights["W_c"], weights["b_c"]
    model.W_o, model.b_o = weights["W_o"], weights["b_o"]

    messagebox.showinfo("Loaded", "Model weights loaded successfully!")

def delete_weights(filename="lstm_weights"):
    if os.path.exists(filename):
        os.remove(filename)
        messagebox.showinfo("Deleted", "Model weights deleted successfully!")
    else:
        messagebox.showwarning("Not Found", "No saved weights to delete!")
 
root = Tk()
root.title("Data Display with Tabs")


font_label = ("Arial", 12)
font_entry = ("Arial", 10)

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)

notebook.add(tab1, text='Date de intrare')
notebook.add(tab2, text='Date de intrare normalizate')
notebook.add(tab3, text='Antrenare')


def insert_data(tab, data_to_display):
    text_widget = Text(tab, wrap='none')
    text_widget.pack(expand=True, fill='both')

    scrollbar_y = Scrollbar(tab, command=text_widget.yview)
    scrollbar_y.pack(side='right', fill='y')
    
    scrollbar_x = Scrollbar(tab, command=text_widget.xview, orient='horizontal')
    scrollbar_x.pack(side='top', fill='x')

    text_widget.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    text_widget.insert(END, str(data_to_display))

insert_data(tab1, data_i)
insert_data(tab2, data) 

frame_controls = Frame(tab3)
frame_controls.pack(side='top', fill='x')

frame_graph = Frame(tab3)
frame_graph.pack(side='bottom', fill='both', expand=True)

fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Errors Grafic")
ax.set_xlabel("Epoci")
ax.set_ylabel("RMSE")
ax.grid()

canvas = FigureCanvasTkAgg(fig, master=frame_graph)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(expand=True, fill='both')

input_label=tk.Label(frame_controls, text="Number of neurons on input layer:",font=font_label, bg="#f0f0f5").grid(row=1, column=0,padx=5, pady=10)
input_entry=tk.Entry(frame_controls, font=font_entry )
input_entry.grid(row=1, column=1)
input_entry.insert(0, 12)

hidden_label=tk.Label(frame_controls, text="Number of hidden layer's:",font=font_label, bg="#f0f0f5").grid(row=2, column=0,padx=5, pady=10)
hidden_entry=tk.Entry(frame_controls, font=font_entry )
hidden_entry.grid(row=2, column=1)
hidden_entry.insert(0, "2")

nr_hidden_label=tk.Label(frame_controls, text="Number of neurons on hidden layer's:",font=font_label, bg="#f0f0f5").grid(row=3, column=0,padx=5, pady=10)
nr_hidden_entry=tk.Entry(frame_controls, font=font_entry )
nr_hidden_entry.grid(row=3, column=1)
nr_hidden_entry.insert(0,"64")

output_label=tk.Label(frame_controls, text="Number of neurons on output layer:",font=font_label, bg="#f0f0f5").grid(row=4, column=0,padx=5, pady=10)
output_entry=tk.Entry(frame_controls, font=font_entry )
output_entry.grid(row=4, column=1)
output_entry.insert(0, "1")

learn_label=tk.Label(frame_controls, text="Learning rate:",font=font_label, bg="#f0f0f5").grid(row=5, column=0)
learn_entry=tk.Entry(frame_controls, font=font_entry)
learn_entry.grid(row=5, column=1)
learn_entry.insert(0,"0.001")

#activare_label=tk.Label(frame_controls, text="Activare:",font=font_label, bg="#f0f0f5").grid(row=2, column=2)
#dropdown_activare = ttk.Combobox(frame_controls, values=["sigmoid","tanh","relu"], font=font_entry)
#dropdown_activare.grid(row=2, column=3)
#dropdown_activare.set("sigmoid")

error_label=tk.Label(frame_controls, text="Admited error:",font=font_label, bg="#f0f0f5").grid(row=1, column=5)
error_entry=tk.Entry(frame_controls, font=font_entry)
error_entry.grid(row=1, column=6)
error_entry.insert(0, "0.01")

ephoch_label=tk.Label(frame_controls, text="Ephocs:",font=font_label, bg="#f0f0f5").grid(row=6, column=0)
ephoch_entry=tk.Entry(frame_controls, font=font_entry)
ephoch_entry.grid(row=6, column=1)
ephoch_entry.insert(0, "100")

button_start = tk.Button(frame_controls, text="Start", font=font_label, bg="#a3c2c2", fg="black", command=lambda: start())
button_start.grid(row=3, column=4, pady=10)

button_save = tk.Button(frame_controls, text="Save Model", font=font_label, bg="#a3c2c2", fg="black",
                         command=lambda: save_weights(model))
button_save.grid(row=3, column=5, pady=10)

button_load = tk.Button(frame_controls, text="Load Model", font=font_label, bg="#a3c2c2", fg="black",
                         command=lambda: load_weights(model))
button_load.grid(row=3, column=6, pady=10)

button_delete = tk.Button(frame_controls, text="Delete Model", font=font_label, bg="#a3c2c2", fg="black",
                           command=lambda: delete_weights())
button_delete.grid(row=3, column=7, pady=10)


root.mainloop()