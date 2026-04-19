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


data = pd.read_csv("Date.txt")
data_i = pd.read_csv("Date.txt")

scaler = MinMaxScaler()
data['AverageTemperature'] = scaler.fit_transform(data[['AverageTemperature']])

epochs_list = []
rmse_list = []


 
def start():
 global epochs_list, rmse_list
 epochs_list = []
 rmse_list = []
 
 def create_sequences(data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            sequences.append(seq)
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)


 sequence_length = 12
 data_values = data[['AverageTemperature']].values
 X, y = create_sequences(data_values, sequence_length)


 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


 X_train = np.array(X_train, dtype=np.float32)
 y_train = np.array(y_train, dtype=np.float32)
 X_test = np.array(X_test, dtype=np.float32)
 y_test = np.array(y_test, dtype=np.float32)


 def sigmoid(x):
    return 1 / (1 + np.exp(-x))

 def tanh(x):
    return np.tanh(x)

 def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

 def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


 class LSTM:
    def __init__(self, input_size, hidden_size, num_layers, output_size,learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
         
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
         
        self.h_prev = np.zeros((hidden_size, 1))
        self.C_prev = np.zeros((hidden_size, 1))

    def forward(self, x_t):
       if len(x_t.shape) == 1:
        x_t = x_t.reshape(-1, 1)

       if x_t.shape[0] != self.input_size:
        raise ValueError(f"x_t shape mismatch: Expected ({self.input_size}, 1), got {x_t.shape}")
    
       combined = np.vstack((self.h_prev, x_t))
    
       if combined.shape[0] != self.input_size + self.hidden_size:
        raise ValueError(f"Combined shape mismatch: Expected ({self.input_size + self.hidden_size}, 1), got {combined.shape}")
    
     
       f_t = sigmoid(np.dot(self.W_f, combined) + self.b_f)
       i_t = sigmoid(np.dot(self.W_i, combined) + self.b_i)
       C_tilde = tanh(np.dot(self.W_c, combined) + self.b_c)
       o_t = sigmoid(np.dot(self.W_o, combined) + self.b_o)
     
       C_t = f_t * self.C_prev + i_t * C_tilde
       h_t = o_t * tanh(C_t)
    
       self.h_prev = h_t
       self.C_prev = C_t
    
       return h_t, C_t, f_t, i_t, o_t, C_tilde



    def backward(self, x_t, y_t, h_t, C_t, f_t, i_t, o_t, C_tilde, d_loss):
        combined = np.vstack((self.h_prev, x_t))
         
        d_o = d_loss * tanh(C_t) * sigmoid_derivative(h_t)
        d_C_t = d_loss * o_t * tanh_derivative(C_t) + f_t * self.C_prev
        
        d_f = d_C_t * self.C_prev * sigmoid_derivative(f_t)
        d_i = d_C_t * C_tilde * sigmoid_derivative(i_t)
        d_C_tilde = d_C_t * i_t * tanh_derivative(C_tilde)
         
        dW_o = np.dot(d_o, combined.T)
        db_o = d_o
        dW_f = np.dot(d_f, combined.T)
        db_f = d_f
        dW_i = np.dot(d_i, combined.T)
        db_i = d_i
        dW_c = np.dot(d_C_tilde, combined.T)
        db_c = d_C_tilde
      
        self.gradients = {
            "W_o": dW_o, "b_o": db_o,
            "W_f": dW_f, "b_f": db_f,
            "W_i": dW_i, "b_i": db_i,
            "W_c": dW_c, "b_c": db_c,
        }

    def update_weights(self): 
        for param in ["W_o", "b_o", "W_f", "b_f", "W_i", "b_i", "W_c", "b_c"]:
            grad = self.gradients[param]
            setattr(self, param, getattr(self, param) - self.learning_rate * grad)



 input_size = int(input_entry.get())
 hidden_size = int(hidden_entry.get())
 num_layers = int(nr_hidden_entry.get())
 output_size = int(output_entry.get())
 learning_rate = float(learn_entry.get())
 num_epochs = int(ephoch_entry.get())
 admited_error = float(error_entry.get())
 
 model = LSTM(input_size, hidden_size, num_layers, output_size,learning_rate)
 def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

 for epoch in range(num_epochs):
     
    
    loss = 0
    model.h_prev = np.zeros((model.hidden_size, 1))
    model.C_prev = np.zeros((model.hidden_size, 1))

    for t in range(len(X_train)):
        x_t = X_train[t].reshape(-1, 1)
        y_t = y_train[t].reshape(-1, 1)

        h_t, C_t, f_t, i_t, o_t, C_tilde = model.forward(x_t)

 
        d_loss = 2 * (h_t - y_t) 
        loss += mse_loss(y_t, h_t)
 
        model.backward(x_t, y_t, h_t, C_t, f_t, i_t, o_t, C_tilde, d_loss)
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
        messagebox.showinfo("Training Stopped", "The admitted error was reached!")  # Pop-up
        return



 
 predictions = []
 for t in range(len(X_test)):
    x_t = X_test[t].reshape(-1, 1)
    h_t, C_t, f_t, i_t, o_t, C_tilde = model.forward(x_t)
    predictions.append(h_t.flatten())

 predictions = np.array(predictions)
 predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros_like(predictions))))[:, 0]
 y_test = scaler.inverse_transform(np.column_stack((y_test, np.zeros_like(y_test))))[:, 0]
 
 plt.figure(figsize=(10, 6))
 plt.plot(y_test, label='Valori reale')
 plt.plot(predictions, label='Predicții')
 plt.legend()
 plt.show()
 
 def predict_future(model, data, seq_length, num_months, scaler):
    predictions = []
    input_seq = data[-seq_length:]

    for _ in range(num_months):
        input_tensor = np.array(input_seq).reshape(-1, 1)
        h_t, C_t, f_t, i_t, o_t, C_tilde = model.forward(input_tensor)
        
        pred = h_t.flatten()[0]
        predictions.append(pred)
        
        next_input = np.array([pred]).reshape(-1, 1)
        input_seq = np.vstack((input_seq[1:], next_input))

    predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros_like(predictions))))[:, 0]
    return predictions

 
 num_future_months = 48 
 future_predictions = predict_future(
    model=model,
    data=data_values,
    seq_length=sequence_length,
    num_months=num_future_months,
    scaler=scaler
 )
 
 plt.figure(figsize=(12, 6))
 plt.plot(range(1, num_future_months + 1), future_predictions, label='Predicții viitoare')
 plt.xlabel('Luna')
 plt.ylabel('Temperatura (de-normalizată)')
 plt.legend()
 plt.show()

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
ax.set_title("Evoluția RMSE în funcție de numărul de epoci")
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

#button_stop = tk.Button(frame_controls, text="Stop", font=font_label, bg="#a3c2c2", fg="black", command=lambda: stop_training_function() )
#button_stop.grid(row=3, column=6, pady=10)

root.mainloop()