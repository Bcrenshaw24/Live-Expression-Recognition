import torch 
import torch.nn as nn
import torch.optim as optim 
import os
from data import load_train
from model import NeuralNetwork
os.makedirs('models', exist_ok=True)

model = NeuralNetwork()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device)
data = 'face_data/train'
dataloader = load_train(data) 

#Use at least 30 for this dataset 
epochs = 40

#For Classification and Backpropagation 
criterion = nn.CrossEntropyLoss() 

#Small lr for Adam compared to SGD (e.g. 0.01)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999)) 

classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"] 

n_total_steps = len(dataloader) 
best_val_loss = float('inf') 
for epoch in range(epochs): 
    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0
    for i, (images, labels) in enumerate(dataloader): 

        # Size: [Batch_size, 1, 48, 48], 
        images = images.to(device) 
        labels = labels.to(device) 

        #Forward Pass 
        outputs = model(images) 
        loss = criterion(outputs, labels) 
        #Accuracy
        _, predicted = torch.max(outputs, 1)
        epoch_correct += (predicted == labels).sum().item()
        epoch_total += labels.size(0)
        epoch_loss += loss.item()

   
        #Backward and optimization
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
    epoch_accuracy = epoch_correct / epoch_total
    avg_loss = epoch_loss / n_total_steps

    print(f" epoch {epoch+1} of {epochs} total epochs Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy*100:.2f}") 
           
    







