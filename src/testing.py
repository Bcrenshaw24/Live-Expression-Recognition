import torch 
from model import NeuralNetwork 
from data import eval_data 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = torch.load('best_model/full_model.pth', weights_only=False, map_location=device) 
model.to(device)

model.eval()
correct = 0
total = 0
test_data = 'face_data/test'
testloader = eval_data(test_data)
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = correct / total
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
