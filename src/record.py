import torch 
import cv2 
from model import NeuralNetwork
import os
from loader import CustomDataset, clear_folder 


model = torch.load('best_model/full_model.pth', weights_only=False, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"] 
model.eval()
#Starts live camera
cap = cv2.VideoCapture(0) 

if not cap.isOpened(): 
    raise IOError("Cannot open webcam") 

#Keeps camera on till closed, not available on kaggle
while True:
    preds = []
    #Captures 60 or n frames
    for i in range(60): 
        ret, frame = cap.read() 

        if not ret: 
            break 
        filename = f"frame_{i:04d}.jpg" 
        frame_path = os.path.join('images', filename)
        cv2.imwrite(frame_path, frame) 
    #Loads data with custom dataset, no classes
    data = CustomDataset('images')
    dataloader = torch.utils.data.DataLoader(data, batch_size=60)

    with torch.no_grad(): 
        for images in dataloader: 
            images = images.to(device) 
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            preds.append(predicted)
    preds = torch.cat(preds)
    #Takes the emotion that go the most frames
    mode = torch.mode(preds, dim=0)
    #Dataset 'FER2013' is known to be imbalanced
    print(classes[mode[0].item()])
    #Clears images after every iteration
    clear_folder(images)

    