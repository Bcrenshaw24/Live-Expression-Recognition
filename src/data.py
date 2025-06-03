import torch
from torchvision import transforms, datasets

def load_train(data):
    data_dir = data
    #Adds more randomness to photos to reduce overfitting
    transform = transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),                  
        transforms.RandomRotation(20),                      
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.Grayscale(num_output_channels=1),        
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(data_dir, transform=transform) 
    #This can cause heavy CPU and GPU strain
    #If not using kaggle/colab lower num of workers to 1 and half batch size
    #CPU may bottleneck GPU
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=300, shuffle=True, pin_memory=True, num_workers=4) 

    return dataloader   

def eval_data(data): 
    #Keeps images consistent
    transform = transforms.Compose (
        [ transforms.Grayscale(num_output_channels=1),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))]
    ) 
    test_data = datasets.ImageFolder(data, transform=transform) 

    dataloader = torch.utils.data.DataLoader(test_data, batch_size=128) 

    return dataloader 


