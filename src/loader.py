from torch.utils.data import Dataset
from PIL import Image 
from torchvision import transforms 
import shutil
import os

#Dataset is like an abstract class/interface
#Must implement methods
#Pytorch's Dataset requires class folders in data source
class CustomDataset(Dataset): 
    def __init__(self, folder): 
        self.paths = [folder + '/' + f for f in os.listdir(folder)] 
        #Important to resize live feed and grayscale it before it goes to the model
        self.transform = transforms.Compose([transforms.Resize((48, 48), interpolation=transforms.InterpolationMode.BILINEAR), 
          transforms.Grayscale(num_output_channels=1),                                  
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))]) 
    def __len__(self): 
        return len(self.paths)
    def __getitem__(self, index):
        image = Image.open(self.paths[index]) 
        return self.transform(image) 

def clear_folder(path): 
    if os.path.exists(path): 
        shutil.rmtree(path, ignore_errors=True) 