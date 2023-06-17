import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import glob



class Custom_Dataset(data.Dataset):
    def __init__(self, path_g, path_hz, mode='train', batch_size=1  ):
        super(Custom_Dataset, self).__init__()
        self.path_g = path_g
        self.path_hz = path_hz
        self.mode = mode
        self.batch_size = batch_size
        self.mode = mode
    
    def __getitem__(self, index):
        
        num = "0"
        if index < 9:
            num = "0" + str(index + 1)
        else:
            num = str(index + 1)

        img_clear = Image.open(os.path.join(self.path_g, num + '_GT.png'))
        img_hazy = Image.open(os.path.join(self.path_hz, num + '_hazy.png'))
        

        hazy = transforms.ToTensor()(img_hazy)
        gt = transforms.ToTensor()(img_clear)

        return gt, hazy

    def __len__(self):
        trainList = glob.glob(self.path_g + "/*.png")
        return len(trainList)
    
    

DenseHaze_train_loader = DataLoader(dataset=Custom_Dataset(mode='train', path_g="./data/train_aug/GT", path_hz="./data/train_aug/hazy"), batch_size=1, shuffle=True)
DenseHaze_test_loader = DataLoader(dataset=Custom_Dataset(mode='test', path_g="./data/test/GT", path_hz="./data/test/hazy"), batch_size=1, shuffle=False)


        
    
       