import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms
import glob
from PIL import Image

d = Path(__file__).resolve().parents[1]

def resizedImages(input_path, save_path):
     
    print("Resizing starting.....")
    for name in os.listdir(input_path):
        img_name = name
        img = cv2.imread(os.path.join(input_path, img_name))

        targetSize  = 720
        img_resized = cv2.resize(img, (720, 540));

        cv2.imwrite(os.path.join(save_path, img_name), img_resized)
    
    print("Resizing End.....")

def plot(loss, title, losses):
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("# epoch")
    plt.ylabel(loss)
    plt.title(title)
    plt.savefig(str(d) + '/CMP-719-Project/plots/' + title + '.png')
    plt.close()


#plot two graphs in one figure
def multiPlot(acc1, acc2, title, ylabel):
    fig = plt.figure()
    plt.plot(acc1, label="Train")
    plt.plot(acc2, label="Validation")
    plt.xlabel("# epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(str(d) + '/CMP-719-Project/plots/' + title + '.png')
    plt.close()


def Data_Augmentation():

    path_g = "./data/train_aug/GT"
    path_hz = "./data/train_aug/hazy"
    length = glob.glob(path_g + "/*.png")

    count = len(length)
    print("Augmentation starting.....")
    for name in os.listdir(path_g):
        count += 1
        splitted = name.split("_")
        img_clear = Image.open(os.path.join(path_g, name))
        img_hazy = Image.open(os.path.join(path_hz, splitted[0] + "_hazy.png"))

        hazy = transforms.ToTensor()(img_hazy)
        gt = transforms.ToTensor()(img_clear)

        horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        vertical_flip = transforms.RandomVerticalFlip(p=1)
        crop = transforms.RandomCrop(size=(224,312))
        rotate =  transforms.RandomRotation(degrees=66)

        hf_hazy = horizontal_flip(hazy)
        hf_gt = horizontal_flip(gt)

        hf_hazy = transforms.ToPILImage()(hf_hazy)
        hf_gt = transforms.ToPILImage()(hf_gt)

        hf_hazy.save(os.path.join(path_hz, str(count) + "_hazy.png"))
        hf_gt.save(os.path.join(path_g,  str(count) + "_GT.png"))

        count += 1

        vf_hazy = vertical_flip(hazy)
        vf_gt = vertical_flip(gt)

        vf_hazy = transforms.ToPILImage()(vf_hazy)
        vf_gt = transforms.ToPILImage()(vf_gt)

        vf_hazy.save(os.path.join(path_hz, str(count) + "_hazy.png"))
        vf_gt.save(os.path.join(path_g,  str(count) + "_GT.png"))

        count += 1

        c_hazy = crop(hazy)
        c_gt = crop(gt)

        c_hazy = transforms.ToPILImage()(c_hazy)
        c_gt = transforms.ToPILImage()(c_gt)

        c_hazy.save(os.path.join(path_hz, str(count) + "_hazy.png"))
        c_gt.save(os.path.join(path_g,  str(count) + "_GT.png"))

        count += 1

        r_hazy = rotate(hazy)
        r_gt = rotate(gt)

        r_hazy = transforms.ToPILImage()(r_hazy)
        r_gt = transforms.ToPILImage()(r_gt)

        r_hazy.save(os.path.join(path_hz, str(count) + "_hazy.png"))
        r_gt.save(os.path.join(path_g,  str(count) + "_GT.png"))



        



if __name__ == "__main__":
    
    Data_Augmentation()

