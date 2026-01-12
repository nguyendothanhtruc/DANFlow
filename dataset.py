import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTECDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.input_size = input_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train


    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        #deal with grayscale images
        image = Image.open(image_file).convert('RGB')
        
        
        if self.is_train:
            image = self.image_transform(image)
            return image
        else:
            image = self.image_transform(image)
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)

class AUGMENTEDMVTECDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
                
        if category == "grid":
            self.image_transform_augment = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            
        elif category in ["carpet", "tile", "leather"]:
            self.image_transform_augment = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(0, translate= (0.05, 0.05))]), p = 0.5),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees = 90)]), p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            
        elif category == "wood":
            self.image_transform_augment = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(2, translate= (0.05, 0.05))]), p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif category == "metal_nut":
            self.image_transform_augment = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees = 90)]), p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif category == "pill":
            self.image_transform_augment = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(3, translate= (0.1, 0.1), scale=(0.98, 1.02))]), p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif category == "transistor":
            self.image_transform_augment = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            
        elif category == "hazelnut":
            self.image_transform_augment = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees = 90)]), p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            
        else:
            self.image_transform_augment = self.image_transform
        

        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        #deal with grayscale images
        image = Image.open(image_file).convert('RGB')
        
        
        if self.is_train:
            image = self.image_transform_augment(image)
            return image
        else:
            image = self.image_transform(image)
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)
#####################################################################

class VISADataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.input_size = input_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.JPG")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.JPG"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train


    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        #deal with grayscale images
        image = Image.open(image_file).convert('RGB')
        
        
        if self.is_train:
            image = self.image_transform(image)
            return image
        else:
            image = self.image_transform(image)
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(".JPG", ".png")
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)

#####################################################################

class BTADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size = (input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            if category == "02":
                self.image_files = glob(
                os.path.join(root, category, "train", "ok", "*.png")
                )
            else:
                self.image_files = glob(
                    os.path.join(root, category, "train", "ok", "*.bmp")
                )
        else:
            if category == "02":
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.bmp"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(size = (input_size, input_size)),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        #deal with grayscale images
        image = Image.open(image_file).convert('RGB')
        
        image = self.image_transform(image)
        if self.is_train:
            #print("Img shape: ", image.shape)
            return image
        else:
            if os.path.dirname(image_file).endswith("ok"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                if "/01/test/" in image_file:
                    target = Image.open(
                        image_file.replace("/test/", "/ground_truth/").replace(".bmp", ".png"))
                else:
                    target = Image.open(
                        image_file.replace("/test/", "/ground_truth/"))
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)

#####################################################################

class KLTSDDDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.input_size = input_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "ok", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        #deal with grayscale images
        image = Image.open(image_file).convert('RGB')
        if self.is_train:
            image = self.image_transform(image)
            return image
        else:
            image = self.image_transform(image)
            if os.path.dirname(image_file).endswith("ok"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/")
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)