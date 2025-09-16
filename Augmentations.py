import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_crop_resize = A.Compose([
    A.RandomCrop(width=180, height=224),
    A.Resize(224, 224),
    ToTensorV2()
])

transform_crop_rotate_pad = A.Compose([
    A.RandomCrop(width=180, height=224),
    A.Rotate(limit=25, p=1),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
    ToTensorV2()
])

transform_vertical_flip = A.Compose([
    A.Resize(224, 224),
    A.VerticalFlip(p=1),
    ToTensorV2()
])

transform_random_gamma = A.Compose([
    A.Resize(224, 224),
    A.RandomGamma(gamma_limit=(70, 130), p=1),
    ToTensorV2()
])

list1=[transform_crop_resize, transform_crop_rotate_pad, transform_vertical_flip, transform_random_gamma]

import matplotlib.pyplot as plt
import numpy as np

def plot_image(image_array):
    if isinstance(image_array, np.ndarray):
        image_to_plot = image_array
    else:
        image_to_plot = image_array.numpy()

    if image_to_plot.shape[0] == 3: 
        image_to_plot = np.transpose(image_to_plot, (1, 2, 0))  
    
    plt.imshow(image_to_plot)
    plt.axis('off') 
    plt.show()


import torch

def overlay_mask_on_image(frame_img, mask_img, alpha=0.5, cmap='Reds'):

    if isinstance(frame_img, torch.Tensor):
        frame_img = frame_img.numpy()
    if isinstance(mask_img, torch.Tensor):
        mask_img = mask_img.numpy()

    if frame_img.shape[0] == 3:
        frame_img = np.transpose(frame_img, (1, 2, 0))
    if mask_img.shape[0] == 3:
        mask_img = np.transpose(mask_img, (1, 2, 0))
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)

    plt.imshow(frame_img.astype(np.uint8))
    plt.imshow(mask_img, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.show()



import os
import numpy as np
import cv2

encoder_directory=r"C:\Users\AnirudhVijan\Desktop\study\MedicalImaging\project\data\encoder_training"
decoder_directory=r"C:\Users\AnirudhVijan\Desktop\study\MedicalImaging\project\data\decoder_training"


encoder_directory_new="encoder_directory_new"
decoder_directory_new="decoder_directory_new"

os.makedirs(encoder_directory_new, exist_ok=True )
os.makedirs(decoder_directory_new, exist_ok=True)

masks=[]
for files  in os.listdir(decoder_directory):
    masks.append(files)

masks.sort()


frames=[]
for files  in os.listdir(encoder_directory):
    frames.append(files)

frames.sort()
count=0
for  i in range(len(frames)):
    mask=masks[i]
    frame=frames[i]

    mask_path=os.path.join(decoder_directory, mask)
    frame_path=os.path.join(encoder_directory, frame)

    bool1 = np.random.randint(0, 2)

    if(bool1):
        aug_helper=np.random.randint(0, 3)
        # aug_helper=3
        # print(aug_helper)
        augmentation_used=list1[aug_helper]

        frame_image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        mask_image = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        # print(mask_image.shape)
        # plot_image(frame_image)
        
        augmented = augmentation_used(image=frame_image, mask=mask_image)

        framed_image = augmented['image']
        masked_image = augmented['mask']
    
        
        masked_image = masked_image.numpy()
        framed_image = framed_image.permute(1, 2, 0).cpu().numpy()
        print(masked_image.shape)


        # Save images in the new directory with unique filenames
        masked_image_filename = f"mask_{count}.png"
        framed_image_filename = f"framed_{count}.png"

        mask_save_path = os.path.join(decoder_directory_new, masked_image_filename)
        framed_image_save_path = os.path.join(encoder_directory_new, framed_image_filename)

        cv2.imwrite(mask_save_path, masked_image)
        cv2.imwrite(framed_image_save_path, framed_image)

        count += 1

        print(f"Saved masked image and framed image as {masked_image_filename} and {framed_image_filename}")
        masked_image_filename = f"mask_{count}.png"
        framed_image_filename = f"framed_{count}.png"

        mask_save_path = os.path.join(decoder_directory_new, masked_image_filename)
        framed_image_save_path = os.path.join(encoder_directory_new, framed_image_filename)

        cv2.imwrite(mask_save_path, mask_image)
        cv2.imwrite(framed_image_save_path, frame_image)

        count += 1

        # print(f"Saved mask image and frame image as {masked_image_filename} and {framed_image_filename}")
    else:
        frame_image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        mask_image = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        masked_image_filename = f"mask_{count}.png"
        framed_image_filename = f"framed_{count}.png"

        mask_save_path = os.path.join(decoder_directory_new, masked_image_filename)
        framed_image_save_path = os.path.join(encoder_directory_new, framed_image_filename)

        cv2.imwrite(mask_save_path, mask_image)
        cv2.imwrite(framed_image_save_path, frame_image)

        count += 1

        print(f"Saved mask image and frame image as {masked_image_filename} and {framed_image_filename}")

print(count)