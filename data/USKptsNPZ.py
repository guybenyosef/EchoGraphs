import torch
import os
import numpy as np
import cv2
import albumentations as A
from PIL import Image
import glob
from typing import List, Dict

# internal:
from data.USKpts import USKpts

class USKptsNPZ(USKpts):
    """ Implements USKpts base dataset class for reading 2ch data provided by the GEHC Haifa team. """
    def __init__(self, dataset_config, filenames_list: str = None, transform: A.core.composition.Compose = None):
        super().__init__(dataset_config, filenames_list, transform)

    def create_img_list(self, filenames_list: str) -> None:
        """ Creates a list containing paths to frames in the dataset."""
        self.filenames_list = filenames_list

        img_list_from_file = []
        if filenames_list is not None:
            if not type(filenames_list)==list:  # (if single element then put in list)
                filenames_list = [filenames_list]

            for filenames_sublist in filenames_list:
                if os.path.exists(filenames_sublist):  # (filenames_sublist is file)
                    with open(filenames_sublist) as f:
                        img_list_from_file.extend(f.read().splitlines())
                else:   # (filenames_sublist is a case_name)
                    img_list_from_file.append(filenames_sublist)

        self.img_list = []
        if len(img_list_from_file) > 0:
            for f in img_list_from_file:
                fullpath = os.path.join(self.img_folder, f)
                if os.path.exists(fullpath):
                    if os.path.isdir(fullpath): # f is a case_name
                        to_add = glob.glob(os.path.join(fullpath, "*.png"))
                        to_add = [f.replace(self.img_folder, "") for f in to_add]
                        self.img_list.extend(to_add)
                    else:   # fullpath is a single frame
                        self.img_list.append(f)
        else:
            img_list_from_folder = glob.glob(os.path.join(self.img_folder, "**/*.png"))
            self.img_list = [os.path.basename(f) for f in img_list_from_folder]

    def load_kpts_annotations(self, img_list: List) -> np.ndarray:
        """ Creates an array of annotated keypoints coorinates for the frames in the dataset. """
        KP_COORDS = []
        if self.anno_dir is not None:
            for fname in img_list:
                kpts = np.load(os.path.join(self.anno_dir, fname.replace("png", "npz")), allow_pickle=True)["kpts"]
                KP_COORDS.append(kpts)
            KP_COORDS = np.array(KP_COORDS).swapaxes(0, 1)
        return KP_COORDS

    def img_to_torch(self, img: np.ndarray) -> torch.Tensor:
        """ Convert original image format to torch.Tensor """
        # resize:
        if img.shape[0] != self.input_size:
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR_EXACT)
        # transform:
        img = Image.fromarray(np.uint8(img)).convert("RGB")
        img = self.basic_transform(img)

        return img
