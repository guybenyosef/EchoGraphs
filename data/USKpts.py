import torch
from torch.utils import data
import torchvision.transforms as torch_transforms
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict
import albumentations as A

# internal:
from utils.utils_plot import draw_kpts, plot_kpts_pred_and_gt
from utils.utils_files import ultrasound_img_load

########################################################################
########################################################################
#     Base Dataset class for Ultrasound Keypoints detection models
########################################################################
########################################################################
class USKpts(data.Dataset):
    """
    Base class for reading and parsing ultrasound image and keypoints data.
    The only assumption we made here is: frames are read from image files in img_folder,
    while annotations (if exists) are read from files in anno_dir.
    Attributes:
        dataset_config(Dict): contains the following parameters:
            img_folder(str): the path to image data folder.
            anno_dir(str): the path to annotation data folder.
            input_size(int): the input size for DNN model.
            kpts_info (Dict): contains information about the number and configuration of annotated keypoints
        filenames_list (str): text file or list containing case/frame names in the dataset
        transform (A.core.composition.Compose) Image transfrom for augmentation based on albumentations
    """
    def __init__(self, dataset_config: Dict, filenames_list: str = None, transform: A.core.composition.Compose = None):

        self.img_folder = dataset_config["img_folder"]
        self.transform = transform
        self.input_size = dataset_config["input_size"]

        # get list of files in dataset:
        self.create_img_list(filenames_list=filenames_list)

        # kpts info:
        self.kpts_info = dataset_config["kpts_info"]
        self.num_kpts = len(self.kpts_info["names"])

        # get kpts annotations
        self.anno_dir = dataset_config["anno_folder"]
        self.KP_COORDS = self.load_kpts_annotations(self.img_list)

        # basic transformations:
        self.basic_transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # fig for item plots:
        self.fig = plt.figure(figsize=(16, 10))

    def create_img_list(self, filenames_list: str) -> None:
        """
        Called during construction. Creates a list containing paths to frames in the dataset
        """
        pass

    def load_kpts_annotations(self, img_list: List) -> np.ndarray:
        """
        Called during construction. Creates an array of annotated keypoints coorinates for the frames in the dataset
        """
        pass

    def get_img_and_kpts(self, index: int):
        """
        Load and parse a single data point.
        Args:
            index (int): Index
        Returns:
            img (ndarray): RGB frame in required input_size
            kpts (ndarray): Denormalized, namely in img coordinates
            img_path (string): full path to frame file in image format (PNG or equivalent)
        """
        # ge paths:
        img_path = os.path.join(self.img_folder, self.img_list[index])
        # get image: (PRE-PROCESS UNIQUE TO UltraSound data)
        img, ignore_margin = ultrasound_img_load(img_path)

        kpts = np.zeros([self.num_kpts, 2])     # default
        if self.anno_dir is not None:
            kpts = self.KP_COORDS[:, index, :].astype(int)
            if ignore_margin > 0:
                kpts[:, 0] = kpts[:, 0] - ignore_margin

        # resize to DNN input size:
        ratio = [self.input_size / float(img.shape[1]), self.input_size / float(img.shape[0])]
        if img.shape[0] != self.input_size:
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR_EXACT)
        # resizing keypoints:
        kpts = np.round(kpts * ratio)   # also cast int to float

        data = {"img": img,
                "kpts": kpts,
                "img_path": img_path,
                "ignore_margin": ignore_margin,
                "ratio": ratio
                }
        return data

    def img_to_torch(self, img: np.ndarray) -> torch.Tensor:
        """ Convert original image format to torch.Tensor """
        # resize:
        if img.shape[0] != self.input_size:
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR_EXACT)
        # transform:
        img = Image.fromarray(np.uint8(img))
        img = self.basic_transform(img)

        return img

    def normalize_pose(self, pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """ Normalizing a set of frame keypoint to [0, 1] """
        for p in pose:
            p[0] = p[0] / frame.shape[1]
            p[1] = p[1] / frame.shape[0]
        return pose

    def denormalize_pose(self, pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """ DeNormalizing a set of frame keypoint back to image coordinates """
        for p in pose:
            p[0] = p[0] * frame.shape[1]
            p[1] = p[1] * frame.shape[0]
        return pose

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = self.get_img_and_kpts(index)
        if self.transform is not None:
            transformed = self.transform(image=data["img"], keypoints=data["kpts"], teacher_kpts=[])
            img, kpts = transformed["image"], np.asarray(transformed["keypoints"])
        else:
            img = data["img"]
            kpts = data["kpts"]

        kpts = self.normalize_pose(pose=kpts, frame=img)
        kpts = torch.tensor(kpts).float()

        # transform:
        img = Image.fromarray(np.uint8(img))
        img = self.basic_transform(img)

        return img, kpts, data["img_path"]

    def plot_item(self, index: int, do_augmentation: bool = True, print_folder: str = './visu/') -> None:
        """ Plot frame and gt annotations for a single data point """
        data = self.get_img_and_kpts(index)

        basename = os.path.splitext(data["img_path"].replace(self.img_folder, ""))[0].replace("/", "_")
        print_fname = "{}_INDX{}_gt".format(basename, index)

        # data aug:
        if do_augmentation and self.transform is not None:
            transformed = self.transform(image=data["img"], keypoints=data["kpts"])
            img, kpts = transformed['image'], transformed['keypoints']
            print_fname = "{}_aug".format(print_fname, index)
        else:
            img = data["img"]
            kpts = data["kpts"]

        # plot:
        img_and_gt = draw_kpts(img, kpts, kpts_connections=self.kpts_info["connections"], colors_pts=self.kpts_info["colors"])
        plt.clf()
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(img.astype(np.int))
        ax2 = plt.subplot(1, 2, 2)
        plt.imshow(img_and_gt.astype(np.int))
        plt.axis('off')

        nnm = os.path.join(print_folder, print_fname)
        plt.savefig(nnm)
        print(nnm)

    def plot_prediction(self, fig: plt.Figure, img_path: str, predicted_pose: np.ndarray, gt_pose: np.ndarray,
                        is_normalized: bool = True, print_output_filename: str = None) -> plt.Figure:
        """
        Plot keypoints prediction on input frame.
        Args:
            fig: plt.Figure
            img_path: str
            predicted_pose: (numpy array, size num_kpts x 2)
            gt_pose: (numpy array, size num_kpts x 2)
            is_normalized: bool
            print_output_filename: str
        """

        img, _ = ultrasound_img_load(img_path=img_path)
        #img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_AREA)

        #frame_pose = to_numpy(predicted_pose).reshape(self.num_kpts, 2)

        if not is_normalized:
            predicted_pose = self.denormalize_pose(predicted_pose, img)
            gt_pose = self.denormalize_pose(gt_pose, img)

        plot_kpts_pred_and_gt(fig, img, gt_kpts=gt_pose, pred_kpts=predicted_pose,
                              kpts_info=self.kpts_info, closed_contour=self.kpts_info['closed_contour'])

        if print_output_filename is not None:
            #cv2.imwrite(print_output_filename, image)
            fig.savefig(print_output_filename)
            print("Cardiac contour is shown in {}".format(print_output_filename))

        return fig

