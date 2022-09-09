import torch
import os
import numpy as np
from PIL import Image
import glob
from typing import List, Dict, Tuple
import albumentations as A

# internal:
from data.USKpts import USKpts
from utils.utils_plot import plot_grid, draw_kpts
from utils.utils_data import ultrasound_npy_sequence_load

class USKpts_EchoNet(USKpts):
    """
    Dataset for US keypoints sequences
    loads a npy sequence of images and a npz file containing keypoints and meta information
    kpts, ef (taken from the gt excel sheet), vol1 (volume of the first frame),
    vol2 (volume of the second frame). The retrieved image sequence concatenates two annotated frames.
    - img[B,C,W,H,2]
    - kpts[B,N,2,2]
    - ef/vol1/vol2[B,1]
    """
    def __init__(self, dataset_config, filenames_list: str = None, transform: A.core.composition.Compose = None):

        self.num_frames = dataset_config["num_frames"]
        self.frame_selection_mode = dataset_config["frame_selection_mode"]   #options are: 'edToEs', 'random', 'randomStart'
        #self.sample_rate = dataset_config["num_frames"]#24
        self.load_single = (self.num_frames == 1)
        self.num_additional_annotation = 2
        if self.load_single:
            self.num_additional_annotation = 1

        super().__init__(dataset_config, filenames_list, transform)

        self.echonet_frame_info_csvfile = './files/FileList.csv'

    def create_img_list(self, filenames_list: List) -> None:
        self.filenames_list = filenames_list

        img_list_from_file = []
        if filenames_list is not None:
            if not type(filenames_list) == list:  # (if single element then put in list)
                filenames_list = [filenames_list]

            for filenames_sublist in filenames_list:
                if os.path.exists(filenames_sublist):  # (filenames_sublist is file)
                    with open(filenames_sublist) as f:
                        img_list_from_file.extend(f.read().splitlines())
                else:  # (filenames_sublist is a case_name)
                    img_list_from_file.append(filenames_sublist)

        self.img_list = []
        if len(img_list_from_file) > 0:
            for f in img_list_from_file:
                f = f.replace("png", "npy")
                fullpath = os.path.join(self.img_folder, f)
                if os.path.exists(fullpath):
                    if os.path.isdir(fullpath):  # f is a case_name
                        to_add = glob.glob(os.path.join(fullpath, "*.png"))
                        to_add = [f.replace(self.img_folder, "") for f in to_add]
                        self.img_list.extend(to_add)
                    else:  # fullpath is a single frame
                        if self.load_single == False:
                            self.img_list.append(f)
                        else:
                            self.img_list.append(f + '_0')
                            self.img_list.append(f + '_1')
        else:
            img_list_from_folder = glob.glob(os.path.join(self.img_folder, "**/*.png"))
            if self.load_single == False:
                self.img_list = [os.path.basename(f) for f in img_list_from_folder]

    def normalize_ef(self, ef):
        return ef * 0.01

    def denormalize_ef(self, ef):
        return ef * 100

    def normalize_sd(self, sd):
        """ normalize to [0,..,16] where 0 is a transition (non ES or ED) frame."""
        return sd + 1

    def denormalize_sd(self, sd):
        """ denormalize to [-1,..,15] where -1 is a transition (non ES or ED) frame """
        return sd - 1

    def get_img_and_kpts(self, index):
        """ Load images and annotations for a single data point """
        # paths:
        basename = self.img_list[index].split('.')[0]
        img_path = os.path.join(self.img_folder, "{}.npy".format(basename))
        kpts_path = os.path.join(self.anno_dir,  "{}.npz".format(basename))
        # get images and annotations:
        img, kpts, ef, vol1, vol2, fnum1, fnum2, frames_inds = ultrasound_npy_sequence_load(img_path=img_path,
                                                                                            kpts_path=kpts_path,
                                                                                            num_kpts=self.num_kpts,
                                                                                            frame_length=self.num_frames,
                                                                                            frame_step=2,   # for 'random' mode
                                                                                            mode=self.frame_selection_mode)

        # resize image to DNN input size:
        img = np.uint8(img)
        img = img.transpose((2, 3, 1, 0))

        data = {"img": img,
                "kpts": kpts,
                "img_path": img_path,
                "ef": ef,
                "vol_frame1": vol1,
                "vol_frame2": vol2,
                "index_frame1": fnum1,
                "index_frame2": fnum2,
                "frames_inds": frames_inds
                }
        return data

    def do_augmentations_to_sequence(self, img: np.ndarray, kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Augmentations for a sequence of frames """
        additionaltargets = {}
        for i in range(1, img.shape[3]):
            additionaltargets = {**additionaltargets, "image{}".format(i): img[:, :, :, i]}

        for i in range(1, self.num_additional_annotation):
            additionaltargets = {**additionaltargets, "keypoints{}".format(i): kpts[:, :, i]}

        transformed = self.transform(image=img[:, :, :, 0], keypoints=kpts[:, :, 0], **additionaltargets)

        img[:, :, :, 0] = transformed['image']
        for idx in range(1, img.shape[3]):
            img[:, :, :, idx] = transformed["image{}".format(idx)]

        kpts[:, :, 0] = transformed['keypoints']
        for idx in range(1, self.num_additional_annotation):
            kpts[:, :, idx] = transformed["keypoints{}".format(idx)]

        return img, kpts

    def __getitem__(self, index):

        data = self.get_img_and_kpts(index)
        img, kpts = data["img"], data["kpts"]

        if self.transform is not None:
            img, kpts = self.do_augmentations_to_sequence(img, kpts)

        kpts = np.asarray(kpts).astype(int)
        kpts = kpts.astype(float)
        kpts = self.normalize_pose(pose=kpts, frame=img)
        kpts = torch.tensor(kpts).float()

        ef = self.normalize_ef(data["ef"])
        ef = torch.tensor(ef).float()

        index_frame1, index_frame2 = self.normalize_sd(data["index_frame1"]), self.normalize_sd(data["index_frame2"])

        vol1 = torch.tensor(data["vol_frame1"]).float()
        vol2 = torch.tensor(data["vol_frame2"]).float()

        # basic transform:
        if len(img.shape) == 4:
            resized_img = torch.zeros([img.shape[2], self.input_size, self.input_size, img.shape[3]])
            for idx in range(img.shape[3]):
                img_slice = img[:, :, :, idx]
                img_slice = Image.fromarray(np.uint8(img_slice))
                img_slice = self.basic_transform(img_slice) 
                resized_img[:, :, :, idx] = img_slice
            img = resized_img
        else:
            img = Image.fromarray(np.uint8(img))
            img = self.basic_transform(img)

        return img, kpts, data["img_path"], ef, vol1, vol2, index_frame1, index_frame2


    def plot_item(self, index: int, do_augmentation: bool = True, print_folder: str = './visu/') -> None:

        """ Plot frame and gt annotations for a single data point """
        data = self.get_img_and_kpts(index)

        basename = os.path.splitext(data["img_path"].replace(self.img_folder, ""))[0].replace("/", "_")
        print_fname = "{}_INDX{}_gt".format(basename, index)

        # data aug:
        img, kpts = data["img"], data["kpts"]
        if do_augmentation and self.transform is not None:
            img, kpts = self.do_augmentations_to_sequence(img, kpts)
            print_fname = "{}_aug".format(print_fname, index)

        # plot:
        frames = [img[:, :, :, ii] for ii in range(self.num_frames)]  # np.reshape(img, img.shape[::-1])
        frames_inds = data["frames_inds"]

        # for k in [0, -1]:
        #     frames[k] = draw_kpts(img[:, :, :, k], kpts[:, :, k], kpts_connections=self.kpts_info["connections"], colors_pts=self.kpts_info["colors"])
        for ii, extermum_index in enumerate([data["index_frame1"], data["index_frame2"]]):
            if extermum_index > -1:
                # Assuming kpts[:,:,:,0] belong to frame_index1, kpts[:,:,:,1] belong to frame_index2
                frames[extermum_index] = draw_kpts(img[:, :, :, extermum_index], kpts[:, :, ii],
                                                   kpts_connections=self.kpts_info["connections"],
                                                   colors_pts=self.kpts_info["colors"])

        plot_filename = os.path.join(print_folder, "{}.png".format(print_fname))
        seq_plot = plot_grid(frames=frames, labels=frames_inds, thumbnail_size=112)
        seq_plot.save(plot_filename)
        print(plot_filename)
