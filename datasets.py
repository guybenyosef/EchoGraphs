import os
import albumentations as A
import CONST
from typing import Dict

from data.USKpts import USKpts
from data.USKptsNPZ import USKptsNPZ
from data.USKpts_EchoNet import USKpts_EchoNet
from transforms import load_transform
from utils.utils_files import copy_train_data

########################
# Loaders:
########################
class datas(object):
    """
    A simple class to hold the train/val/test dataset objects.
    """
    def __init__(self, loader_func: USKpts, dataset_config: Dict, input_transform: A.core.composition.Compose,
                 train_filenames_list: str, val_filenames_list: str, test_filenames_list: str):
        self.loader_func = loader_func
        self.input_transform = input_transform

        self.dataset_config = dataset_config
        self.dataset_config["kpts_info"] = self.create_kpts_info(num_kpts=dataset_config["num_kpts"], closed_contour=dataset_config["closed_contour"])

        assert os.path.exists(dataset_config["img_folder"]), "image repository does not exist."
        self.trainset = self.load_train(train_filenames_list)
        self.valset = self.load_test(val_filenames_list)
        self.testset = self.load_test(test_filenames_list)

    def create_kpts_info(self, num_kpts: int, closed_contour: bool) -> Dict:
        kpts_info = {'names':[], 'connections':[], 'colors':[]}
        kpts_info['names'] = {}
        for kpt_indx in range(num_kpts):
            kpts_info['names']["kp{}".format(kpt_indx+1)] = kpt_indx
        kpts_info['connections'] = [[i, i+1] for i in range(len(kpts_info['names'])-1)]
        kpts_info['colors'] = [[0, 0, 255], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                               [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                               [170, 55, 0], [85, 55, 0], [0, 55, 0], [0, 55, 85]
                               ]  # Note: Limited to 12 classes.
        kpts_info['closed_contour'] = closed_contour
        return kpts_info

    def load_train(self, train_filenames_list: str) -> USKpts:
        trainset = None
        if train_filenames_list is not None:
            trainset = self.loader_func(dataset_config=self.dataset_config, filenames_list=train_filenames_list, transform=self.input_transform)
        return trainset

    def load_test(self, test_filenames_list: str) -> USKpts:
        testset = None
        if test_filenames_list is not None:
            testset = self.loader_func(dataset_config=self.dataset_config, filenames_list=test_filenames_list, transform=None)
        return testset


# ============================
# main load dataset module:
# ============================
def load_dataset(ds_name: str, input_transform: A.core.composition.Compose = None, input_size: int = 256, num_frames: int = 1) -> datas:

    us_data_folder = CONST.US_MultiviewData
    # Copy data to host, if needed
    if hasattr(CONST, 'US_MultiviewData_MASTER'):
        copy_train_data(master_root_path=CONST.US_MultiviewData_MASTER, host_root_path=us_data_folder, folder_path_from_root="preprocessed/40/")


    if ds_name == 'apical':     # 427 + 107 = 534 examples
        img_dirname = os.path.join(us_data_folder,  "apical/frames/")  #"/shared-data5/MultiView/apical/movies/"  #"/shared-data5/MultiView/apical/frames/"
        anno_dirname = os.path.join(us_data_folder,  "apical/annotations/")    #os.path.join(us_data_folder,  "apical/annotations_movies/"    #os.path.join(us_data_folder,  "apical/annotations/"
        loader_func = USKptsNPZ
        train_filenames_list = 'files/filenames/apical_train_filenames.txt'   #'files/apical_test_filenames.txt' #'files/apical_train_filenames.txt'
        val_filenames_list = 'files/filenames/apical_val_filenames.txt'  #'files/apical_test_filenames.txt' #'files/apical_val_filenames.txt'
        test_filenames_list = 'files/filenames/apical_test_filenames.txt' #'files/apical_test_filenames.txt'  #'files/apical_test_filenames.txt'
        frame_selection_mode = None
        nb_classes, closed_contour = 12, False

    elif ds_name == 'echonet40':     # 19800 examples
        img_dirname = os.path.join(us_data_folder, "preprocessed/40/frames/")
        anno_dirname = os.path.join(us_data_folder, "preprocessed/40/annotations/")
        loader_func = USKptsNPZ
        train_filenames_list = 'files/filenames/40/echonet_train_filenames.txt'
        val_filenames_list = 'files/filenames/40/echonet_val_filenames.txt'
        test_filenames_list = 'files/filenames/40/echonet_test_filenames.txt'
        frame_selection_mode = None
        nb_classes, closed_contour = 40, False

    elif ds_name == 'echonet_cycle':     # 10000 examples # files can be created using preprocess_echonet.py
        loader_func = USKpts_EchoNet
        img_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/frames/")
        anno_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/annotations/")
        train_filenames_list = 'files/filenames/echonet_cycle_train_filenames.txt'
        val_filenames_list = 'files/filenames/echonet_cycle_val_filenames.txt'
        test_filenames_list = 'files/filenames/echonet_cycle_test_filenames.txt'
        frame_selection_mode = 'edToEs'
        nb_classes, closed_contour = 40, False

    elif ds_name == 'echonet_random':     # 10000 examples # files can be created using preprocess_echonet.py
        loader_func = USKpts_EchoNet
        img_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/frames/")
        anno_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/annotations/")
        train_filenames_list = 'files/filenames/echonet_cycle_train_filenames.txt'
        val_filenames_list = 'files/filenames/echonet_cycle_val_filenames.txt'
        test_filenames_list = 'files/filenames/echonet_cycle_test_filenames.txt'
        frame_selection_mode = 'random'
        nb_classes, closed_contour = 40, False

    elif ds_name == 'debug':     # 10000 examples # files can be created using preprocess_echonet.py
        loader_func = USKpts_EchoNet
        img_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/frames/")
        anno_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/annotations/")
        train_filenames_list = 'files/filenames/echonet_cycle_valsmall_filenames.txt'
        val_filenames_list = 'files/filenames/echonet_cycle_valsmall_filenames.txt'
        test_filenames_list = 'files/filenames/echonet_cycle_valsmall_filenames.txt'
        frame_selection_mode = 'random'#'edToEs'
        nb_classes, closed_contour = 40, False

    elif ds_name == 'debug_edtosd':     # 10000 examples # files can be created using preprocess_echonet.py
        loader_func = USKpts_EchoNet
        img_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/frames/")
        anno_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/annotations/")
        train_filenames_list = 'files/filenames/echonet_cycle_valsmall_filenames.txt'
        val_filenames_list = 'files/filenames/echonet_cycle_valsmall_filenames.txt'
        test_filenames_list = 'files/filenames/echonet_cycle_valsmall_filenames.txt'
        frame_selection_mode = 'edToEs'#'edToEs'
        nb_classes, closed_contour = 40, False

    elif ds_name == 'sliding_window':     # 10000 examples # files can be created using preprocess_echonet.py
        loader_func = USKpts_EchoNet
        img_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/frames/")
        anno_dirname = os.path.join(us_data_folder, "preprocessed/40/cycle/annotations/")
        train_filenames_list = 'files/filenames/echonet_cycle_test_filenames.txt'
        val_filenames_list = 'files/filenames/echonet_cycle_test_filenames.txt'
        test_filenames_list = 'files/filenames/echonet_cycle_test_filenames.txt'
        frame_selection_mode = 'all'#'edToEs'
        nb_classes, closed_contour = 40, False


    else:
        raise NotImplementedError("Can't use dataset {}.".format(ds_name))

    dataset_config = {"img_folder": img_dirname, "anno_folder": anno_dirname, "transform": input_transform, "input_size": input_size,
                      "num_kpts": nb_classes, "closed_contour": closed_contour, "num_frames": num_frames, "frame_selection_mode": frame_selection_mode}

    ds = datas(loader_func=loader_func, dataset_config=dataset_config, input_transform=input_transform,
               train_filenames_list=train_filenames_list, val_filenames_list=val_filenames_list, test_filenames_list=test_filenames_list)

    if ds.trainset is not None and ds.testset is not None:
            print("loading dataset : {}.. number of train examples is {}, number of val examples is {}, number of test examples is {}."
                  .format(ds_name, len(ds.trainset), len(ds.valset), len(ds.testset)))
    else:
            print('loading empty dataset.')

    return ds


if __name__ == '__main__':
    ds_name = "sliding_window"#"debug"#"echonet_random"#"echonet_random"#"echonet_cycle"
    input_size = 112 #112#256#128 #708 # 224
    num_frames = 16    #4, 24
    augmentation_type = "strong_echo_cycle" #"strongkeep" #"twochkeep" #"strongkeep"

    # ds_name = "echonet40"    #"2ch5dist", "2ch5ext", "2ch5_debug"
    # augmentation_type = "strongkeep_echo" #"strongkeep" #"twochkeep" #"strongkeep"


    #input_transform = None
    input_transform = load_transform(augmentation_type=augmentation_type, augmentation_probability=1.0, input_size=input_size, num_frames=num_frames)

    ds = load_dataset(ds_name=ds_name, input_transform=input_transform, input_size=input_size, num_frames=num_frames)
    g = ds.trainset#ds.valset#ds.trainset
    for k in range(10, 30, 1): #len(g)):
    #for k in range(len(g)):
        dat = g.get_img_and_kpts(index=k)
        g.plot_item(k, do_augmentation=False, print_folder=os.path.join("./visu/", ds_name))
        g.plot_item(k, do_augmentation=True, print_folder=os.path.join("./visu/", ds_name))

