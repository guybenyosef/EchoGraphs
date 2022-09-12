import torch
import torchvision
import numpy as np
import os
from torch.utils.tensorboard.summary import hparams
import imageio
from distutils.dir_util import copy_tree
from typing import Tuple, List

# define for later use:
TorchToPIL = torchvision.transforms.ToPILImage()
PILtoTorch = torchvision.transforms.ToTensor()
Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


############################
############################
# Files utils
############################
############################
def to_numpy(some_tensor: torch.Tensor) -> np.ndarray:
    """ Converts torch Tensor to numpy array """
    if torch.is_tensor(some_tensor):
        return some_tensor.detach().cpu().numpy()
    elif type(some_tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(some_tensor)))
    return some_tensor


def to_numpy_img(tensor_img: torch.Tensor) -> np.ndarray:
    """ Converts torch Tensor image to numpy array """
    img = to_numpy(tensor_img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img

def to_torch(some_ndarray: np.ndarray) -> torch.Tensor:
    """ Converts ndarray to torch Tensor """
    if type(some_ndarray).__module__ == 'numpy':
        return torch.from_numpy(some_ndarray)
    elif not torch.is_tensor(some_ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(some_ndarray)))
    return some_ndarray

def ultrasound_img_load(img_path: str) -> Tuple[np.ndarray, int]:
    """ Load ultrasound scan image without margin """
    img = imageio.imread(img_path)
    ignore_margin = 0
    if img.shape[0] != img.shape[1]:
        # Crop according to us scan region: 708 x 708 size
        ignore_margin = int(
            0.5 * (max(img.shape[0], img.shape[1]) - min(img.shape[0], img.shape[1])))  # 154  # 154*2+708=1016
        img = img[:, ignore_margin: -ignore_margin, :]

    return img, ignore_margin


def copy_train_data(master_root_path: str, host_root_path: str,
                    folder_path_from_root: str) -> bool:  # local_folder_path=img_dirname
    """ Copy training data files to local host ssd folder """
    copied = False
    host_folder_abs_path = os.path.join(host_root_path, folder_path_from_root)
    if not os.path.exists(host_folder_abs_path):
        print("{} does not exist. Copying files from {}..".format(host_folder_abs_path, master_root_path))
        os.makedirs(host_folder_abs_path)
        master_folder_abs_path = os.path.join(master_root_path, folder_path_from_root)
        copy_tree(src=master_folder_abs_path, dst=host_folder_abs_path)
        print("Copying files done!")
        copied = True

    return copied

def ultrasound_npy_sequence_load(img_path: str, kpts_path: str, num_kpts:int , sample_rate: int = 2):
    """ Loads a sequence of ultrasound scan images without margin
        Placing ED frame at index 0, and ES frame at index -1
    """
    full_img_seq = np.load(img_path, allow_pickle=True)
    full_img_seq = full_img_seq.swapaxes(0, 1)
    kpts_list = np.load(kpts_path, allow_pickle=True)
    idx_list = []
    num = 0
    ef = kpts_list['ef']
    vol1 = kpts_list['vol1']
    vol2 = kpts_list['vol2']

    kpts = np.zeros([num_kpts, 2, 2])  # default
    for kpt in kpts_list['fnum'].tolist().keys():
        idx_list.append(int(kpt))
        kpts[:, :, num] = kpts_list['kpts'][num]
        num = num + 1

    if np.argmax(np.array(idx_list)) == 0:
        tmp = idx_list[0]
        idx_list[0] = idx_list[1]
        idx_list[1] = tmp
        kpts[:, :, 0] = kpts_list['kpts'][1]
        kpts[:, :, 1] = kpts_list['kpts'][0]
        vol2 = kpts_list['vol1']
        vol1 = kpts_list['vol2']

    x0 = max(idx_list[1], idx_list[0])
    x1 = min(idx_list[1], idx_list[0])
    step = min(x0, (x0 - x1) / (sample_rate - 1))

    frames_inds = [int(x1 + step * i) for i in range(sample_rate)]

    img = []
    for i in range(sample_rate):
        img.append(full_img_seq[frames_inds[i]])
    img = np.asarray(img)

    fnum1 = 0
    fnum2 = sample_rate - 1

    return img, kpts, ef, vol1, vol2, fnum1, fnum2, frames_inds


def ultrasound_npy_load_consecutive(img_path, kpts_path, frame_num=0):
    full_img_seq = np.load(img_path, allow_pickle=True)
    full_img_seq = full_img_seq.swapaxes(0, 1)
    kpts_list = np.load(kpts_path, allow_pickle=True)
    idx_list = []
    for kpt in kpts_list['fnum'].tolist().keys():
        idx_list.append(int(kpt))
    num_frames = full_img_seq.shape[0]

    ef = kpts_list['ef']
    vol1 = kpts_list['vol1']
    vol2 = kpts_list['vol2']

    kpts = kpts_list['kpts'][frame_num]
    img = []

    # non-annotated frame 0
    # duplicate annotated frame if frame_num =0
    if idx_list[frame_num] == 0:
        img.append(full_img_seq[idx_list[frame_num]])
    else:
        img.append(full_img_seq[idx_list[frame_num] - 1])

    # annotated frame 1
    img.append(full_img_seq[idx_list[frame_num]])

    # duplicate annotated frame if frame_num =last frame
    if idx_list[frame_num] > num_frames - 2:
        img.append(full_img_seq[idx_list[frame_num]])
    else:
        img.append(full_img_seq[idx_list[frame_num] + 1])

    img = np.asarray(img)

    return img, kpts, ef, vol1, vol2


def ultrasound_npy_sequence_load_UVT(img_path, kpts_path, num_kpts, sample_rate=2):
    full_img_seq = np.load(img_path, allow_pickle=True)
    full_img_seq = full_img_seq.swapaxes(0, 1)
    kpts_list = np.load(kpts_path, allow_pickle=True)
    idx_list = []
    num = 0
    ef = kpts_list['ef']
    vol1 = kpts_list['vol1']
    vol2 = kpts_list['vol2']

    kpts = np.zeros([num_kpts, 2, 2])  # default
    for kpt in kpts_list['fnum'].tolist().keys():
        idx_list.append(int(kpt))
        kpts[:, :, num] = kpts_list['kpts'][num]
        num = num + 1
    if np.argmax(np.array(idx_list)) == 0:
        # idx_list[0], idx_list[1] = idx_list[1], idx_list[0]
        tmp = idx_list[0]
        idx_list[0] = idx_list[1]
        idx_list[1] = tmp
        kpts[:, :, 0] = kpts_list['kpts'][1]
        kpts[:, :, 1] = kpts_list['kpts'][0]
        vol2 = kpts_list['vol1']
        vol1 = kpts_list['vol2']

    x0 = idx_list[0]  # max(idx_list[1],idx_list[0])
    x1 = idx_list[1]  # min(idx_list[1],idx_list[0])

    padding = True
    fixed_length = sample_rate
    max_length = sample_rate
    samp_size = abs(idx_list[0] - idx_list[1])
    if samp_size > fixed_length:
        full_img_seq = full_img_seq[::2, :, :, :]
        large_key = int(idx_list[1] // 2)
        small_key = int(idx_list[0] // 2)
    else:
        large_key = idx_list[1]
        small_key = idx_list[0]

    # Frames, Channel, Height, Width
    f, c, h, w = full_img_seq.shape

    first_poi = min(small_key, large_key)
    last_poi = max(small_key, large_key)
    dist = abs(small_key - large_key)

    divider = np.random.random_sample() * 5 + 2
    start_index = first_poi - dist // divider
    start_index = int(max(0, start_index) // 2 * 2)

    divider = np.random.random_sample() * 5 + 2
    end_index = last_poi + 1 + dist // divider  # +1 to INCLUDE the frame
    end_index = int(min(f, end_index) // 2 * 2)

    end_index = int(min(f, end_index) // 2 * 2)
    step = 1  # int(np.ceil((end_index - start_index) / max_length))
    # print(int(np.ceil((end_index - start_index) / max_length)))
    large_key = large_key - start_index
    small_key = small_key - start_index
    video = full_img_seq[start_index:end_index:step, :, :, :]

    window_width = video.shape[0]
    # large_key = window_width-1
    # Add blank frames to avoid confusing the network with unlabeled ED and ES frames
    missing_frames = fixed_length - window_width
    if missing_frames > 0:
        missing_frames_before = np.random.randint(missing_frames)
        missing_frames_after = missing_frames - missing_frames_before
        video = np.concatenate((video, np.zeros((missing_frames_after, c, h, w))), axis=0)

    else:
        missing_frames_before = 0
        missing_frames_after = missing_frames - missing_frames_before

    large_key = large_key + missing_frames_before
    small_key = small_key + missing_frames_before
    # print(idx_list[0],idx_list[1],small_key,large_key)
    # if padding is not None:
    #    p = padding
    #    full_img_seq = np.pad(full_img_seq, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)

    # step = min(x0,(x0 - x1) / (sample_rate - 1))
    # frames = [x1 + step * i for i in range(sample_rate)]
    # img = []
    # for i in range(sample_rate):
    #    img.append(full_img_seq[idx_list[i]])#int(frames[i])])
    video = np.asarray(video)
    # print(idx_list[0],idx_list[1],small_key,large_key)
    return video, kpts, ef, vol1, vol2, small_key, large_key


def get_kpts(heatmap, img_h=368.0, img_w=368.0):
    kpts = []

    for m in heatmap[1:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x, y])

    return kpts


############################
############################
# Log utils
############################
############################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def better_hparams(writer, hparam_dict=None, metric_dict=None):
    """Add a set of hyperparameters to be compared in TensorBoard.
    Args:
        hparam_dict (dictionary): Each key-value pair in the dictionary is the
          name of the hyper parameter and it's corresponding value.
        metric_dict (dictionary): Each key-value pair in the dictionary is the
          name of the metric and it's corresponding value. Note that the key used
          here should be unique in the tensorboard record. Otherwise the value
          you added by `add_scalar` will be displayed in hparam plugin. In most
          cases, this is unwanted.

        p.s. The value in the dictionary can be `int`, `float`, `bool`, `str`, or
        0-dim tensor
    Examples::
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter() as w:
            for i in range(5):
                w.add_hparams({'lr': 0.1*i, 'bsize': i},
                              {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
    Expected result:
    .. image:: _static/img/tensorboard/add_hparam.png
       :scale: 50 %
    """
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    # writer.file_writer.add_summary(sei)
    # for k, v in metric_dict.items():
    #     writer.add_scalar(k, v)
    # with SummaryWriter(log_dir=os.path.join(self.file_writer.get_logdir(), str(time.time()))) as w_hp:
    #     w_hp.file_writer.add_summary(exp)
    #     w_hp.file_writer.add_summary(ssi)
    #     w_hp.file_writer.add_summary(sei)
    #     for k, v in metric_dict.items():
    #         w_hp.add_scalar(k, v)

    return sei

    ########################3
    print('====================== Summary: ==============================')
    print('==============================================================\n')
    print("\nTested %d examples from %s." % (i + 1, ds.testset.img_folder))
    print(
        "\nMean Euclidean distance between predicted and ground-truth key-points is: %.2f" % np.mean(dist_pred_gt_kpts))
    print("\nMean Euclidean distance for each keypoint is:")
    print(np.mean(dist_pred_gt_kpts, 0))
    # print("\nHighest Mean Euclidean distance for test image (top 10)")
    # dist_for_file = np.mean(dist_pred_gt_kpts, 1)
    # top_indx = dist_for_file.argsort()[-10:][::-1] # top 10
    # for k in top_indx:
    #     print("File: %s, Mean distance: %.2f" % (total_filenames[k], dist_for_file[k]))
    # print("\n")


def log_kpts_statistics(dist_pred_gt_kpts, iou_pred_gt_contours, area_pred_gt_kpts, sor_pred_gt_kpts, ds,
                        total_filenames, textfilename):
    with open(textfilename, 'w') as f:

        strr = '====================== Summary: =============================='
        f.write("{}\n".format(strr))
        print(strr)
        strr = "Tested {} examples from {}".format(len(total_filenames), ds.testset.img_folder)
        f.write("{}\n\n".format(strr))
        print(strr)
        strr = "Mean Euclidean distance between predicted and ground-truth key-points is: {:.2f}%".format(
            np.mean(dist_pred_gt_kpts))
        f.write("{}\n".format(strr))
        print(strr)
        dist = np.mean(dist_pred_gt_kpts, 0)
        kpts_dist_text = "    ".join(
            ["[P{}]{:.2f}".format(kpt_index, dist[kpt_index]) for kpt_index in range(len(dist))])
        strr = "Mean Euclidean distance for each keypoint is:\n {}".format(kpts_dist_text)
        f.write("{}\n\n".format(strr))
        print(strr)

        other = [area_pred_gt_kpts, sor_pred_gt_kpts]
        other_names = ["Area", "SOR"]
        for ii, p in enumerate(other):
            strr = "Mean {} error is: {:.2f}".format(other_names[ii], np.mean(p))
            num_elements = len(p)
            f.write("{}\n".format(strr))
            print(strr)
            hist = np.histogram(p)
            hist_text = "    ".join(
                ["[{:.2f},{:.2f}]{}".format(hist[1][bin_index], hist[1][bin_index + 1],
                                            hist[0][bin_index] / num_elements)
                 for bin_index in range(len(hist[0]))]
            )
            # older version:
            # hist = np.histogram(sor_pred_gt_kpts, range=(0, 1))[0]/len(sor_pred_gt_kpts)
            # hist_text = "    ".join(
            #     ["[{}-{}%]{:.2f}".format(bin_index * 10, (bin_index + 1) * 10, hist[bin_index]) for bin_index in
            #      range(len(hist))] + ["[>100%]{:.2f}".format(np.mean(sor_pred_gt_kpts > 1))])

            strr = "{} error histogram is:\n {}".format(other_names[ii], hist_text)
            f.write("{}\n\n".format(strr))
            print(strr)

        strr = "Highest Mean Euclidean distance for test image (top 10)"
        f.write("{}\n".format(strr))
        print(strr)

        dist_for_file = np.mean(dist_pred_gt_kpts, 1)
        top_indx = dist_for_file.argsort()[-10:][::-1]  # top 10
        for k in top_indx:
            strr = "File: %s, Mean distance: %.2f" % (total_filenames[k], dist_for_file[k])
            f.write("{}\n".format(strr))
            print(strr)

    print("Evaluation report was saved to %s" % textfilename)
