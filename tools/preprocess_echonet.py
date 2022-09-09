import os
import sys
import argparse
import pandas as pd
import imageio
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils_contour import echonet_trace_to_mask
import CONST

""" Looping over the entire echonet dataset to create following files:
- array of the frame (width*height)
- npz array of the keypoints and masks (num keypoints*2, width*height)
- array of the segmentation mask (width*height)

- cycle array of all frames (frame_num*width*height)
- npz array of pairs of the keypoints, masks, both volumes and ef (num keypoints*2, width*height)

- format:
    - frames/filename_framenum.png (112,112,3)
    - annotations/filename_framenum.png
    - annotations/filename_framenum.npz ['kpts']['mask']
    - cycle/frames/filename.npy(112,112,3,x)
    - cycle/annotations/filename.npz ['kpts'](40,2,2) ['mask'](112,112,3,x) ['ef'] ['vol1'] ['vol2']

To save the filenames in the correct folders, set the global CONST variables first
CONST.US_MultiviewData must point towards the echonet data folder
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default=CONST.US_MultiviewData)
    parser.add_argument('-o', '--output_dir', type=str, default=CONST.US_MultiviewData + '/preprocessed/')
    parser.add_argument('-kpts', '--save_kpts', type=bool, default=True)
    parser.add_argument('-masks', '--save_masks', type=bool, default=True)
    parser.add_argument('-imgs', '--save_imgs', type=bool, default=True)
    args = parser.parse_args()

    return args


def loadvideo(filename: str) -> np.ndarray:
    """
    Function taken from the echonet repository https://github.com/echonet/dynamic
    Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))

    return v


def preprocess_data(input_path, output_path, save_kpts, save_masks, save_imgs):

    if not os.path.exists(input_path + '/VolumeTracings.csv'):
        raise ValueError('Directory does not contain the volume tracing csv file')
    if not os.path.exists(input_path + '/FileList.csv'):
        raise ValueError('Directory does not contain the file list csv file')

    echonet_pts = pd.read_csv(input_path + '/VolumeTracings.csv')
    echonet_pts.head()
    with open(os.path.join(input_path, "FileList.csv")) as f:
        data = pd.read_csv(f)

    fnames = data["FileName"].tolist()
    fnames = [fn + ".avi" for fn in fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix

    x_train = [data["FileName"][idx] for idx in data.index if data["Split"][idx] == 'TRAIN']
    x_val = [data["FileName"][idx] for idx in data.index if data["Split"][idx] == 'VAL']
    x_test = [data["FileName"][idx] for idx in data.index if data["Split"][idx] == 'TEST']

    # save two single frames per cycle
    output_list_train = []
    output_list_test = []
    output_list_val = []
    output_list_invalid = []

    # save full cycle
    output_list_test_cycle = []
    output_list_train_cycle = []
    output_list_val_cycle = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nnum = 0
    for fname in fnames:
        name = fname.split('.')[0]
        ef = data[data['FileName'] == name]['EF'].item()
        esv = data[data['FileName'] == name]['ESV'].item()
        edv = data[data['FileName'] == name]['EDV'].item()
        patient_df = echonet_pts[echonet_pts.FileName == fname]
        video_path = os.path.join(input_path, "Videos", fname)
        video = loadvideo(video_path).astype(np.uint8)
        print(nnum, fname)
        nnum += 1

        # check whether file exists
        if not os.path.exists(video_path):
            print('Not found: ', video_path)
            output_list_invalid.append("{0}".format(name))
            continue

        pts_pairs = []
        frame_pairs = {}
        vol_pairs = []
        frames = []
        for frame in set(patient_df.Frame):
            frame_df = patient_df[patient_df.Frame == frame]
            gt_pts = np.array(frame_df.loc[:, "X1":"Y2"])

            # check whether frame contains a strange number of points
            if ((np.size(gt_pts) / 2) != 42):
                output_list_invalid.append("{0}".format(name))
                break

            mask = echonet_trace_to_mask(gt_pts, (112, 112))
            vol_pairs.append(np.sum(mask))
            frames.append(int(frame))

            x1, y1, x2, y2 = gt_pts[1:, 0], gt_pts[1:, 1], gt_pts[1:, 2], gt_pts[1:, 3]
            x = np.concatenate((x1[:], np.flip(x2[:])))
            y = np.concatenate((y1[:], np.flip(y2[:])))
            pts = np.array([x, y]).transpose()
            num_pts = str(len(pts))

            # only use if you want to downsample the points instead of using the 40 gt points
            # lvc = LeftVentricleUnorderedContour(mask=mask)
            # oc = lvc.to_ordered_contour(num_pts=40)["myo"]
            # oc = np.asarray(oc).swapaxes(0,1)

            if name in x_test:
                output_list_test.append(str("{0}_{1}.png".format(name, frame)))
            if name in x_train:
                output_list_train.append(str("{0}_{1}.png".format(name, frame)))
            if name in x_val:
                output_list_val.append(str("{0}_{1}.png".format(name, frame)))

            frame_img = video[:, frame, :, :].transpose(1, 2, 0)
            frame_pairs[str(frame)] = frame_img

            frames_folder = os.path.join(output_path, num_pts, "frames/")
            anno_folder = os.path.join(output_path, num_pts, "annotations/")
            frames_cycle_folder = os.path.join(output_path, num_pts, "cycle/frames/")
            anno_cycle_folder = os.path.join(output_path, num_pts, "cycle/annotations/")
            file_dir = '../files/filenames/' + num_pts

            if not os.path.exists(os.path.join(output_path, str(len(pts)))):
                os.makedirs(frames_folder)
                os.makedirs(anno_folder)

            pts_pairs.append(pts)
            if save_imgs:
                imageio.imsave(frames_folder + "{0}_{1}.png".format(name, frame), frame_img)

                # Draw eval images (only for a sanity check)
                # img = np.zeros((frame_img.shape[0], frame_img.shape[1], 3), np.uint8)
                # cv2.circle(img, (int(pts[0][0]), int(pts[1][0])), 3, [255,255,0], -1)

                # for i in range(1,40):
                #    cv2.circle(img, (int(pts[0][i]), int(pts[1][i])), 1, [255, 255, 255], -1)

                # cv2.circle(img, (int(pts[0][5]), int(pts[1][5])), 3, [0, 255,0], -1)
                # frame_img[img>0] = 0
                # imageio.imsave(output_path+'frames_eval/'+"{0}_{1}.png".format(name,frame),frame_img+img)

            if save_masks:
                imageio.imsave(anno_folder + "{0}_{1}.png".format(name, frame), mask * 255)
            if save_kpts:
                np.savez(anno_folder + "{0}_{1}".format(name, frame), mask=mask * 255,
                         kpts=np.asarray(pts), ef=ef)

            if len(pts_pairs) == 2:
                if not os.path.exists(frames_cycle_folder):
                    os.makedirs(frames_cycle_folder)
                if not os.path.exists(anno_cycle_folder):
                    os.makedirs(anno_cycle_folder)
                np.save(frames_cycle_folder + "{0}".format(name), video)

                if np.argmin(vol_pairs) == 0:
                    vol1 = float(esv)
                    vol2 = float(edv)
                else:
                    vol1 = float(edv)
                    vol2 = float(esv)

                np.savez(anno_cycle_folder+ "{0}".format(name), fnum=frame_pairs,
                         kpts=np.array(pts_pairs),
                         ef=ef, vol1=vol1, vol2=vol2)

                if name in x_test:
                    output_list_test_cycle.append(str("{0}.png".format(name)))
                if name in x_train:
                    output_list_train_cycle.append(str("{0}.png".format(name)))
                if name in x_val:
                    output_list_val_cycle.append(str("{0}.png".format(name)))

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    output_file_train = open(file_dir + '/echonet_train_filenames.txt','w')
    output_file_val = open(file_dir + '/echonet_val_filenames.txt', 'w')
    output_file_test = open(file_dir + '/echonet_test_filenames.txt', 'w')
    output_file_invalid = open(file_dir + '/echonet_invalid_filenames.txt','w')

    output_file_train_cycle = open(file_dir + '/echonet_cycle_train_filenames.txt', 'w')
    output_file_val_cycle = open(file_dir + '/echonet_cycle_val_filenames.txt', 'w')
    output_file_test_cycle = open(file_dir + '/echonet_cycle_test_filenames.txt', 'w')

    for name in output_list_train_cycle:
        if name not in output_list_invalid:
            output_file_train_cycle.write(name + '\n')
    for name in output_list_test_cycle:
        if name not in output_list_invalid:
            output_file_test_cycle.write(name + '\n')
    for name in output_list_val_cycle:
        if name not in output_list_invalid:
            output_file_val_cycle.write(name + '\n')

    # in case of invalid single frames remove all occurences
    for name in output_list_test:
        if name.split('_')[0] not in output_list_invalid:
            output_file_test.write(name + '\n')
    for name in output_list_train:
        if name.split('_')[0] not in output_list_invalid:
            output_file_train.write(name + '\n')
    for name in output_list_val:
        if name.split('_')[0] not in output_list_invalid:
            output_file_val.write(name + '\n')
    for name in output_list_invalid:
        output_file_invalid.write(name + '\n')

    output_file_test.close()
    output_file_train.close()
    output_file_val.close()
    output_file_invalid.close()

    output_file_test_cycle.close()
    output_file_train_cycle.close()
    output_file_val_cycle.close()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.input_dir):
        raise ValueError('Input directory does not exist.')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess_data(input_path=args.input_dir,
                    output_path=args.output_dir,
                    save_kpts=args.save_kpts,
                    save_masks=args.save_masks,
                    save_imgs=args.save_imgs)
