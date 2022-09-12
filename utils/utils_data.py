import numpy as np
import torch
import cv2
import torchvision.transforms as torch_transforms
import pydicom
from PIL import Image

def ultrasound_npy_sequence_load(img_path: str ,kpts_path: str ,num_kpts: int, frame_length: int = 32, frame_step: int = 1, mode: str = 'edToEs'):
    """ Loads a sequence of ultrasound scan images without margin. Optional modes are: 'edToEs', 'randomStart', 'random'
        'edToEs': Placing ED frame at index 0, and ES frame at index -1
        'randomStart': Compress sequence so that ed and es are definitely included. Set random starting point
        'random': select random frames,
    """
    # ====== Load Video frames and case info ======
    video = np.load(img_path, allow_pickle=True)
    video = video.swapaxes(0, 1)
    kpts_list = np.load(kpts_path, allow_pickle=True)
    num = 0
    ef, vol1, vol2 = kpts_list['ef'], kpts_list['vol1'], kpts_list['vol2']

    # Collect keypotins:
    idx_list = []
    kpts = np.zeros([num_kpts, 2, 2])  # default
    for kpt in kpts_list['fnum'].tolist().keys():
        idx_list.append(int(kpt))
        kpts[:, :, num] = kpts_list['kpts'][num]
        num += 1

    # Swap if ED before ES:
    if idx_list[0] > idx_list[1]:  # np.argmax(np.array(idx_list)) == 0:
        idx_list.reverse()
        kpts = np.flip(kpts, axis=2)   # equivalent to: kpts[:, :, 0], kpts[:, :, 1] = kpts_list['kpts'][1], kpts_list['kpts'][0]
        vol1, vol2 = vol2, vol1

    # ====== Select frames according to select mode ======
    if mode == 'edToEs':
        # take ed as first and es as last frame and sample      all frames in between
        # compute step:
        x0 = max(idx_list[1], idx_list[0])
        x1 = min(idx_list[1], idx_list[0])
        step = min(x0, (x0 - x1) / (frame_length - 1))
        # select frames inds:
        frame_inds = [int(idx_list[0] + step * i) for i in range(frame_length)]

        # Collect frames:
        frames = []
        for i in range(frame_length):
            frames.append(video[frame_inds[i]])
        imgs = np.asarray(frames)

        fnum1, fnum2 = 0, frame_length - 1

    elif mode == 'randomStart':
        # compress sequence so that ed and es are definitely included
        # set random starting point
        # note that short sequences are mirrored by default, so that random starting positions might lead
        # to sequences that have mirrored frames
        pass    # Fixme, code below

    elif mode == 'random':

        large_key, small_key = idx_list[-1], idx_list[0]
        num_frames = video.shape[0]

        # mirror frames if length is too short
        if num_frames < frame_length * frame_step:
            video = np.concatenate((video, video[-2::-1]), axis=0)
            num_frames = video.shape[0]

        starting_idx = np.random.randint(num_frames - frame_length * frame_step - 1)
        label = np.zeros(num_frames)
        label[small_key] = 1
        label[large_key] = 2


        frame_inds = list(range(starting_idx, starting_idx + frame_length * frame_step, frame_step))
        imgs, label = video[frame_inds], label[frame_inds]

        # find the new key or set it to -1 (no annotation present) otherwise
        fnum1, fnum2 = -1, -1
        small_key, large_key = np.where(label == 1)[0], np.where(label == 2)[0]
        if len(small_key) > 0:
            fnum1 = small_key[0]
        if len(large_key) > 0:
            fnum2 = large_key[0]

    elif mode == 'all':

        large_key, small_key = idx_list[-1], idx_list[0]
        num_frames = video.shape[0]

        starting_idx = 0
        label = np.zeros(num_frames)
        label[small_key] = 1
        label[large_key] = 2

        frame_inds = list(range(num_frames))
        imgs, label = video[frame_inds], label[frame_inds]

        # find the new key or set it to -1 (no annotation present) otherwise
        fnum1, fnum2 = -1, -1
        small_key, large_key = np.where(label == 1)[0], np.where(label == 2)[0]
        if len(small_key) > 0:
            fnum1 = small_key[0]
        if len(large_key) > 0:
            fnum2 = large_key[0]


    imgs = np.asarray(imgs)

    return imgs, kpts, ef, vol1, vol2, fnum1, fnum2, frame_inds

def crop(frame, crop_image_region=False):
    """

    """
    data = frame.pixel_array.astype('uint8')
    if crop_image_region and hasattr(frame, "SequenceOfUltrasoundRegions"):
        regions = frame.SequenceOfUltrasoundRegions[0]
        y_min, y_max = regions.RegionLocationMinY0, regions.RegionLocationMaxY1
        x_min, x_max = regions.RegionLocationMinX0, regions.RegionLocationMaxX1
        if len(data.shape) == 3:
            return data[y_min:y_max+1, x_min:x_max+1]
        elif len(data.shape) == 4:
            return data[:, y_min:y_max+1, x_min:x_max+1]
        else:
            raise NotImplemented(f"Pixel array of shape {data.shape} not supported")
    else:
        if not hasattr(frame, "SequenceOfUltrasoundRegions"):
            print("Crop location not found in dicom file. Returning data without cropping.")
        return data

def load_png_image_as_batch(path: str) -> np.array:
    """

    """
    img = Image.open(path)
    img = np.array(img)
    return np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

def load_dcm_image_as_batch(path: str, frame_num: int):
    """

    """
    scan_conv = True

    dcm = pydicom.dcmread(path)
    if scan_conv == True:
        movie = crop(dcm,scan_conv)
    else:
        movie = dcm.pixel_array.astype('uint8')

    img = movie[int(frame_num)]

    if dcm.PhotometricInterpretation == "YBR_FULL_422":
        img = pydicom.pixel_data_handlers.util.convert_color_space(img, "YBR_FULL_422", "RGB")
    elif dcm.PhotometricInterpretation == "YBR_FULL":
        img = pydicom.pixel_data_handlers.util.convert_color_space(img, "YBR_FULL_422", "RGB")

    im_pil= Image.fromarray(np.uint8(img)).convert("L")
    img = np.array(im_pil)
    return np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

def load_dcm_sequence(path: str) -> np.array:
    """

    """
    scan_conv = True #crop outside area of the dcm frame

    dcm = pydicom.dcmread(path)
    if scan_conv == True:
        movie = crop(dcm,scan_conv)
    else:
        movie = dcm.pixel_array.astype('uint8')

    if dcm.PhotometricInterpretation == "YBR_FULL_422":
        movie = pydicom.pixel_data_handlers.util.convert_color_space(movie, "YBR_FULL_422", "RGB")
    elif dcm.PhotometricInterpretation == "YBR_FULL":
        movie = pydicom.pixel_data_handlers.util.convert_color_space(movie, "YBR_FULL_422", "RGB")

    return movie

def load_avi_sequence(filename: str) -> np.ndarray:
    """
    Function adapted from the echonet repository https://github.com/echonet/dynamic
    Loads a video from a file.
    """
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

    return v

def load_sequence_as_npy(file_name: str = None) -> np.ndarray:
    """

    """
    if file_name.split('.')[-1] == 'npy':
        npy_image= np.load(file_name)
        npy_image = npy_image.swapaxes(0, 1)

    elif file_name.split('.')[-1] == 'dcm':
        npy_image = load_dcm_sequence(file_name)
    elif file_name.split('.')[-1] == 'avi':
        npy_image = load_avi_sequence(file_name)
    else:
        raise ValueError('Image file not supported..',file_name.split('.')[-1])
    return npy_image

def load_image_as_npy(file_name: str = None, frame_num =0 ) -> np.ndarray:
    """

    """
    if file_name.split('.')[-1] == 'npy':
        npy_image= np.load(file_name)
        npy_image = npy_image.swapaxes(0, 1)
    elif file_name.split('.')[-1] == 'dcm':
        npy_image = load_dcm_image_as_batch(file_name,frame_num)
    elif file_name.split('.')[-1] == 'png':
        npy_image = load_png_image_as_batch(file_name)
    else:
        raise ValueError('Image file not supported..',file_name.split('.')[-1])
    return npy_image

def transform_image_sequence_to_tensor(sequence: np.array, device:torch.device) -> torch.Tensor:
    """

    """
    for idx in range(0,len(sequence)):
       if idx ==0:
            images = transform_image_to_tensor(sequence[idx])
            images = images.unsqueeze(0)
       else:
            image = transform_image_to_tensor(sequence[idx]).unsqueeze(0)
            images = torch.cat((images, image), dim=0)
    images= images.permute(0, 1, 2, 3)
    return images.to(device)

def transform_image_to_tensor(image:np.array, image_size:int = 112) -> torch.Tensor:
    """

    """
    basic_transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
        torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.fromarray(np.uint8(image)).convert('RGB') # channel must be last
    image = image.resize((image_size,image_size), resample=0)
    image = basic_transform(image)
    return image

def transform_image_sequence_to_tensor(sequence:np.array, device:str, image_size:int = 112) -> torch.Tensor:
    """

    """
    for idx in range(0,len(sequence)):
       if idx ==0:
            images = transform_image_to_tensor(sequence[idx],image_size)
            images = images.unsqueeze(0)
       else:
            image = transform_image_to_tensor(sequence[idx],image_size).unsqueeze(0)
            images = torch.cat((images, image), dim=0)
    images= images.permute(0, 1, 2, 3)
    return images.to(device)

"""
    if mode == 'randomStart':   
        # compress sequence so that ed and es are definitely included
        # set random starting point
        # note that short sequences are mirrored by default, so that random starting positions might lead
        # to sequences that have mirrored frames

        fixed_length = frame_length
        large_key, small_key = idx_list[-1], idx_list[0]

        # Mirror video with frames if too short
        video = np.concatenate((video, video[-2::-1]), axis=0)
        f, c, h, w = video.shape

        label = np.zeros(f)
        label[small_key] = 1  # End systole (small)
        label[large_key] = 2  # End diastole (large)

        step = int(np.ceil((large_key - small_key + 1) / fixed_length))
        label[small_key:small_key + step] = 1
        label[large_key - step + 1:large_key + 1] = 2

        # compress video to fit ed and es in one batch
        video = video[::step, :, :, :]
        f, c, h, w = video.shape

        # function should be replaced taken from the transformer paper
        # probably label = label[::step] does the very same
        label = torch.nn.functional.max_pool1d(torch.tensor(label[None, None, :]), step).squeeze().numpy()

        # select random starting postion
        first_occurence = int(np.where(label == 1)[0][0])
        last_occurence = int(np.where(label == 2)[0][-1])
        difference = last_occurence - first_occurence + 1
        difference_right = f - last_occurence
        if fixed_length - difference > 0:
            difference_left = np.random.randint(fixed_length - difference)
            if (difference_right + difference_left + difference) < fixed_length:
                difference_left = fixed_length - difference
            first_occurence = max(0, first_occurence - difference_left)

        video = video[first_occurence:first_occurence + fixed_length, :, :, :]
        label = label[first_occurence:first_occurence + fixed_length]

        fnum1 = int(np.where(label == 1)[0][0])
        fnum2 = int(np.where(label == 2)[0][-1])

        imgs = video
        frame_inds = label

        assert video.shape[0] != frame_length, "Error? first_occurence={}, last_occurence={}, difference={}, f={}, video_shape={}, small_key={}, large_key={}".\
            format(first_occurence, last_occurence, difference, f, video.shape[0], small_key, large_key)


"""
