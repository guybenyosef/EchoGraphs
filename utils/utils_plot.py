import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import scipy.interpolate as interpolate
from PIL import Image, ImageDraw
from typing import List

############################
############################
# Plot utils
############################
############################
def draw_kpts(img, kpts, kpts_connections=[], colors_pts=None, color_connection=[255, 255, 255]):

    im = img.copy() # workaround a bug in Python OpenCV wrapper: https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    # draw points
    ii = 0
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        #if colors_pts is None:
        c = (0, 0, 255)
        #else:
        #    c = colors_pts[ii]
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=c)
        ii += 1
    # draw lines
    for i in range(len(kpts_connections)):
        cur_im = im.copy()
        limb = kpts_connections[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, color_connection)
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)
    return im

def plot_kpts_pred_and_gt(fig, img, gt_kpts=None, pred_kpts=None, kpts_info=[], closed_contour=False):

    fig.clf()
    clean_img = img
    interpolation_step_size = 0.001  # 0.01
    unew = np.arange(0, 1.00, interpolation_step_size)

    if gt_kpts is not None:
        img = draw_kpts(img=img, kpts=gt_kpts,
                        kpts_connections=kpts_info["connections"],
                        colors_pts=[[255, 255, 255] for c in kpts_info['colors']],
                        color_connection=[0, 125, 0])
        if closed_contour:
            gt_kpts = np.concatenate((gt_kpts, gt_kpts[:1, :]), axis=0)
        if len(kpts_info['names']) > 3:
            gt_tck, _ = interpolate.splprep([gt_kpts[:, 0], gt_kpts[:, 1]], s=0)
            gt_interpolate = interpolate.splev(unew, gt_tck)

    if pred_kpts is not None:
        img = draw_kpts(img=img, kpts=pred_kpts,
                        kpts_connections=kpts_info["connections"],
                        colors_pts=kpts_info['colors'],
                        color_connection=[255, 255, 255])
        if closed_contour:
            pred_kpts = np.concatenate((pred_kpts, pred_kpts[:1,:]), axis=0)
        if len(kpts_info['names']) > 3:
            pred_tck, _ = interpolate.splprep([pred_kpts[:, 0], pred_kpts[:, 1]], s=0)
            pred_interpolate = interpolate.splev(unew, pred_tck)

    # option 1: clean img + kpts_img
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(clean_img)
    ax.set_axis_off()
    #
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(clean_img)
    if gt_kpts is not None and len(kpts_info['names']) > 3:
        ax.scatter(gt_interpolate[0], gt_interpolate[1], marker='.', c='green', s=2)
    if pred_kpts is not None and len(kpts_info['names']) > 3:
        ax.scatter(pred_interpolate[0], pred_interpolate[1], marker='.', c='yellow', s=2) # fixme: change color back to 'white'
    ax.set_axis_off()
    #
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img)
    #ax.set_axis_off()
    # option 2: kpts_img only
    #plt.imshow(img)

    return fig

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def resize_image(image: np.array, image_size) -> np.array:
    img = cv2.resize(image.transpose(1, 2, 0), (image_size, image_size))
    return img.transpose(2,0,1)

def plot_inference_movie(test_movie, test_movie_pred_kpts, input_size, metric_name, value):     # Andy Gilbert
    dpi = 70.0
    c ,xpixels, ypixels = resize_image(test_movie[0],input_size).shape


    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    ax = plt.gca()
    ax.axis('off')
    ax.set_title("{}: {}".format(metric_name,value))
    interpolation_step_size = 0.001  # 0.01
    unew = np.arange(0, 1.00, interpolation_step_size)

    mv = plt.imshow(rgb2gray(resize_image(test_movie[0],input_size).transpose(1,2,0)), cmap='gray') #rgb2gray(resize_image(test_movie[0],input_size))

    kpts = test_movie_pred_kpts[0]
    kpts *= input_size
    #     pred_tck, _ = interpolate.splprep([kpts[:, 0], kpts[:, 1]], s=0)
    #     pred_interpolate = interpolate.splev(unew, pred_tck)
    #     lns, = ax.scatter(pred_interpolate[0], pred_interpolate[1], marker='.', c='yellow', s=2)
    pts, = ax.plot(kpts[:, 0], kpts[:, 1], marker='o', c='red')

    def animate(i):
        mv.set_array(rgb2gray(resize_image(test_movie[i],input_size).transpose(1,2,0)))

        kpts = test_movie_pred_kpts[i] #model_inference(test_movie[i], config)
        kpts *= input_size
        #         pred_tck, _ = interpolate.splprep([kpts[:, 0], kpts[:, 1]], s=0)
        #         pred_interpolate = interpolate.splev(unew, pred_tck)
        #         lns.set_data(pred_interpolate[0], pred_interpolate[1])
        pts.set_data(kpts[:, 0], kpts[:, 1])
        return (mv, pts)

    anim = animation.FuncAnimation(fig, animate, frames=test_movie.shape[0], blit=False)
    return anim

# def plot_test_img2type(fig, filenames, pred_label, gt_label):
#     # plot:
#     num_examples_to_plot = 16
#
#     for indx in range(min(len(filenames), num_examples_to_plot)):
#         ax = fig.add_subplot(4, 4, indx + 1)
#         img = cv2.imread(filenames[indx])
#         img = cv2.resize(img, (100, 100))
#         ax.imshow(img)
#         ax.set_axis_off()
#         ax.set_title('pr:%d, gt:%d' % (pred_label[indx], gt_label[indx]), fontsize=40)
#         #suptitle('test title', fontsize=20)
#
#     return fig


def plot_grid(frames: List, labels: List, thumbnail_size: int = 30) -> Image:
    """ Plot grid of images and labels """

    num_frames = len(frames)
    dim_length = int(num_frames ** 0.5) + 1     # default: 10
    grid_image_length = dim_length * (thumbnail_size + 10)

    new_im = Image.new('RGB', (grid_image_length + 10, grid_image_length + 10))

    index = 0
    idddxxx = range(min(100, num_frames)) #np.random.randint(0, num_frames, min(100, num_frames))
    for i in range(10, grid_image_length, thumbnail_size + 10):
        for j in range(10, grid_image_length, thumbnail_size + 10):
            if index < num_frames:
                #im = Image.open(files[idddxxx[index]])
                numpy_image = frames[idddxxx[index]]
                label = labels[idddxxx[index]]
                im = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
                im.thumbnail((thumbnail_size, thumbnail_size))
                draw = ImageDraw.Draw(im)
                draw.text((0, 0), str(label), fill=128)
                #draw.text((0, 0), str(label), fill=0) #color=(255, 0, 255))#'magenta')
                new_im.paste(im, (i, j))
                index += 1
    #new_im.show()
    #input("Press Enter to continue...")

    return new_im
