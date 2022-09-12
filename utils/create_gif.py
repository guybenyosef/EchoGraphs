import glob
import cv2
import argparse
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gp', '--gif_prefix', type=str, default="/home/guy/code/ultrasound_ge/storage/logs/apical/CNNGCN/resnext101/6/weights_apical_CNNGCN_best_kptsErr_output/val")
    parser.add_argument('-gs', '--gif_suffix', type=str, default="DETECT-apical_BY-CNNGCN")
    parser.add_argument('-fp', '--frame_prefix', type=str, default="/home/guy/code/ultrasound_ge/storage/logs/apical/CNNGCN/resnext101/6/weights_apical_CNNGCN_best_kptsErr_output/val")
    parser.add_argument('-f', '--indx_from', type=int, default=0)
    parser.add_argument('-t', '--indx_to', type=int, default=10)

    args = parser.parse_args()

    return args


def plot_gifs(gif_prefix, gif_suffix, frame_prefix, indices_range):
    # CONSTS:
    text_font = cv2.FONT_HERSHEY_SIMPLEX    # font
    text_org = (50, 50)
    text_fontScale = 1
    text_color = (255, 0, 0)     # Blue color in BGR
    text_thickness = 2  # Line thickness of 2 px
    for ii in indices_range:
        key_word = "{}{}*.jpg".format(frame_prefix,  ii)
        list_of_files = glob.glob(key_word)
        if len(list_of_files) > 0:
            vis_frames = []
            for ff in list_of_files:
                vis_frames.append(imageio.imread(ff))
                # Using cv2.putText() method
                #vis_img = cv2.putText(vis_img, text_on_frame, text_org, text_font, text_fontScale, text_color, text_thickness, cv2.LINE_AA)
            full_print_fname = "{}{}_{}.gif".format(gif_prefix, ii, gif_suffix)
            imageio.mimsave(full_print_fname, vis_frames, duration=0.20)
            print("Animated GIF file was written to {}".format(full_print_fname))


if __name__ == '__main__':
    args = parse_args()
    indices_range = range(args.indx_from, args.indx_to)
    plot_gifs(gif_prefix=args.gif_prefix,
              gif_suffix=args.gif_suffix,
              frame_prefix=args.frame_prefix,
              indices_range=indices_range)

