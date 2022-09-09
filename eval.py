import os
import torch
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from scipy.special import softmax
from fvcore.common.config import CfgNode

from engine.loops import validate
from engine.checkpoints import load_trained_model
from evaluation.EchonetEvaluator import EchonetEvaluator
from datasets import datas, load_dataset
from utils.utils_files import to_numpy
from config.defaults import cfg_costum_setup, default_argument_parser,overwrite_eval_cfg

def sliding():
    mode = 'sliding_window'

########################################
########################################
# Main
########################################
########################################
def eval_trained_model(model: torch.nn.Module, cfg: CfgNode, ds: datas,
                       basedir: str, basename: str, device: torch.device, batch_size: int, num_workers: int, num_examples_to_plot: int):

    # Load model:
    model = model.to(device)
    dataset_name = cfg.EVAL.DATASET
    model_name = cfg.MODEL.NAME

    # Get dataloaders
    testloader = torch.utils.data.DataLoader(ds.testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True
                                              )

    # run test:
    test_losses, test_outputs, test_inputs = validate(mode='test',
                                                      epoch=1,
                                                      loader=testloader,
                                                      model=model,
                                                      device=device,
                                                      criterion=None)
    test_loss = test_losses["main"].avg
    dataset_info = cfg.EVAL.DATASET
    out_directory = os.path.join(basedir, "{}_eval_on_{}/".format(basename, dataset_info))

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    with open(os.path.join(out_directory,"eval_config.yaml"), "w") as f:
        f.write(cfg.dump())   # save config to file
    # frames_info_file = pd.read_csv(ds.testset.echonet_frame_info_csvfile, index_col=0)
    # frames_info_file = frames_info_file[frames_info_file.Split == "TEST"]

    evaluator = EchonetEvaluator(dataset=ds.testset, tasks=["ef"], output_dir=out_directory)
    evaluator.process(test_inputs, test_outputs)
    evaluator.evaluate()
    evaluator.plot(num_examples_to_plot=min(num_examples_to_plot, len(test_outputs)))

    print(" ** test loss: {}".format(test_loss))
    # compute_stats(total_filenames, total_output_guiding, total_gt_guiding, textfilename=textfilename)

def eval_sliding_window(model:torch.nn.Module, cfg: CfgNode, ds: datas, device: torch.device, basedir: str, basename: str,):
    g = ds.testset
    window_size = 16
    frame_step = 2
    predictions = dict()

    # !!!
    model.eval()

    for case_index in range(len(g)):
        prediction = dict()
        case_data = g.get_img_and_kpts(index=case_index)
        all_frames = case_data["img"]
        num_frames_in_case = all_frames.shape[-1]
        frame_size = all_frames.shape[:2]
        data_path_from_root = case_data["img_path"].replace(g.img_folder, "")
        prediction["ef_prediction"], prediction["sd_prediction"], prediction["keypoints_prediction"] = [], [], []
        prediction["data_path_from_root"] = data_path_from_root
        prediction["ef"] = case_data["ef"]
        prediction["sd"] = np.asarray([case_data["index_frame1"], case_data["index_frame2"]])

        #for ii in range(num_frames_in_case - window_size * frame_step):
        for ii in range(0, num_frames_in_case - window_size * frame_step, window_size * frame_step):
        #for ii in list(np.random.randint(num_frames_in_case - window_size * frame_step, size=10)):
            indices = list(range(ii, ii + window_size * frame_step, frame_step))
            #indices = list(range(16))
            img = all_frames[:, :, :, indices]
            # image norm:
            resized_img = torch.zeros([img.shape[2], g.input_size, g.input_size, img.shape[3]])
            for idx in range(img.shape[3]):
                img_slice = img[:, :, :, idx]
                img_slice = Image.fromarray(np.uint8(img_slice))
                img_slice = g.basic_transform(img_slice)
                resized_img[:, :, :, idx] = img_slice
            img = resized_img

            #img = [g.basic_transform(Image.fromarray(np.uint8(img[:, :, :, k]))) for k in range(window_size)]
            #img = torch.stack(img)
            #img = torch.reshape(img, (1, 3, window_size, frame_size[0], frame_size[1]))
            img = img.unsqueeze(dim=0)
            img = img.to(device)
            ef_pred, kpts_pred, sd_pred = model(img)
            to_numpy(img)
            prediction["ef_prediction"].append(g.denormalize_ef(to_numpy(ef_pred)[0][0]))
            # fix code duplication, taken from EchoNetEvaluator
            sd_pred = np.argmax(softmax(to_numpy(sd_pred)[0]), axis=0)  # convert to logits format, same as gt
            prediction["sd_prediction"].append(g.denormalize_sd(sd_pred))
            prediction["keypoints_prediction"].append(to_numpy(kpts_pred)[0])
            #

        prediction["ef_mean_prediction"] = np.mean(np.array(prediction["ef_prediction"]))
        prediction["mEFerr"] = np.abs(prediction["ef_mean_prediction"] - prediction["ef"])
        if case_index % 10 == 0:
            print("done running sliding window ({} iterations) for case {} [{}/{}].  Pred ef={}, ef={},   mEFerr={}".
                  format(num_frames_in_case - window_size, data_path_from_root, case_index, len(g),
                         prediction["ef_mean_prediction"], prediction["ef"], prediction["mEFerr"]))
        predictions[data_path_from_root] = prediction

    print("done eval")
    all_ef_predictions = np.asarray([prediction[1]["ef_mean_prediction"] for prediction in predictions.items()])
    all_ef = np.asarray([prediction[1]["ef"] for prediction in predictions.items()])
    total_mEFerr = np.mean(np.abs(all_ef - all_ef_predictions))
    print("total EF error for test set: {}".format(total_mEFerr))
    dataset_info = cfg.EVAL.DATASET
    out_directory = os.path.join(basedir, "{}_eval_on_{}/".format(basename, dataset_info))
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    with open(os.path.join(out_directory,"eval_config.yaml"), "w") as f:
        f.write(cfg.dump())   # save config to file

    evaluator = EchonetEvaluator(dataset=ds.testset, tasks=["ef"], output_dir=out_directory)
    fig = evaluator._plot_ef_scatters(ef=all_ef, ef_prediction=all_ef_predictions)
    file_path = os.path.join(out_directory, "scatter_plot_echonet_sliding_window_OVERLAP.png")
    fig.savefig(file_path)
    file_path = os.path.join(out_directory, "echonet_sliding_window_predictions_OVERLAP.npz")
    np.savez(file_path, predictions=predictions)
    print("predictions were saved to {}".format(file_path))



if __name__ == '__main__':
    args = default_argument_parser()
    cfg_eval = cfg_costum_setup(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_model, _ = load_trained_model(weights_filename=cfg_eval.EVAL.WEIGHTS)
    cfg = overwrite_eval_cfg(cfg_model,cfg_eval)

    model = model.to(device)
    basedir = os.path.dirname(cfg.EVAL.WEIGHTS)
    basename = os.path.splitext(os.path.basename(cfg.EVAL.WEIGHTS))[0]

    if cfg.EVAL.MODE == 'normal':
        ds = load_dataset(ds_name=cfg.EVAL.DATASET, input_transform=None, input_size=cfg.EVAL.INPUT_SIZE, num_frames=cfg.NUM_FRAMES)
        eval_trained_model(model=model, cfg=cfg, ds=ds,
                           basedir=basedir,
                           basename=basename,
                           device=device,
                           batch_size=cfg.EVAL.BATCH_SIZE,
                           num_workers=cfg.EVAL.NUM_WORKERS,
                           num_examples_to_plot=cfg.EVAL.EXAMPLES_TO_PLOT
                           )

    elif cfg.EVAL.MODE == 'sliding_window':
        ds = load_dataset(ds_name="sliding_window", input_transform=None, input_size=cfg.EVAL.INPUT_SIZE, num_frames=cfg.NUM_FRAMES)
        #ds = load_dataset(ds_name="echonet_random", input_transform=None, input_size=train_params.input_size, num_frames=16)
        eval_sliding_window(model=model,cfg=cfg, ds=ds, device=device,
                            basedir=basedir,
                            basename=basename,
                            )


