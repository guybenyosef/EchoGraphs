import numpy as np
import copy
import cv2
import random
import logging
import os
import shutil
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
from typing import Dict, List, Tuple
import pickle

from datasets import datas
from .BaseEvaluator import DatasetEvaluator
from utils.utils_plot import plot_grid, draw_kpts, plot_kpts_pred_and_gt
from utils.utils_stat import match_two_kpts_set

class EchonetEvaluator(DatasetEvaluator):
    """
    Evaluate EchoNet segmentation predictions for a single iteration of the cardiac navigation model
    """

    def __init__(
        self,
        dataset: datas,
        tasks: List = ["kpts", "ef"],
        output_dir: str = "./visu",
        verbose: bool = True
    ):
        """
        Args:
            dataset (dataset object): Note: used to be dataset_name: name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "single_iter", "multi_iter".
                By default, will infer this automatically from predictions.
            output_dir (str): optional, an output directory to dump results.
        """

        self._dataset = dataset
        self._verbose = verbose
        self._tasks = tasks
        self._output_dir = output_dir
        if self._verbose:
            self.set_logger(logname=os.path.join(output_dir, "eval_log.log"))
            self._logger = logging.getLogger(__name__)

        self._cpu_device = torch.device("cpu")
        self._do_evaluation = True  # todo: add option to evaluate without gt

    def reset(self):
        self._predictions = dict()

    def set_logger(self, logname):
        print("Evaluation log file is set to {}".format(logname))
        logging.basicConfig(filename=logname,
                            filemode='w', #'a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)    #level=logging.DEBUG)    # level=logging.INFO)


    def process(self, inputs: Dict, outputs: Dict) -> None:
        """
        Args:
            inputs: the inputs to a EF and Kpts model. It is a list of dicts. Each dict corresponds to an image and
                contains keys like "keypoints", "ef".
            outputs: the outputs of a EF and Kpts model. It is a list of dicts with keys
                such as "ef_prediction" or "keypoints_prediction" that contains the proposed ef measure or keypoints coordinates.
        """
        some_val_output_item = next(iter(outputs.items()))[1]
        tasks = []
        if some_val_output_item["keypoints_prediction"] is not None:
            tasks.append("kpts")
        if some_val_output_item["ef_prediction"] is not None:
            tasks.append("ef")
        if some_val_output_item["sd_prediction"] is not None:
            tasks.append("sd")
        self._tasks = tasks

        self._predictions = dict()
        for ii, data_path in enumerate(outputs):
            prediction = dict()

            # get predictions:
            if some_val_output_item["ef_prediction"] is not None:
                prediction["ef_prediction"] = self._dataset.denormalize_ef(outputs[data_path]["ef_prediction"])
                prediction["ef"] = self._dataset.denormalize_ef(inputs[data_path]["ef"])

            if some_val_output_item["keypoints_prediction"] is not None:
                prediction["keypoints_prediction"] = outputs[data_path]["keypoints_prediction"]
                prediction["keypoints"] = inputs[data_path]["keypoints"]

            if some_val_output_item["sd_prediction"] is not None:
                prediction["sd_prediction"] = outputs[data_path]["sd_prediction"]
                prediction["sd_prediction"] = np.argmax(softmax(prediction["sd_prediction"]), axis=0)   # convert to logits format, same as gt
                prediction["sd_prediction"] = self._dataset.denormalize_sd(prediction["sd_prediction"])

                prediction["sd"] = self._dataset.denormalize_sd(inputs[data_path]["sd"])


            # get case name:
            prediction["data_path_from_root"] = data_path.replace(self._dataset.img_folder, "")

            self._predictions[data_path] = prediction


    def evaluate(self, tasks: List = None):
        if tasks is not None:
            self._tasks = tasks

        predictions = self._predictions

        if len(predictions) == 0 and self._verbose:
            self._logger.warning("[EchonetEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir is not None:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            #file_path = os.path.join(self._output_dir, "echonet_predictions.npz")
            #np.savez(file_path, predictions=predictions)
            file_path = os.path.join(self._output_dir, "echonet_predictions.pkl")
            with open(file_path, 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not self._do_evaluation and self._verbose:
            self._logger.info("Annotations are not available for evaluation.")
            return

        if self._verbose:
            self._logger.info("Evaluating predictions ...")
        self._results = OrderedDict()
        tasks = self._tasks #or self._tasks_from_predictions(lvis_results)
        for task in sorted(tasks):
            if self._verbose:
                self._logger.info("Preparing results in the EchoNet format for task {} ...".format(task))
            if task == "ef":
                res = self._eval_ejection_fraction_predictions(predictions)
            if task == "kpts":
                res = self._eval_keypoints_predictions(predictions)
            if task == "sd":
                res = self._eval_diastolic_systolic_predictions(predictions)

            self._results[task] = res

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def plot(self, num_examples_to_plot: int) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(16, 16))
        plot_directory = os.path.join(self._output_dir, "plots")
        if os.path.exists(plot_directory):
            shutil.rmtree(plot_directory)
        os.makedirs(plot_directory)
        self._logger.info("plotting {} prediction examples to {}".format(num_examples_to_plot, plot_directory))
        for data_path in random.sample(list(self._predictions), num_examples_to_plot):
            prediction = self._predictions[data_path]
            fig.clf()
            if "ef" in self._tasks:
                keypoints_prediction = prediction["keypoints_prediction"] if "kpts" in self._tasks else None
                sd_prediction = prediction["sd_prediction"] if "sd" in self._tasks else None
                ax1 = fig.add_subplot(1, 1, 1)  #ax1 = fig.add_subplot(1, 2, 1)
                self._plot_EF_prediction(ax=ax1, data_path_from_root=prediction["data_path_from_root"],
                                         ef_prediction=prediction["ef_prediction"],
                                         keypoints_prediction=keypoints_prediction,
                                         sd_prediction=sd_prediction)
            else:
                fig = self._plot_kpts_single_frame(fig, data_path_from_root=prediction["data_path_from_root"],
                                                   keypoints_prediction=prediction["keypoints_prediction"])
            plot_filename = "{}.jpg".format(os.path.splitext(prediction["data_path_from_root"])[0].replace("/", "_"))
            fig.savefig(fname=os.path.join(plot_directory, plot_filename))

    def set_tasks(self, tasks: List) -> None:
        self._tasks = tasks

    def get_tasks(self) -> List:
        return self._tasks

    def _eval_ejection_fraction_predictions(self, predictions: Dict) -> Dict:
        """
        Evaluate ejection_fraction predictions.
        Args:
            predictions (list[dict]): list of predictions from the model, as well as source EchoNet data
        """

        if self._verbose:
            self._logger.info("Eval stats for Ejection Fraction")

        ef_prediction = np.stack([output[1]["ef_prediction"] for output in predictions.items()])
        ef = np.stack([output[1]["ef"] for output in predictions.items()])
        mEfERR = np.mean(abs(ef - ef_prediction))
        if self._verbose:
            self._logger.info("Mean ef error is {}".format(mEfERR))
        if self._output_dir is not None:
            fig = self._plot_ef_scatters(ef=ef, ef_prediction=ef_prediction)
            fig.suptitle("EF scatters for EchoNet test set, size={}".format(len(predictions)))
            fig.savefig(os.path.join(self._output_dir, "ef_scatters.jpg".format()))

        return mEfERR


    def _eval_keypoints_predictions(self, predictions: Dict) -> Dict:
        """
        Evaluate keypoints predictions
        Args:
            predictions (list[dict]): list of predictions from the model
        """
        if self._verbose:
            self._logger.info("Eval stats for keypoints")

        dist_pred_gt_kpts = []
        num_kpts = self._dataset.num_kpts
        num_annotated_frames = 2 if "ef" in self._tasks else 1
        for prediction in predictions.values():
            dist_pred_gt_kpts.append(100 * match_two_kpts_set(prediction["keypoints"].reshape(num_kpts * num_annotated_frames, 2),
                                                              prediction["keypoints_prediction"].reshape(num_kpts * num_annotated_frames, 2)))
        mKptsERR = np.mean(np.stack(dist_pred_gt_kpts))
        if self._verbose:
            self._logger.info("Mean keypoints error is {}".format(mKptsERR))

        return mKptsERR

    def _eval_diastolic_systolic_predictions(self, predictions):
        """
        Evaluate frame labels for diastolic systolic predictions.
        Args:
            predictions (list[dict]): list of predictions from the model
        """
        if self._verbose:
            self._logger.info("Eval stats for keypoints")

        dist_pred_gt_SD = []
        for prediction in predictions.values():
            dist_pred_gt_SD.append(abs(prediction["sd"] - prediction["sd_prediction"]))     # FixMe
        mKsdERR = np.mean(np.stack(dist_pred_gt_SD))
        if self._verbose:
            self._logger.info("Average Frame Distance is {}".format(mKsdERR))

        return mKsdERR

    def _plot_EF_prediction(self, ax, data_path_from_root, ef_prediction, keypoints_prediction=None, sd_prediction=None):

        datapoint_index = self._dataset.img_list.index(data_path_from_root)
        data = self._dataset.get_img_and_kpts(datapoint_index)
        img = data["img"]
        frames_inds = data["frames_inds"]
        ef = data["ef"]

        frames = [img[:, :, :, ii] for ii in range(self._dataset.num_frames)]

        if keypoints_prediction is not None:
            extrema_indices = [0, self._dataset.num_frames - 1]
            if sd_prediction is not None:
                extrema_indices = sd_prediction
            for ii, extermum_index in enumerate(extrema_indices):
                if extermum_index > -1:
                    thumbnail = img[:, :, :, extermum_index]
                    thumbnail_keypoints = self._dataset.denormalize_pose(keypoints_prediction[:, :, ii], thumbnail)
                    frames[extermum_index] = draw_kpts(thumbnail, thumbnail_keypoints,
                                                       kpts_connections=self._dataset.kpts_info["connections"],
                                                       colors_pts=self._dataset.kpts_info["colors"])

        seq_plot = plot_grid(frames=frames, labels=frames_inds, thumbnail_size=112)

        ax.imshow(np.array(seq_plot))
        prediction_text = "EF={:.1f}, EF_prediction={:.1f}, EF_L1={:.2f}".format(ef, ef_prediction, abs(ef - ef_prediction))
        if sd_prediction is not None:
            prediction_text = "{} ED={}, ED_prediction={}, ES={}, ES_prediction={}".format(prediction_text,
                                                                                           data["index_frame1"], sd_prediction[0],
                                                                                           data["index_frame2"], sd_prediction[1])
        ax.set_title(prediction_text)
        ax.axis('off')

        return ax

    def _plot_kpts_single_frame(self, fig, data_path_from_root, keypoints_prediction):
        datapoint_index = self._dataset.img_list.index(data_path_from_root)
        data = self._dataset.get_img_and_kpts(datapoint_index)
        img = data["img"]
        keypoints = data["kpts"]
        # normalize:
        keypoints = self._dataset.normalize_pose(keypoints, img)
        img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        keypoints_prediction = self._dataset.denormalize_pose(keypoints_prediction, img)
        keypoints = self._dataset.denormalize_pose(keypoints, img)

        plot_kpts_pred_and_gt(fig, img, gt_kpts=keypoints, pred_kpts=keypoints_prediction,
                              kpts_info=self._dataset.kpts_info, closed_contour=self._dataset.kpts_info['closed_contour'])

        #prediction_text = "Keypoints err: {:.2f}".format(np.mean(dist_pred_gt_kpts[img_index]))
        return fig

    def _plot_navigation_histograms(self, towards_target: np.ndarray) -> plt.Figure:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        metrics = ['x[mm]', 'y[mm]', 'z[mm]', 'roll[\u00b0]', 'pitch[\u00b0]', 'yaw[\u00b0]']
        positive_move = towards_target >= 0
        mean_positive_move_per_coordinate = np.mean(positive_move, axis=0) * 100
        for rr in range(2):
            for cc in range(3):
                coordinate_index = cc + rr * 3
                axs[rr, cc].hist(towards_target[:, coordinate_index])
                axs[rr, cc].set_title("{}, positive move in {:.1f}% of frames".format(metrics[coordinate_index],
                                                                                      mean_positive_move_per_coordinate[coordinate_index]))
        return fig

    def _plot_ef_scatters(self, ef: np.ndarray, ef_prediction: np.ndarray) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(18, 18))
        metrics = ['%', '%']
        labels = ['ef', 'ef prediction']

        for rr in range(1):
            for cc in range(1):
                L1 = abs(ef - ef_prediction)
                L1_mean, L1_std = np.mean(L1), np.std(L1)
                axs[rr, cc].plot(ef, ef_prediction, marker='.', linestyle='None', color='black', markersize=2.5)
                axs[rr, cc].set(xlabel=labels[0], ylabel=labels[1])
                axs[rr, cc].set_title("EF[{}] vs. EF Prediction. L1: Mean={}, Std={}".format(metrics[0], L1_mean, L1_std))
                axs[rr, cc].set_aspect('equal', adjustable='box')
                axs[rr, cc].plot([0, 1], [0, 1], color='red', ls="--", transform=axs[rr, cc].transAxes)

        return fig
