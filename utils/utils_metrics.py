import logging
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from utils.utils_echonet_dynamic import calculateVolume, volumeMethodOfDisks
from utils.utils_stat import pre_process_kpts_for_volume_calc, contour2volume
from utils.utils_contour import LeftVentricleUnorderedContour, echonet_trace_to_mask

def get_intersection_and_sums(prediction: np.float, target: np.float) -> tuple:
    """
    Calculates the intersection between output and target as well as the individual sums of each
    Args: prediction mask, target mask
    Returns: intersection (prediction AND target), sum of all positive prediction pixels,
     sum of all positive target pixels
    """
    intersection = np.sum(prediction * target).astype(np.float)
    output_sum = np.sum(prediction).astype(np.float)
    target_sum = np.sum(target).astype(np.float)
    return intersection, output_sum, target_sum

def get_largest_contour(contours):
    """ find all contours above threshold """
    largest = None
    current_biggest = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > current_biggest:
            largest = contour
            current_biggest = area
    if largest is None:
        raise ValueError("no contours in image > 0 area")
    return [largest]


def compute_ef_from_masks_echonet(mask1, mask2):
    """ compute ef from masks using the method provided by echonet dynamic """
    volume1, x1s, y1s, x2s, y2s, degrees = calculateVolume(mask1, 20, sweeps=0)
    volume2, x1s, y1s, x2s, y2s, degrees = calculateVolume(mask2, 20, sweeps=0)
    edv = max(volume1[0], volume2[0])
    eds = min(volume1[0], volume2[0])
    ef = 100 * np.abs(edv - eds) / edv
    return ef

def compute_ef_from_pts_guy(pts1, pts2):
    """ compute ef by the equation from guy """
    pts1 = pre_process_kpts_for_volume_calc(pts1, closed_contour=False, input_size=112)
    pts2 = pre_process_kpts_for_volume_calc(pts2, closed_contour=False, input_size=112)

    volume1 = contour2volume(pts1[0], pts1[1])
    volume2 = contour2volume(pts2[0], pts2[1])

    edv = max(volume1, volume2)
    eds = min(volume1, volume2)
    ef = 100 * np.abs(edv - eds) / edv
    return ef


def compute_ef_from_comp_points(pts1, pts2, sorting=True):
    """ compute ef by andy sorting and echonet point handling"""
    if sorting == True:
        lvcontour1 = LeftVentricleUnorderedContour(contour=pts1)
        lvcontour2 = LeftVentricleUnorderedContour(contour=pts2)
        pts1 = np.array(lvcontour1.to_ordered_contour(num_pts=40)["myo"]).swapaxes(0, 1)
        pts2 = np.array(lvcontour2.to_ordered_contour(num_pts=40)["myo"]).swapaxes(0, 1)
        #here the apex from the interpolated points is used to get a better estimate for the apex point
        apex1 = lvcontour1.apex
        apex2 = lvcontour2.apex

        basal_mid_point1 = np.array([(pts1[0][0] + pts1[-1][0]) / 2, (pts1[0][1] + pts1[-1][1]) / 2])
        basal_mid_point2 = np.array([(pts2[0][0] + pts2[-1][0]) / 2, (pts2[0][1] + pts2[-1][1]) / 2])

        pts1_sort = []
        pts2_sort = []
        for n in range(0, int(len(pts1) / 2)):
            pts1_sort.append([pts1[n, 0], pts1[n, 1], pts1[-n + 1, 0], pts1[-n + 1, 1]])
            pts2_sort.append([pts2[n, 0], pts2[n, 1], pts2[-n + 1, 0], pts2[-n + 1, 1]])

        pts1 = np.array(pts1_sort)
        pts2 = np.array(pts2_sort)

    else:
        pts1_sort = []
        pts2_sort = []
        for n in range(0, int(len(pts1) / 2)):
            pts1_sort.append([pts1[n, 0], pts1[n, 1], pts1[-n + 1, 0], pts1[-n + 1, 1]])
            pts2_sort.append([pts2[n, 0], pts2[n, 1], pts2[-n + 1, 0], pts2[-n + 1, 1]])

        basal_mid_point1 = np.array([(pts1_sort[0][0] + pts1_sort[0][2]) / 2,
                                     (pts1_sort[0][1] + pts1_sort[0][3]) / 2])
        basal_mid_point2 = np.array([(pts2_sort[0][0] + pts2_sort[0][2]) / 2,
                                     (pts2_sort[0][1] + pts2_sort[0][3]) / 2])

        # get the apex from the upmost point pair which is not that accurate,
        # the resulting contour should be interpolated instead
        # it is still not quite clear how it is done for the gt from the echonet xlsx sheet

        apex1 = np.array([(pts1_sort[-1][0] + pts1_sort[-1][2]) / 2,
                          (pts1_sort[-1][1] + pts1_sort[-1][3]) / 2])
        apex2 = np.array([(pts2_sort[-1][0] + pts2_sort[-1][2]) / 2,
                          (pts2_sort[-1][1] + pts2_sort[-1][3]) / 2])

        pts1 = np.array(pts1_sort)
        pts2 = np.array(pts2_sort)

    volume1 = volumeMethodOfDisks(apex1[0], apex1[1], basal_mid_point1[0], basal_mid_point1[1],
                                  20,
                                  [pts1[:, 0], pts1[:, 1]], [pts1[:, 2], pts1[:, 3]])
    volume2 = volumeMethodOfDisks(apex2[0], apex2[1], basal_mid_point2[0], basal_mid_point2[1],
                                  20,
                                  [pts2[:, 0], pts2[:, 1]], [pts2[:, 2], pts2[:, 3]])

    edv = max(volume1, volume2)
    eds = min(volume1, volume2)
    ef = 100 * np.abs(edv - eds) / edv
    return ef


def compute_ef_from_gt_points_echonet(pts1, pts2, computeApex=True):
    """
    compute ef from gt points if all 42 points are provided in the given format
    x1, y1, x2, y2 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    """
    volume1 = volumeMethodOfDisks(pts1[0, 0], pts1[0, 1], pts1[0, 2], pts1[0, 3],
                                  20,
                                  [pts1[1:, 0], pts1[1:, 1]], [pts1[1:, 2], pts1[1:, 3]])
    volume2 = volumeMethodOfDisks(pts2[0, 0], pts2[0, 1], pts2[0, 2], pts2[0, 3],
                                  20,
                                  [pts2[1:, 0], pts2[1:, 1]], [pts2[1:, 2], pts2[1:, 3]])

    edv = max(volume1, volume2)
    eds = min(volume1, volume2)
    ef = 100 * np.abs(edv - eds) / edv

    return ef


def compute_ef_from_gt_points_self(pts1, pts2, computeApex=True):
    """ alternative method for ef calculation (not used) """
    # x1, y1, x2, y2 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    l1 = ((pts1[0, 0] - pts1[0, 2]) ** 2 + (pts1[0, 1] - pts1[0, 3]) ** 2) ** 0.5
    l2 = ((pts2[0, 0] - pts2[0, 2]) ** 2 + (pts2[0, 1] - pts2[0, 3]) ** 2) ** 0.5

    area1 = 0.0
    area2 = 0.0

    for pt in pts1[1:]:
        disc_area = math.pi * np.sum((np.array([pt[0], pt[1]]) - np.array([pt[2], pt[3]])) ** 2) / 4
        area1 += disc_area

    for pt in pts2[1:]:
        disc_area = math.pi * np.sum((np.array([pt[0], pt[1]]) - np.array([pt[2], pt[3]])) ** 2) / 4
        area2 += disc_area

    volume1 = area1 * l1
    volume2 = area2 * l2

    edv = max(volume1, volume2)
    eds = min(volume1, volume2)
    ef = 100 * np.abs(edv - eds) / edv

    return ef


class MetricCollection:
    """ class to handle a set of outputs """

    def __init__(self, filenames, predictions, targets, input_type='kpts', dataset_name=''):
        self.predictions = predictions
        self.targets = targets
        self.filenames = filenames
        self.results_dataframe = pd.DataFrame()
        self.dataset_name = dataset_name
        # self.prepare_data(input_type)
        self.input_type = input_type
        # todo check outputs

    def prepare_data(self, input_type):
        if input_type == 'kpts':
            self.predictions_kpts = self.predictions
            self.targets_kpts = self.targets
            self.predictions_mask = self.convert_to_mask(self.predictions)
            self.targets_mask = self.convert_to_mask(self.targets)
        elif input_type == 'mask':
            self.predictions_mask = self.predictions
            self.targets_mask = self.targets
            self.predictions_kpts = self.convert_to_keypoints(self.predictions)
            self.targets_kpts = self.convert_to_keypoints(self.targets)
        else:
            assert ('wrong input type selected')

    def convert_to_mask(self, kpts, img_size=112):

        if len(kpts.shape)==2:  # if shape is (X,Y)
            kpts_sort = []
            for n in range(0, int(len(kpts) / 2)):
                kpts_sort.append([kpts[n, 0], kpts[n, 1], kpts[-(n + 1), 0], kpts[-(n + 1), 1]])
            kpts_sort = np.array(kpts_sort)
            mask = echonet_trace_to_mask(kpts_sort, (img_size, img_size))
        else:  # X1,Y1,X2,Y2
            mask = echonet_trace_to_mask(kpts, (img_size, img_size))

        return mask

    def convert_to_keypoints(self, mask, num_kpts=40):
        lvc = LeftVentricleUnorderedContour(mask=mask)
        kpts = lvc.to_ordered_contour(num_pts=num_kpts)["myo"]
        kpts = np.asarray(kpts).swapaxes(0, 1)
        return kpts

    def compute_statistics(self, external_ef_df=None, single_frame=False):

        results = []
        if single_frame == True:

            for i, j, k in zip(self.filenames, self.predictions, self.targets):
                if self.input_type == 'kpts':
                    j_mask = self.convert_to_mask(j)
                    k_mask = self.convert_to_mask(k)
                    j_kpts = j
                    k_kpts = k
                elif self.input_type == 'mask':
                    j_kpts = self.convert_to_keypoints(j)
                    k_kpts = self.convert_to_keypoints(k)
                    j_mask = j
                    k_mask = k
                else:
                    assert ('Input type error: Wrong input type selected')

                metric_points = PointsMetrics(i, j_kpts, k_kpts)
                result_points = metric_points.compute_metrics()
                metric_volume = VolumeMetrics(i, j_mask, k_mask)
                result_volume = metric_volume.compute_metrics()

                results.append({'name': i, **result_volume[i], **result_points[i]})

        else:
            for i in range(0, len(self.filenames), 2):
                if self.filenames[i].split('_')[0] == (self.filenames[i + 1].split('_')[0]):
                    if self.input_type == 'kpts':
                        j_mask_1 = self.convert_to_mask(self.predictions[i])
                        k_mask_1 = self.convert_to_mask(self.targets[i])
                        j_kpts_1 = self.predictions[i]
                        k_kpts_1 = self.targets[i]
                        j_mask_2 = self.convert_to_mask(self.predictions[i + 1])
                        k_mask_2 = self.convert_to_mask(self.targets[i + 1])
                        j_kpts_2 = self.predictions[i + 1]
                        k_kpts_2 = self.targets[i + 1]
                    elif self.input_type == 'mask':
                        j_kpts_1 = self.convert_to_keypoints(self.predictions[i])
                        k_kpts_1 = self.convert_to_keypoints(self.targets[i])
                        j_mask_1 = self.predictions[i]
                        k_mask_1 = self.targets[i]
                        j_kpts_2 = self.convert_to_keypoints(self.predictions[i + 1])
                        k_kpts_2 = self.convert_to_keypoints(self.targets[i + 1])
                        j_mask_2 = self.predictions[i + 1]
                        k_mask_2 = self.targets[i + 1]
                    else:
                        assert ('Input type error: Wrong input type selected')

                    metric_points = PointsMetrics(self.filenames[i], j_kpts_1, k_kpts_1)
                    result_points_1 = metric_points.compute_metrics()
                    metric_volume = VolumeMetrics(self.filenames[i], j_mask_1, k_mask_1)
                    result_volume_1 = metric_volume.compute_metrics()
                    metric_points = PointsMetrics(self.filenames[i + 1], j_kpts_2, k_kpts_2)
                    result_points_2 = metric_points.compute_metrics()
                    metric_volume = VolumeMetrics(self.filenames[i + 1], j_mask_2, k_mask_2)
                    result_volume_2 = metric_volume.compute_metrics()

                    metric_pairs_points = PointsPairMetrics(self.filenames[i], self.filenames[i + 1],
                                                            j_kpts_1, j_kpts_2, k_kpts_1, k_kpts_2)

                    metric_pairs_volume = VolumePairMetrics(self.filenames[i], self.filenames[i + 1],
                                                            j_mask_1, j_mask_2, k_mask_1, k_mask_2)
                    real_ef = -1
                    if external_ef_df is not None:
                        real_ef = external_ef_df[self.filenames[i].split('.')[0].split('_')[0]]

                    result_pairs_points = metric_pairs_points.compute_metrics(real_ef)
                    result_pairs_volume = metric_pairs_volume.compute_metrics(real_ef)

                    results.append({'name': self.filenames[i], **result_volume_1[self.filenames[i]],
                                    **result_points_1[self.filenames[i]],
                                    **result_pairs_points[self.filenames[i]],
                                    **result_pairs_volume[self.filenames[i]]})
                    results.append({'name': self.filenames[i + 1], **result_volume_2[self.filenames[i + 1]],
                                    **result_points_2[self.filenames[i + 1]],
                                    **result_pairs_points[self.filenames[i + 1]],
                                    **result_pairs_volume[self.filenames[i + 1]]})

        self.results_dataframe = pd.DataFrame(results, columns=list(results[0].keys()))
        print(self.results_dataframe)
        for el in list(results[0].keys()):
            if el != 'name':
                print('Metric:', el, self.results_dataframe.mean()[el], self.results_dataframe.std()[el])

    def print_results(self, outputfolder, plot=True):
        self.results_dataframe.to_csv(outputfolder + self.dataset_name + '.csv', sep=',')
        if plot == True:
            self.plot_outcome(outputfolder)

    def get_results(self):
        return self.results_dataframe

    def plot_outcome(self, outputfolder):
        len_plt = int(np.ceil((len(self.results_dataframe.keys())-1)/2)) #exclude name

        fig, axs = plt.subplots(2,len_plt, figsize=(5*len_plt, 8))
        fig.suptitle(self.dataset_name)
        i=0
        j=0
        for (columnName, columnData) in self.results_dataframe.iteritems():
            if columnName != 'name':
                sns.boxplot(data=self.results_dataframe, y=columnName,
                            ax=axs[j][i], color='darkblue', boxprops=dict(alpha=.3))

                if(len_plt-1==i):
                    j += 1
                i = (i+1) % len_plt

        plt.show()

class MetricBase:
    def __init__(self, filename, prediction, target):
        self.prediction = prediction
        self.target = target
        self.filename = filename
        self.metrics = dict()
        self.metrics[filename] = {}


class MetricPairBase:
    def __init__(self, filename_1, filename_2, prediction_1, prediction_2, target_1, target_2):
        self.prediction_1 = prediction_1
        self.target_1 = target_1
        self.prediction_2 = prediction_2
        self.target_2 = target_2
        self.filename_1 = filename_1
        self.filename_2 = filename_2
        self.metrics = dict()
        self.metrics[filename_1] = {}
        self.metrics[filename_2] = {}


class VolumePairMetrics(MetricPairBase):
    """
    Class for all metrics derived from masks that require two input frames - ejection fraction (ef)
    :param filename: used to always link the name to the measurement
    :param prediction: mask provided by the network
    :param target: ground truth mask
    :Returns: metric dict with ef related metrics (same for each output frame)
    """
    def __init__(self, filename_1, filename_2, prediction_1, prediction_2, target_1, target_2):
        MetricPairBase.__init__(self, filename_1, filename_2, prediction_1, prediction_2,
                                target_1, target_2)

    def compute_metrics(self, external_ef):
        if external_ef == -1:
            reference_ef = compute_ef_from_masks_echonet(self.target_1, self.target_2)
        else:
            reference_ef = external_ef

        computed_ef = compute_ef_from_masks_echonet(self.prediction_1, self.prediction_2)
        self.metrics[self.filename_1]['ef_error_masks'] = abs(computed_ef - reference_ef)
        self.metrics[self.filename_2]['ef_error_masks'] = abs(computed_ef - reference_ef)

        return self.metrics


class PointsPairMetrics(MetricPairBase):
    """
    Class for all metrics derived from keypoints that require two input frames - ejection fraction (ef)
    :param filename: used to always link the name to the measurement
    :param prediction: keypoints provided by the network
    :param target: ground truth keypoints
    :Returns: metric dict with ef related metrics (same for each output frame)
    """
    def __init__(self, filename_1, filename_2, prediction_1, prediction_2, target_1, target_2):
        MetricPairBase.__init__(self, filename_1, filename_2, prediction_1, prediction_2,
                                target_1, target_2)

    def compute_metrics(self, external_ef):
        if external_ef == -1:
            reference_ef = compute_ef_from_pts_guy(self.target_1, self.target_2)
        else:
            reference_ef = external_ef

        computed_ef = compute_ef_from_pts_guy(self.prediction_1, self.prediction_2)
        self.metrics[self.filename_1]['ef_error_points'] = abs(computed_ef - reference_ef)
        self.metrics[self.filename_2]['ef_error_points'] = abs(computed_ef - reference_ef)

        return self.metrics


class VolumeMetrics(MetricBase):
    """
    Class for all metrics derived from binary masks
    :param filename: used to always link the name to the measurement
    :param prediction: mask provided by the network
    :param target: ground truth mask
    :Returns: metric dict with different volume related metrics such as dice, iou (jaccard), bias, simplicity and convexity
    """
    def __init__(self, filename, prediction, target):
        MetricBase.__init__(self, filename, prediction, target)

    def compute_metrics(self):
        self.metrics[self.filename]['dice'] = self.get_dice_score(self.prediction, self.target)
        self.metrics[self.filename]['iou'] = self.get_iou(self.prediction, self.target)
        self.metrics[self.filename]['bias'] = self.get_bias(self.prediction, self.target)
        self.metrics[self.filename]['simplicity'] = self.get_simplicity_deviation(self.prediction, self.target)
        self.metrics[self.filename]['convexity'] = self.get_convexity_deviation(self.prediction, self.target)
        return self.metrics

    def get_dice_score(self, prediction, target) -> float:
        intersection, output_sum, target_sum = get_intersection_and_sums(prediction, target)
        dice = 2 * intersection / (output_sum + target_sum)
        return dice

    def get_iou(self, prediction, target) -> float:
        intersection, output_sum, target_sum = get_intersection_and_sums(prediction, target)
        iou = intersection / (output_sum + target_sum - intersection)
        return iou

    def get_simplicity_deviation(self, prediction, target) -> float:
        simplicity_target = self.get_simplicity(target)
        simplicity_prediction = self.get_simplicity(prediction)
        return abs(simplicity_target - simplicity_prediction)

    @staticmethod
    def get_simplicity(mask) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            largest = get_largest_contour(contours)[0]
        except ValueError as e:
            logging.warning(f"finding metric Simplicity failed because {e}, skipping...")
            return None
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        return np.sqrt(4 * np.pi * area) / perimeter

    def get_convexity_deviation(self, prediction, target) -> float:
        convexity_target = self.get_convexity(target)
        convexity_prediction = self.get_convexity(prediction)
        return abs(convexity_target - convexity_prediction)

    @staticmethod
    def get_convexity(mask) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            largest = get_largest_contour(contours)[0]
        except ValueError as e:
            logging.warning(f"finding metric Convexity failed because {e}, skipping...")
            return None
        area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        return area / hull_area

    def get_bias(self, prediction, target) -> float:
        intersection, output_sum, target_sum = get_intersection_and_sums(prediction, target)
        bias = (output_sum - target_sum) / (0.5 * output_sum + 0.5 * target_sum)
        return bias


class PointsMetrics(MetricBase):
    """
    Class for all metrics derived from the keypoints
    :param filename: used to always link the name to the measurement
    :param prediction: keypoints provided by the network
    :param target: keypoints ground truth
    """

    def __init__(self, filename, prediction, target):
        MetricBase.__init__(self, filename, prediction, target)
        self.check_input_format(prediction, target)

    def compute_metrics(self):
        mean_kpts_error, max_kpts_error = self.kpts_error(self.prediction, self.target)
        self.metrics[self.filename]['mean_kpts_error'] = mean_kpts_error
        self.metrics[self.filename]['max_kpts_error'] = max_kpts_error
        return self.metrics

    def check_input_format(self, prediction, target):
        """ check input format of outpt and target
        output: BxCxHxW tensor where C is the number of classes
        target: Bx1xHxW tensor where every entry is an int in the range [0, C-1]
        """
        try:
            assert len(prediction.shape) == 2
            assert len(prediction.shape) == len(target.shape)
            assert len(target.shape) == 2
            assert target.shape[1] == 2
        except AssertionError:
            raise ValueError(f"Shape error: \nOutput should be [keypoint_num,pts_dim], found {prediction.shape} "
                             f"\nTarget should be [keypoint_num,pts_dim], found {target.shape}. ")

    def kpts_error(self, prediction, target) -> float:
        kpts_dist = np.zeros(len(prediction))

        for kpt_indx in range(len(prediction)):
            pred_pt = np.array(prediction[kpt_indx])
            target_pt = np.array(target[kpt_indx])
            dist_kpts = np.linalg.norm(pred_pt - target_pt)
            kpts_dist[kpt_indx] = dist_kpts

        return np.mean(kpts_dist), np.max(kpts_dist)


