import os
import matplotlib.pyplot as plt
import math
import scipy.interpolate as interpolate
import numpy as np

########################################
########################################
# Metrics
#######################################
########################################

def kpts_dist_error(ds, total_filenames, total_gt_kpts, total_output_kpts):
    dist_pred_gt_kpts = np.zeros([len(total_filenames),len(total_gt_kpts[0])])
    if ds.testset.anno_dir is not None:
        for ii, filename in enumerate(total_filenames):
            dist_pred_gt_kpts[ii] = 100 * match_two_kpts_set(total_gt_kpts[ii],
                                                             total_output_kpts[ii])#     / ds.testset.input_img_size
    return dist_pred_gt_kpts

def find_consistency_Andy(test_movie_pred_kpts):
    kpts_pvs_frame = test_movie_pred_kpts[0]
    consistency = 0
    num_frames = 0
    for kpts_this_frame in test_movie_pred_kpts[1:]:
        consistency += np.linalg.norm(kpts_this_frame - kpts_pvs_frame)
        num_frames += 1
        kpts_pvs_frame = kpts_this_frame
    consistency /= num_frames
    return consistency

def find_kpt_consistencies_Andy(test_movie_pred_kpts):
    kpts_pvs_frame = test_movie_pred_kpts[0]
    consistency = 0
    num_frames = 0
    indices = dict(basal_left=0, basal_right=-1, apex=find_apex_index_Andy(kpts_pvs_frame))
    consistencies = dict(basal_left=0, basal_right=0, apex=0)

    for kpts_this_frame in test_movie_pred_kpts[1:]:

        for kpt_name in consistencies:
            kpt_index = indices[kpt_name]
            consistencies[kpt_name] += np.linalg.norm(kpts_this_frame[kpt_index]-kpts_pvs_frame[kpt_index])
        kpts_pvs_frame = kpts_this_frame
    return consistencies

############################
############################
# Eval utils
############################
############################
def match_two_kpts_set(kpts1, kpts2):
    """
    :param kpts1: normalized keypoints set1
    :param kpts2: normalized keypoints set2
    :return:
    """

    kpts_dist = np.zeros(len(kpts1))

    for kpt_indx in range(len(kpts1)):

        PA = np.array(kpts1[kpt_indx])
        PO = np.array(kpts2[kpt_indx])
        dist_kpts = np.linalg.norm(PA - PO)
        kpts_dist[kpt_indx] = dist_kpts

    return kpts_dist

def contour2volume(x, y):   # from Dani Pinkovich:

    n_discs = 20

    middle_ind = int(len(x)/2)
    index_step_size = middle_ind / n_discs

    start_coords = np.array([(x[0]+x[-1])/2, (y[0]+y[-1])/2])
    middle_coords = np.array([x[middle_ind], y[middle_ind]])
    interval_along_axis = np.linalg.norm(start_coords - middle_coords) / n_discs

    volume = 0
    for i in range(n_discs):
        ind_left = int(index_step_size*(i+0.5))
        left_point = np.array([x[ind_left], y[ind_left]])
        ind_right = int(len(x) - ind_left)
        right_point = np.array([x[ind_right], y[ind_right]])
        disc_area = math.pi * np.sum((right_point-left_point)**2) / 4
        volume += disc_area * interval_along_axis

    # Orig by Dani Pinkovich:
    """
    apex_ind = np.argmin(y)
    base_coords = np.array([(x[0]+x[-1])/2, (y[0]+y[-1])/2])
    apex_coords = np.array([x[apex_ind], y[apex_ind]])

    lv_length = np.sqrt(np.sum((apex_coords-base_coords)**2))
    n_discs = 20
    disc_height = lv_length / n_discs

    volume = 0
    for i in range(n_discs):
        ind_left = int(apex_ind/n_discs*(i+0.5))
        left_point = np.array([x[ind_left], y[ind_left]])
        ind_right = int(len(x) - (len(x)-apex_ind)/n_discs*(i+0.5))
        right_point = np.array([x[ind_right], y[ind_right]])

        disc_area = math.pi * np.sum((right_point-left_point)**2) / 4
        volume += disc_area * disc_height
    """

    return volume

def contour2mask(x, y, thickness=1, mask_size=(224, 224)):
        mask = np.zeros(mask_size)
        for t in range(len(x)):
            for ii in range(-thickness, thickness, 1):
                for jj in range(-thickness, thickness, 1):
                    mask[int(x[t]) + ii, int(y[t]) + jj] = 1
        return mask.astype(bool)

def compute_mask_iou(predicted_mask, groundtruth_mask):
    overlap = groundtruth_mask * predicted_mask  # Logical AND
    union = groundtruth_mask + predicted_mask  # Logical OR
    iou = overlap.sum() / float(union.sum())  # Treats "True
    return iou

def pre_process_kpts_for_volume_calc(kpts, closed_contour=False, input_size=224):
    interpolation_step_size = 0.001
    unew = np.arange(0, 1.00, interpolation_step_size)

    if closed_contour:
        kpts = np.concatenate((kpts[:, :], kpts[:1, :]), axis=0)

    tck, u = interpolate.splprep([kpts[:, 0], kpts[:, 1]], s=0)
    out = interpolate.splev(unew, tck)

    out[1] = input_size - out[1]    # to match coordinates

    # normalize (option) to 100x100 grid:

    out = [100 * out[0] / input_size, 100 * out[1] / input_size]


    #out = [out[0] / input_size, out[1] / input_size]

    return out

def compute_apex_for_contour(contour_points):

    contour_points = np.array(contour_points).transpose()   # ndarray 1000x2
    side1 = np.linalg.norm(contour_points - contour_points[0], axis=1)
    side2 = np.linalg.norm(contour_points - contour_points[-1], axis=1)

    max_pt_index = np.argmax(side1 + side2)
    max_pt_dist = np.max(side1 + side2)

    # base_pt1 = np.array([contour_points[0][0], contour_points[1][0]])
    # base_pt2 = np.array([contour_points[0][-1], contour_points[1][-1]])
    # for pt_index in range(len(contour_points[0])):
    #     pt = np.array([contour_points[0][pt_index], contour_points[1][pt_index]])
    #     pt_dist = np.linalg.norm(pt - base_pt1) + np.linalg.norm(pt - base_pt2)
    #     if pt_dist > max_pt_dist:
    #         max_pt_dist = pt_dist
    #         max_pt_index = pt_index

    apex_pts = np.asarray([contour_points[0], contour_points[max_pt_index], contour_points[-1]])

    return max_pt_index, max_pt_dist, apex_pts


def calc_apex_error(kpts1, kpts2, closed_contour=False, input_size=224):
    out1 = pre_process_kpts_for_volume_calc(kpts1, closed_contour, input_size)
    out2 = pre_process_kpts_for_volume_calc(kpts2, closed_contour, input_size)

    max_pt_index1, max_pt_dist1, apex_pts1 = compute_apex_for_contour(out1)
    max_pt_index2, max_pt_dist2, apex_pts2 = compute_apex_for_contour(out2)
    #print("hi")

    return np.linalg.norm(apex_pts1 - apex_pts2)


def find_apex_index_Andy(contour):
    assert contour.shape[0] > 3, "contour should be shape Nx2 where N>3"
    possible_points = contour[1:-1]  # exclude basal points from consideration
    # Find the distance from each point to the two basal points
    distances = np.linalg.norm(possible_points-contour[0], axis=1) + np.linalg.norm(possible_points-contour[-1], axis=1)
    return np.argmax(distances)

def internal_stats(area_gt, area_pred, total_filenames):
    plt.clf()
    plt.scatter(area_gt, area_pred)
    for i, label in enumerate(total_filenames):
        plt.annotate(
            os.path.basename(label)[-8:-4],
            xy=(area_gt[i], area_pred[i]), xytext=(-5, 5),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.plot([1000, 4000], [1000, 4000])
    plt.xlim(1000, 4000)
    plt.ylim(1000, 4000)
    plt.xlabel('gt')
    plt.ylabel('pred')


# ===========================
# Main
# ===========================
if __name__ == '__main__':
    pass
