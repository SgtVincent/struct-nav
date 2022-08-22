# Modified on https://gist.github.com/sidharthrajaram/7ba1a06c0bf18c42016cede00381484b
import io
import numpy as np
import time 
from bbox import BBox3D
from bbox.metrics import iou_3d
# Selects best score and then suppresses.
# class score + bounding box = (p, x, y, z, l, w, h)
# p: classification score / probability
# x,y,z: location
# l,w,h: dimensions

def iou(box_a, box_b):

    box_a_top_right_corner = [box_a[1]+box_a[4], box_a[2]+box_a[5]]
    box_b_top_right_corner = [box_b[1]+box_b[4], box_b[2]+box_b[5]]

    box_a_area = (box_a[4]) * (box_a[5])
    box_b_area = (box_b[4]) * (box_b[5])

    xi = max(box_a[1], box_b[1])
    yi = max(box_a[2], box_b[2])

    corner_x_i = min(box_a_top_right_corner[0], box_b_top_right_corner[0])
    corner_y_i = min(box_a_top_right_corner[1], box_b_top_right_corner[1])

    intersection_area = max(0, corner_x_i - xi) * max(0, corner_y_i - yi)

    intersection_l_min = max(box_a[3], box_b[3])
    intersection_l_max = min(box_a[3]+box_a[6], box_b[3]+box_b[6])
    intersection_length = intersection_l_max - intersection_l_min

    iou = (intersection_area * intersection_length) / float(box_a_area * box_a[6] + box_b_area * box_b[6]
                                                            - intersection_area * intersection_length + 1e-5)

    return iou


def NMS(original_boxes, iou_th):
    """non-maximum suppression for 3D bounding boxes 
    
    Args:
        original_boxes (np.array): (N,7) bounding boxes, each bounding box has shape [p, x, y, z, l,w,h]
            p: classification score / probability
            x,y,z: location
            l,w,h: dimensions    
            assume +z as upward direction  
        iou_th (float): 3D IOU threshold to perform NMS
    Returns:
        _type_: _description_
    """
    boxes_probability_sorted_indices = np.flip(np.argsort(original_boxes[:, 0]))
    # boxes_probability_sorted = original_boxes[boxes_probability_sorted_indices, :]
    num_boxes = len(original_boxes)
    box_indices = np.copy(boxes_probability_sorted_indices)
    suppressed_box_indices = []
    preserved_box_indices = []
    tmp_suppress = []

    # while len(box_indices) > 0:
    while len(box_indices) > 0:

        if box_indices[0] not in suppressed_box_indices:
            selected_idx = box_indices[0]
            tmp_suppress = []

            for i in range(len(box_indices)):
                if box_indices[i] != selected_idx:
                    selected_iou = iou(original_boxes[selected_idx], original_boxes[box_indices[i]])
                    if selected_iou > iou_th:
                        suppressed_box_indices.append(box_indices[i])
                        tmp_suppress.append(i)

        box_indices = np.delete(box_indices, tmp_suppress, axis=0)
        preserved_box_indices.append(selected_idx)
        box_indices = box_indices[1:]

    return sorted(preserved_box_indices), sorted(suppressed_box_indices)


if __name__ == "__main__":

    iou_threshold = 0.45
    
    # some random test bounding box data, feel free to try out your own.
    box_0 = np.array([0.96, 10, 10, 10, 10, 10, 10])  # should make it
    box_1 = np.array([0.90, 10, 10, 10, 11, 11, 12])

    box_2 = np.array([0.76, 21, 10, 13, 10, 9.5, 7])
    box_3 = np.array([0.80, 20.5, 12, 10, 11, 11, 12])
    box_4 = np.array([0.92, 21.5, 11, 10, 10, 10.3, 10])  # should make it

    box_5 = np.array([0.77, 3.9, 2, 2.5, 4, 6.5, 12])
    box_6 = np.array([0.84, 4, 2, 2.5, 4, 6.6, 10])  # should make it
    box_7 = np.array([0.95, 2.99, 2.65, 4.5, 4, 6.35, 12])  # should make it

    box_8 = np.array([0.84, 32, 33, 69, 33.2, 10.2, 6.5])  # should make it

    box_9 = np.array([0.89, 43, 44, 55.5, 11, 11, 12])
    box_10 = np.array([0.93, 41.4, 46, 56.6, 12, 10, 10])  # should make it

    boxes = np.stack([box_0, box_1, box_2, box_3,
                      box_4, box_5, box_6,
                      box_7, box_8, box_9, box_10], axis=0)
    # boxes = np.stack([box_0, box_1, box_2, box_3, box_4], axis=0)

    print("{} Input Bounding Boxes (p,x,y,z,w,h,l):".format(len(boxes)))
    print(boxes)
    print()

    ############### test 3d_iou #########################
    # iou_mat = np.eye(5)
    # for i in range(5):
    #     for j in range(5):
    #         iou_mat[i, j] = iou(boxes[i, :], boxes[j, :])
    # print(iou_mat)
    # ref 
    # [[1.         0.68870523 0.         0.         0.        ]
    #  [0.68870523 1.         0.         0.01930502 0.        ]
    #  [0.         0.         1.         0.31531532 0.37119282]
    #  [0.         0.01930502 0.31531532 1.         0.65081477]
    #  [0.         0.         0.37119282 0.65081477 1.        ]]
    iou_mat = np.eye(len(boxes))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            iou_mat[i, j] = iou(boxes[i, :], boxes[j, :])
    print(iou_mat)
    
    start = time.time()
    p, s = NMS(boxes, iou_threshold)
    end = time.time()

    print("{} seconds".format(end-start))
    print("{} Post-NMS Bounding Boxes (p,x,y,z,l,w,h):".format(len(p)))
    print(p)
    print()