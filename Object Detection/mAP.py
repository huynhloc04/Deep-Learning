
import torch
from IoU import intersection_over_union 
from collections import Counter

def mean_acerage_precision(
        pred_bboxes, true_bboxes, iou_threshold = 0.5, box_format = 'pascal_voc', no_classes = 20
):
    """
    Args:
            pred_bboxes: list of lists contraining all bboxes with each bbox specified as:
                            [img_idx, class_idx, prob_score, x1, y1, x2, y2]
            true_bboxes: Similar as pred_bboxes except all the ones 
            iou_threshold: threshold where predicted bboxes is correct
            no_classes: number of classes

    Returns:
            float: mAP value across all classes given a specific IoU threshold
    """

    average_precision = []
    eps = 1e-7

    for c in range(no_classes):
        #   Go through all predictions and targets, and only add the ones that belong to the current class c
        detections = [pred_bbox for pred_bbox in pred_bboxes if pred_bbox[1] == c]
        ground_trues = [true_bbox for true_bbox in true_bboxes if true_bbox[1] == c]

        detections.sort(key = lambda x : x[2], reverse = True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        amount_bboxes = Counter([bbox[0] for bbox in ground_trues])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        if len(ground_trues) == 0:
            continue

        for detect_idx, detect in enumerate(detections):
            best_iou = 0
            ground_true_img = [bbox for bbox in ground_trues if bbox[0] == detect[0]]
            for gt_idx, gt in enumerate(ground_true_img):
                iou = intersection_over_union(torch.tensor(detect[3:]), torch.tensor(gt[3:]), box_format = box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[detect[0]][best_idx] == 0:
                    TP[detect_idx] = 1
                    amount_bboxes[detect[0]][best_idx] = 1
                else:
                    FP[detect_idx] = 1
            else:
                FP[detect_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)
        recall = TP_cumsum / (len(ground_trues) + eps)
        precision = TP_cumsum / (TP_cumsum + FP_cumsum + eps)

        precision = torch.cat((torch.tensor([1]), precision))
        recall = torch.cat((torch.tensor([0]), recall))

        average_precision.append(torch.trapz(precision, recall))

    return torch.mean(average_precision)

