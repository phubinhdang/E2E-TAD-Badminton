import dataclasses
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def ss_to_mmss(num_of_seconds) -> str:
    """Converts start_ss (in seconds) to minute:second format."""
    minutes = int(num_of_seconds // 60)  # Calculate minutes
    seconds = int(num_of_seconds % 60)  # Calculate remaining seconds
    return f"{minutes:02}:{seconds:02}"  # Format as mm:ss


@dataclass
class GroundTruthSegment:
    start_ss: float
    end_ss: float
    start_mmss: str
    end_mmss: str
    is_matched: bool
    score_board: str


@dataclass
class PredictionSegment:
    start_ss: float
    end_ss: float
    start_mmss: str
    end_mmss: str
    best_iou: float
    best_match_gt_segment: Optional[GroundTruthSegment]


fps = 25


def get_gts() -> List[GroundTruthSegment]:
    gts = []
    df_gt = pd.read_csv('data/RallySeg_GT.csv')
    df_gt['start_ss'] = df_gt['Start'] / fps
    df_gt['end_ss'] = df_gt['End'] / fps
    for s, e, score_board in zip(df_gt['start_ss'], df_gt['end_ss'], df_gt['Score']):
        gts.append(GroundTruthSegment(s, e, ss_to_mmss(s), ss_to_mmss(e), False, score_board))
    return gts


def get_preds_hauptprojekt() -> List[PredictionSegment]:
    df_preds = pd.read_csv('data/ginting_axelsen_hauptprojekt.csv')
    pred_segments = []
    for s, e in zip(df_preds['start'], df_preds['end']):
        pred_segments.append(PredictionSegment(s, e, ss_to_mmss(s), ss_to_mmss(e), 0.0, None))
    return pred_segments


def get_preds_grundprojekt() -> List[PredictionSegment]:
    df_preds = pd.read_csv('data/ginting_axelsen_grundprojekt.csv')
    df_preds = df_preds[df_preds['pred_is_rally'] == 1]
    df_preds = df_preds.assign(start_ss=(df_preds['start'] / fps).round(2))
    df_preds = df_preds.assign(end_ss=(df_preds['end'] / fps).round(2))
    raw_pred_segments = []
    for s, e in zip(df_preds['start_ss'], df_preds['end_ss']):
        raw_pred_segments.append(PredictionSegment(s, e, ss_to_mmss(s), ss_to_mmss(e), 0.0, None))
    pred_segments = []
    merged_segment = None
    for i in range(len(raw_pred_segments) - 1):
        current = raw_pred_segments[i]
        next = raw_pred_segments[i + 1]
        if not merged_segment:
            merged_segment = dataclasses.replace(current)
        if current.end_ss == next.start_ss:
            merged_segment.end_ss = next.end_ss
        else:
            pred_segments.append(dataclasses.replace(merged_segment))
            merged_segment = None
        # the last predicted rally
    pred_segments.append(dataclasses.replace(merged_segment))
    return pred_segments


def calc_iou(pred: PredictionSegment, gt: GroundTruthSegment) -> float:
    inter_start = max(pred.start_ss, gt.start_ss)
    inter_end = min(pred.end_ss, gt.end_ss)

    # Calculate the length of the intersection
    intersection = max(0, inter_end - inter_start)

    # Calculate the start and end of the union
    union_start = min(pred.start_ss, gt.start_ss)
    union_end = max(pred.end_ss, gt.end_ss)

    # Calculate the length of the union
    union = union_end - union_start

    # Calculate IoU
    if union == 0:
        return 0.0  # Handle case when both segments are points
    iou = intersection / union
    return iou


def find_best_iou(pred: PredictionSegment, gts: List[GroundTruthSegment]) -> (float, GroundTruthSegment):
    best_iou = 0.0
    best_match_gt_segment = None
    for gt in gts:
        iou = calc_iou(pred, gt)
        if iou > best_iou:
            best_iou = iou
            best_match_gt_segment = gt
    return best_iou, best_match_gt_segment


def create_confusion_matrix(preds: List[PredictionSegment], gts: List[GroundTruthSegment], iou_threshold: float) -> (
        float, float, float):
    # assign iou to each pred segment
    for pred in preds:
        # which gt, for that the pred has the largest iou, the gt wil be then remove from gts
        best_iou, best_gt_match = find_best_iou(pred, gts)
        pred.best_iou = best_iou
        if best_gt_match:
            pred.best_match_gt_segment = dataclasses.replace(best_gt_match)
            gts.remove(best_gt_match)
    # count TP, FP, FN
    tp, fp, fn = 0, 0, 0
    for pred in preds:
        # true positive
        if pred.best_iou >= iou_threshold:
            tp += 1
        # false positive
        else:
            fp += 1
    # based on this tutorial: https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b
    # fn does not include preds that have iou smaller than threshold_iou
    fn = len(gts)
    print("tp =  ", tp)
    print("fp =  ", fp)
    print("fn = ", fn)
    return tp, fp, fn


def calc_precision_and_recall(tp, fp, fn):
    return tp / (tp + fp), tp / (tp + fn)


thresholds = np.arange(start=0.5, stop=0.91, step=0.1)
precisions = []
recalls = []
for i, t in enumerate(thresholds):
    gts = get_gts()
    preds = get_preds_grundprojekt()
    if i == 0:
        print("gts len: ", len(gts))
        print("preds len: ", len(preds))
    print(f"threshold : {t}")
    tp, fp, fn = create_confusion_matrix(preds, gts, t)
    precision, recall = calc_precision_and_recall(tp, fp, fn)
    precisions.append(precision)
    recalls.append(recall)

plt.plot(recalls, precisions, linewidth=4, color="red", marker='o')
plt.xlabel("Recall", fontsize=12, fontweight='bold')
plt.ylabel("Precision", fontsize=12, fontweight='bold')
for i, threshold in enumerate(thresholds):
    plt.text(recalls[i], precisions[i], f'{threshold:.2f}', fontsize=9, ha='right', va='bottom')

plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
plt.show()

precisions = np.array(precisions)
recalls = np.array(recalls)
AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])

print("Average Precision = ", AP)
