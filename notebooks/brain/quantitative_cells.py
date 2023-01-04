import glob

import celldetection as cd
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def read_label_h5(path):
    with h5py.File(path, "r") as file:
        labels = file["labels"][()]
    return labels


class LabelsDataset(Dataset):
    def __init__(self, folder, model):
        self.model = model
        self.file_names = sorted(glob.glob(f"{folder}/{model}/*.hdf5"))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.file_names[idx]
        labels_tar = read_label_h5(path.replace(self.model, "blurry"))
        labels_gen = read_label_h5(path)
        return labels_tar, labels_gen


def avg_iou_scores(results: cd.data.LabelMatcherList, metric="f1", iou_threshs=(.5, .6, .7, .8, .9)):
    assert metric in ["f1", "true_positives", "false_positives", "false_negatives"]
    scores = []
    result_dict = {}
    for results.iou_thresh in iou_threshs:
        res = results._avg_x(metric)
        result_dict[results.iou_thresh] = res
        scores.append(res)
    final_metric = np.mean(scores)
    result_dict["avg"] = final_metric
    return result_dict


def analyse_cell_counts(model, folder, z_depth, dest):
    labels_dset = LabelsDataset(folder, model)

    results = {z: [] for z in range(z_depth)}
    results["min"] = []
    for i in range(len(labels_dset)):
        labels_tar, labels_gen = labels_dset[i]
        for z in range(labels_tar.shape[0]):
            res = cd.data.LabelMatcher(labels_gen[z], labels_tar[z])
            if z == z_depth:
                z = "min"
            results[z].append(res)

    metrics = ["f1", "false_positives", "false_negatives"]
    scores = {z: {m: None for m in metrics} for z in results.keys()}
    for z in results.keys():
        matcher_list = cd.data.LabelMatcherList(results[z])
        for m in metrics:
            scores[z][m] = avg_iou_scores(matcher_list, m)

    rows = []
    for row, cols in scores.items():
        for col, subcols in cols.items():
            for subcol, value in subcols.items():
                rows.append({'Layer': row, 'Metric': col, 'IoU': subcol, 'value': value})

    df = pd.DataFrame(rows, columns=['Layer', 'Metric', 'IoU', 'value'])
    df = df.pivot(index='Layer', columns=['Metric', 'IoU'], values='value')
    df.to_excel(f"{dest}/quantitative_{model}.xlsx")
    return df
