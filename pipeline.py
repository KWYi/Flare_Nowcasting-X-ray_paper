import os, sys
import numpy as np

from torch.utils.data import Dataset
from pandas import read_csv

class XRS_Dataset_Sliding_Window_Timeseries(Dataset):
    def __init__(self, root, folds, data_type=list, flare_classes=list, logarithm=True,
                 AF=None, BF=None):
        super(XRS_Dataset_Sliding_Window_Timeseries, self).__init__()

        self.root = root
        self.folds = folds
        self.logarithm = logarithm

        self.AF = AF
        self.BF = BF

        offsets = list(range(0, 60))
        input_all_cache = []
        target_all_cache = []

        for fold in folds:
            for fc in flare_classes:
                file_path = os.path.join(
                    self.root,
                    f"fold{fold}",
                    f"fold{fold}_timeseries_{fc}.csv"
                )
                _data = read_csv(file_path)

                if self.AF is not None or self.BF is not None:
                    mask = np.zeros(len(_data), dtype=bool)

                    if self.AF is not None:
                        mask |= _data["after_onset"].isin(self.AF)

                    if self.BF is not None:
                        mask |= _data["before_peak"].isin(self.BF)

                    _data = _data[mask].reset_index(drop=True)

                    if len(_data) == 0:
                        continue

                input_datatype_cache = []
                for d_type in data_type:
                    input_cols = [f"{d_type}_seq{m:02d}" for m in offsets]
                    input_datatype_cache.append(
                        _data.loc[:, input_cols].to_numpy(dtype=np.float32)
                    )

                input_parts = np.stack(input_datatype_cache, axis=-1)
                target_parts = _data["target_real_peak_flux"].to_numpy(dtype=np.float32)[:, None]

                input_all_cache.append(input_parts)
                target_all_cache.append(target_parts)

        self.input_data = np.concatenate(input_all_cache, axis=0)
        self.target_data = np.concatenate(target_all_cache, axis=0)

        if self.logarithm is True:
            self.input_data = np.log10(self.input_data + 1e-9)
            self.target_data = np.log10(self.target_data + 1e-9)

        self.len = len(self.input_data)

    def __getitem__(self, idx):
        _input = self.input_data[idx]
        _target = self.target_data[idx]

        return _input, _target

    def __len__(self):
        return self.len

