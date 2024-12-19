# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pandas as pd
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader


def _split_train_val(df, train_proportion):
    num_train = int(df.shape[0] * train_proportion)
    train_df = df.iloc[:num_train].copy()
    valid_df = df.iloc[num_train:].copy()

    return train_df, valid_df


class VerticalDataLoader(XGBDataLoader):
    def __init__(self, data_split_path, label_owner, train_proportion):
        """Reads intersection of dataset and returns train and validation XGB dataset matrices with column split mode.

        Args:
            data_split_path: path to dataset split file
            label_owner: client id that owns the label
            train_proportion: proportion of intersected dataset to use for training
        """
        self.data_split_path = data_split_path
        self.label_owner = label_owner
        self.train_proportion = train_proportion

    def load_data(self):
        client_data_split_path = self.data_split_path.replace("site-x", self.client_id)

        data_split_dir = os.path.dirname(client_data_split_path)
        train_path = os.path.join(data_split_dir, "train.csv")
        valid_path = os.path.join(data_split_dir, "valid.csv")

        if not (os.path.exists(train_path) and os.path.exists(valid_path)):
            df = pd.read_csv(client_data_split_path)
            train_df, valid_df = _split_train_val(df, self.train_proportion)

            train_df.to_csv(path_or_buf=train_path, header=False, index=False)
            valid_df.to_csv(path_or_buf=valid_path, header=False, index=False)

        if self.client_id == self.label_owner:
            label = "&label_column=0"
        else:
            label = ""

        # for Vertical XGBoost, read from csv with label_column and set data_split_mode to 1 for column mode
        dtrain = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)
        dvalid = xgb.DMatrix(valid_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)

        return dtrain, dvalid
