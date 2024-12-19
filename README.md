# Vertical Federated XGBoost
This project implements Vertical Federated Learning using XGBoost, designed to operate with [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data.
Here we use the optimized gradient boosting library [XGBoost](https://github.com/dmlc/xgboost) and leverage its federated learning support.

Before starting please make sure you set up a [virtual environment](../../README.md#set-up-a-virtual-environment) and install the additional requirements:
```
python3 -m pip install -r requirements.txt
```

## Data
We conduct a classification task for fraud detection based on data from a bank and a telecommunication company, each with 31 features and 1 class label on the bank side. 
site-1 contains the bank data, while site-2 contains data from the telecommunication company.


## Vertical XGBoost Federated Learning with FLARE

This Vertical XGBoost example leverages the recently added [vertical federated learning support](https://github.com/dmlc/xgboost/issues/8424) in the XGBoost open-source library. This allows for the distributed XGBoost algorithm to operate in a federated manner on vertically split data.

For integrating with FLARE, we can use the predefined `XGBFedController` to run the federated server and control the workflow.

Next, we can use `FedXGBHistogramExecutor` and set XGBoost training parameters in `config_fed_client.json`, or define new training logic by overwriting the `xgb_train()` method.

Lastly, we must subclass `XGBDataLoader` and implement the `load_data()` method. For vertical federated learning, it is important when creating the `xgb.Dmatrix` to set `data_split_mode=1` for column mode, and to specify the presence of a label column `?format=csv&label_column=0` for the csv file. To support PSI, the dataloader can also read in the dataset based on the calculated intersection, and split the data into training and validation.

> **_NOTE:_** For secure mode, make sure to provide the required certificates for the federated communicator.

## Run the Example

The vertical xgboost job is already created.  Run the vertical xgboost job:
```
nvflare simulator ./jobs/vertical_xgb -w /tmp/nvflare/vertical_xgb -n 2 -t 2
```

The model will be saved to `test.model.json`.

(Feel free to modify the scripts and jobs as desired to change arguments such as number of clients, dataset sizes, training params, etc.)



## Results
Model accuracy can be visualized in tensorboard:
```
tensorboard --logdir /tmp/nvflare/vertical_xgb/server/simulate_job/tb_events
```


