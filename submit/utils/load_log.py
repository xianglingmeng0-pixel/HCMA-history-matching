import os
import sys
import h5py
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

class log:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.experiment_dirs = [
            d for d in os.listdir(self.log_dir)
            if os.path.isdir(os.path.join(self.log_dir, d)) and d.startswith("experiment")
        ]

        self.experiment_num = len(self.experiment_dirs)

    def read_train_result(self, num):
        train_result_file = self.log_dir + '/' + self.experiment_dirs[num] + '/' + 'train_result.log'
        with open(train_result_file, 'r') as f:
            next(f)
            for line in f.readlines():
                r2 = float(line.split('\t')[0])
                rmse = float(line.split('\t')[1])
                time = float(line.split('\t')[2])
        return r2, rmse, time

    def read_val_metric_result(self, num):
        train_result_file = self.log_dir + '/' + self.experiment_dirs[num] + '/' + 'train_process.h5'
        with h5py.File(train_result_file, 'r') as f:
            val_r2 = f['val_r2'][:]
            val_rmse = f['val_rmse'][:]
        return val_r2, val_rmse
