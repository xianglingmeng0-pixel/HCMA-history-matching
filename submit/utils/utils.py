import os
import shutil
import subprocess
import h5py
import numpy as np
import torch
from .metrics import R2

def check_data(grid_shape, split_size, sample_num, grid_num):
    assert grid_shape[0] * grid_shape[1] * grid_shape[2] == grid_num
    assert split_size[0] + split_size[1] + split_size[2] == sample_num

def init_experiment(seed_list, epoches, model_name, batch_size, experiment_dir):
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    with open(experiment_dir + 'experiment.log', 'w') as f:
        f.write('experiment time: ' + str(len(seed_list)) + '\n')
        f.write('random seed: ' + str(seed_list) + '\n')
        f.write('epoches: ' + str(epoches) + '\n')
        f.write('model name: ' + str(model_name) + '\n')
        f.write('batch size: ' + str(batch_size) + '\n')

def log_result(result_dir,
               best_model,
               optimizer,
               best_r2,
               best_rmse,
               train_loss_all,
               val_loss_all,
               train_r2_all,
               train_rmse_all,
               val_r2_all,
               val_rmse_all,
               time_use,
               best_r2_list,
               best_epoch):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # torch.save(best_model.state_dict(), result_dir+f'model.pth')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_state_dict': best_model.encoder.state_dict(),
        'n_feature': best_model.n_feature,
        'n_encoder': best_model.n_encoder,
    }, result_dir+f'model.pth')
    with open(result_dir+'train_result.log', 'w') as f:
        f.write(str('best_r2') + '\t' + str('best_rmse') + '\t' + str('time_use') + '\t' + str('best_epoch') + '\n')
        f.write(str(best_r2) + '\t' + str(best_rmse) + '\t' + str(time_use) + '\t' + str(best_epoch) + '\n')
    with h5py.File(result_dir + f'train_process.h5', 'w') as f:
        f.create_dataset('train_loss', data=train_loss_all)
        f.create_dataset('val_loss', data=val_loss_all)
        f.create_dataset('train_r2', data=train_r2_all)
        f.create_dataset('train_rmse', data=train_rmse_all)
        f.create_dataset('val_r2', data=val_r2_all)
        f.create_dataset('val_rmse', data=val_rmse_all)
        f.create_dataset('best_r2', data=best_r2_list)

def read_r2_rmse(path):
    with h5py.File(path, 'r') as f:
        val_r2 = f['val_r2'][:]
        val_rmse = f['val_rmse'][:]
        train_r2 = f['train_r2'][:]
        train_rmse = f['train_rmse'][:]
    return val_rmse, val_r2, train_rmse, train_r2


def load_experiment_results(base_path, experiment_name, num_folds=10):
    results = {
        'val_rmse': [],
        'val_r2': [],
        'train_rmse': [],
        'train_r2': []
    }

    for fold in range(1, num_folds + 1):
        file_path = f'{base_path}/{experiment_name}/fold_{fold}/result/train_process.h5'

        if os.path.exists(file_path):
            val_rmse, val_r2, train_rmse, train_r2 = read_r2_rmse(file_path)
            results['val_rmse'].append(val_rmse)
            results['val_r2'].append(val_r2)
            results['train_rmse'].append(train_rmse)
            results['train_r2'].append(train_r2)
        else:
            print(f"warning cannot find the file - {file_path}")

    for key in results:
        results[key] = np.stack(results[key], axis=0)

    return results


def calculate_final_metrics(results):
    final_metrics = {}

    for key, data in results.items():
        final_metrics[f'{key}_best'] = np.mean(data, axis=0)[-1]

    return final_metrics


def load_multiple_experiments(base_path, experiment_configs):
    all_experiments = {}

    for exp_name, num_folds in experiment_configs:
        print(f"load experiment: {exp_name}")
        results = load_experiment_results(base_path, exp_name, num_folds)
        final_metrics = calculate_final_metrics(results)

        all_experiments[exp_name] = {
            'raw_results': results,
            'final_metrics': final_metrics
        }

    return all_experiments

def load_experiment_time(base_path, experiment_name, num_folds=10):
    times_ = []
    for fold in range(1, num_folds + 1):

        file_path = f'{base_path}/{experiment_name}/fold_{fold}/result/train_result.log'

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                times_.append(float(lines[1].strip().split('\t')[2]))
        else:
            print(f"warning cannot find the file  - {file_path}")
    return np.mean(np.array(times_))

def load_experiment_time_not_fold(base_path, experiment_name):
    file_path = f'{base_path}/{experiment_name}/result/train_result.log'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            times= float(lines[1].strip().split('\t')[2])

            return times
    else:
        print(f"warning cannot find the file  - {file_path}")
        exit()


def predict_with_surrogate(_model, _x, _y, _train_dataset, grid_shape, if_seq, _ts_feature):
    _x -= _train_dataset.min_x
    _x /= (_train_dataset.max_x - _train_dataset.min_x)
    _x[_x < 0] = 0
    _x[_x > 1] = 1
    _x = np.nan_to_num(_x)
    _x = _x.reshape(-1, grid_shape[2], grid_shape[1], grid_shape[0])
    _x = np.transpose(_x, (0, 3, 2, 1))
    if if_seq:
        _x = _x.reshape(-1, grid_shape[0], grid_shape[1] * grid_shape[2])
    _x = torch.FloatTensor(_x).cuda()
    _predict_ = _model(_x)
    _y = _y.reshape(_y.shape[0], _ts_feature[1], _ts_feature[0])
    _y = _y.swapaxes(1, 2)
    _y = (_y - _train_dataset.min_y) / (_train_dataset.max_y  - _train_dataset.min_y)
    _y[_y < 0] = 0
    _y[_y > 1] = 1
    _y = np.nan_to_num(_y)
    r2_all = []
    for i in range(_predict_.shape[0]):
        r2 = R2()(_predict_.cpu()[i], torch.from_numpy(_y)[i])
        r2_all.append(float(r2))
    _predict = _predict_.cpu().detach().numpy()
    _predict[_predict < 0] = 0
    _predict[_predict > 1] = 1
    _PR = _predict * (_train_dataset.max_y - _train_dataset.min_y) + _train_dataset.min_y
    _PR = _PR.swapaxes(1, 2)
    _PR = _PR.reshape(_PR.shape[0], -1)
    return _PR, r2_all