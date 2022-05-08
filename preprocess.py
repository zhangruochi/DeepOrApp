import os 
import numpy as np
import platform
import pickle
import json
import torch
import xlrd


class Discretizer:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), 'discretizer_config.json')):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(
                zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [
            ["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :]))
                            for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(
            100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(
            100.0 * self._empty_bins_sum / self._done_count))



class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_x -
                                 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret



def load_normalizer(X):

    discretizer = Discretizer(timestep=1.0,
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

    discretizer_header = discretizer.transform(X)[1].split(',')

    # print(len(discretizer_header))
    # exit()

    cont_channels = [i for (i, x) in enumerate(
        discretizer_header) if x.find("->") == -1]
     
     # choose here which columns to standardize
    normalizer = Normalizer(fields=cont_channels)
          # Read data
    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(
        1.0, "previous")
    normalizer_state = os.path.join(
        os.path.dirname(__file__), normalizer_state)

    normalizer.load_params(normalizer_state)

    return normalizer, discretizer


def read_ts_excel_file(ts_filename):
    data = xlrd.open_workbook(ts_filename)
    table = data.sheets()[0]
    ret = []

    start_row = 17

    assert table.cell_value(start_row, 0) == "Hours"
    header = []
    
    for i in range(start_row, table.nrows):
        tmp_row = []
        for j in range(table.ncols):
            if i == start_row:
                header.append(table.cell_value(i, j))
            else:
                v = table.cell_value(i, j)
                try:
                    if int(v) == float(v):
                        v = int(v)
                except:
                    pass
                tmp_row.append(v)

        if i != start_row:
            ret.append(np.array(tmp_row))

    # print(header)
    return (np.stack(ret), header)
            
            
def read_ts_file(ts_filename):
    ret = []
    with open(ts_filename, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    # print(np.stack(ret).shape)
    return (np.stack(ret), header)


def read_demo_excel_file(demo_file):
    data = xlrd.open_workbook(demo_file)
    table = data.sheets()[0]

    start_row = 14
    assert table.cell_value(start_row, 0) == "Ethnicity"

    cur_data = [""] + [table.cell_value(start_row+1, j) for j in range(5)]


    try:
        if cur_data[3] == '':
            cur_data[3] = 60.0
        if cur_data[4] == '':
            cur_data[4] = 160
        if cur_data[5] == '':
            cur_data[5] = 60

        cur_demo = np.zeros(12)
        cur_demo[int(cur_data[1])] = 1
        cur_demo[5 + int(cur_data[2])] = 1
        cur_demo[9:] = cur_data[3:6]
        
        mean_avg_dict = {
            9: [63.83317389452685,17.38727125085254],
            10: [161.9350158239143,8.401452745987278],
            11: [77.87731194329766,46.516482138433176]
        }

        for each_idx in range(9, 12):
            _mean = mean_avg_dict[each_idx][0]
            _std = mean_avg_dict[each_idx][1]
            cur_demo[each_idx] = (
                cur_demo[each_idx] - _mean) / _std
    except:
        cur_demo = np.zeros(12)

    return cur_demo, cur_data
    
    
def read_demo_file(demo_file):
    with open(demo_file, "r") as demo_file:
        header = demo_file.readline().strip().split(',')
        assert header[0] == "Icustay"
            
        cur_data = demo_file.readline().strip().split(',')

        if len(cur_data) == 1:
            cur_demo = np.zeros(12)
        else:
            if cur_data[3] == '':
                cur_data[3] = 60.0
            if cur_data[4] == '':
                cur_data[4] = 160
            if cur_data[5] == '':
                cur_data[5] = 60

            cur_demo = np.zeros(12)
            cur_demo[int(cur_data[1])] = 1
            cur_demo[5 + int(cur_data[2])] = 1
            cur_demo[9:] = cur_data[3:6]

    mean_avg_dict = {
        9: [63.83317389452685,17.38727125085254],
        10: [161.9350158239143,8.401452745987278],
        11: [77.87731194329766,46.516482138433176]
    }

    for each_idx in range(9, 12):
        _mean = mean_avg_dict[each_idx][0]
        _std = mean_avg_dict[each_idx][1]
        cur_demo[each_idx] = (
            cur_demo[each_idx] - _mean) / _std

    return cur_demo, cur_data

    

def get_example(ts_filename, demo_filename):

    if os.path.basename(ts_filename).split(".")[-1] == "csv":
        ts_X, header = read_ts_file(ts_filename)
    else:
        ts_X, header = read_ts_excel_file(ts_filename)
    normalizer, discretizer = load_normalizer(ts_X)
    ts_X = discretizer.transform(ts_X, end=48.0)[0]
    ts_data = normalizer.transform(ts_X)

    ts_tensor = torch.tensor(ts_data,
                 dtype=torch.float).unsqueeze(0)

    assert ts_tensor.shape == (1, 48, 76)

    if os.path.basename(demo_filename).split(".")[-1] == "csv":
        demo_X, demo_data = read_demo_file(demo_filename)
    else:
        demo_X, demo_data = read_demo_excel_file(demo_filename)

    demo_tensor = torch.tensor(demo_X,
                             dtype=torch.float).unsqueeze(0)
    assert demo_tensor.shape == (1, 12)

    return ts_tensor, demo_tensor, demo_data


if __name__ == "__main__":
    ts_tensor, demo_tensor, demo_data = get_example("./icu_patient_1.xlsx", "./icu_patient_1.xlsx")
    
    