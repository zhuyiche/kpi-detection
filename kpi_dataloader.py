import numpy as np
from preprocessing import standardize_kpi, complete_timestamp
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from config import Configuration as cfg

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('datasets'):
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

DATA_PATH = os.path.join(ROOT_DIR, 'data')
from torch.utils.data import Dataset


class SingleWindowUnsupervisedKPIDataLoader(Dataset):
    def __init__(self, datatype, train_datapath, test_datapath, window_size, window_gap):
        self.window_size = window_size
        self.window_gap = window_gap
        self.datatype = datatype
        assert self.datatype in ['train', 'test', 'valid']

        if datatype == 'test':
            self.datapath = test_datapath
        else:
            self.datapath = train_datapath

        self.upper_file = None


        if 'csv' in self.datapath:
            self.upper_file = os.path.dirname(self.datapath)
            _data = self.load_kpi_csv(self.datapath)
            if self.datatype == 'train' or self.datatype == 'valid':
                self.timestamp, self.values, self.id, self.labels = self.retrieve_info_from_csv(_data, datatype)
            else:
                self.timestamp, self.values, self.id, = self.retrieve_info_from_csv(_data, datatype)

            if (cfg.mode == 'unsupervised' and self.datatype == 'test') or cfg.have_label is False:
                self.labels = np.zeros_like(self.values, dtype=np.int32)
            self._assert_unique_timestamp(self.timestamp)

            # preprocssing
            self.values, mean, std = standardize_kpi(values=self.values)


            self.total_train_datasets, self.total_train_event_labels = [], []
            self.total_valid_datasets, self.total_valid_event_labels = [], []
            self.total_test_datasets, self.total_test_event_labels = [], []

            if self.datatype == 'train' or self.datatype == 'valid':
                train_datasets, train_event_labels, valid_datasets, valid_event_labels = self.process_data(
                    train_datasets_list=self.total_train_datasets,
                    train_event_labels_list=self.total_train_event_labels,
                    valid_datasets_list=self.total_valid_datasets,
                    valid_event_labels_list=self.total_valid_event_labels)
            else:
                test_datasets, test_event_labels = self.process_data(test_datasets_list=self.total_test_datasets,
                                                                     test_event_labels_list=self.total_test_event_labels)

        else:
            print('concat multiple csv files')

            self.total_train_datasets, self.total_train_event_labels = [], []
            self.total_valid_datasets, self.total_valid_event_labels = [], []
            self.total_test_datasets, self.total_test_event_labels = [], []
            for index, data_file in enumerate(os.listdir(self.datapath)):
                self.data_file = data_file
                _single_data = self.load_kpi_csv(os.path.join(self.datapath, data_file))
                ### for multiple files
                if self.datatype == 'train' or self.datatype == 'valid':
                    self.timestamp, self.values, self.id, self.labels = self.retrieve_info_from_csv(_single_data, datatype)
                else:
                    self.timestamp, self.values, self.id, = self.retrieve_info_from_csv(_single_data, datatype)

                # preprocssing
                self.values, mean, std = standardize_kpi(values=self.values)


                if (cfg.mode == 'unsupervised' and self.datatype == 'test') or cfg.have_label is False:
                    self.labels = np.zeros_like(self.values, dtype=np.int32)
                self._assert_unique_timestamp(self.timestamp)

                if self.datatype == 'train' or self.datatype == 'valid':
                    self.total_train_datasets, self.total_train_event_labels, \
                    self.total_valid_datasets, self.total_valid_event_labels = self.process_data(
                        train_datasets_list=self.total_train_datasets, train_event_labels_list=self.total_train_event_labels,
                        valid_datasets_list=self.total_valid_datasets, valid_event_labels_list=self.total_valid_event_labels)


                else:
                    self.total_test_datasets, self.total_test_event_labels = self.process_data(
                        test_datasets_list=self.total_test_datasets,
                        test_event_labels_list=self.total_test_event_labels)

            self.total_train_datasets = np.array(self.total_train_datasets)
            self.total_valid_datasets = np.array(self.total_valid_datasets)
            self.total_train_event_labels = np.array(self.total_train_event_labels)
            self.total_valid_event_labels = np.array(self.total_valid_event_labels)
            self.total_train_event_labels = self.total_train_event_labels.reshape(
                self.total_train_event_labels.shape[0], 1)
            self.total_valid_event_labels = self.total_valid_event_labels.reshape(
                self.total_valid_event_labels.shape[0], 1)
            self.total_test_datasets = np.array(self.total_test_datasets)
            self.total_test_event_labels = np.array(self.total_test_event_labels)
            self.total_test_event_labels = self.total_test_event_labels.reshape(self.total_test_event_labels.shape[0],
                                                                                1)

            print('train_size: {}, train_event_labels_size: {}'.format(len(self.total_train_datasets),
                                                                       len(self.total_train_event_labels)))
            print('valid_size: {}, valid_event_labels_size: {}'.format(len(self.total_valid_datasets),
                                                                       len(self.total_valid_event_labels)))
            print('test_size: {}, test_event_labels_size: {}'.format(len(self.total_test_datasets),
                                                                       len(self.total_test_event_labels)))

                #total_data_files.append(_single_data)
            #_data = pd.concat(total_data_files
        #timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))




    def __len__(self):
        if self.datatype == 'train':
            return self.total_train_datasets.shape[0]
        elif self.datatype == 'valid':
            return self.total_valid_datasets.shape[0]
        else:
            return self.total_test_datasets.shape[0]


    def __getitem__(self, index):
        if self.datatype == 'train':
            #train_datasets, train_event_labels, valid_datasets, valid_event_labels = self.process_data()
            #print("train shape !!!!!!!!!!!!!! ", np.array(self.total_train_datasets).shape)
            #print("train label shape !!!!!!!!!!!!!! ", np.array(self.total_train_event_labels).shape)
            #print(self.total_train_datasets)
            #print(self.total_train_event_labels)
            total_datasets, total_event_labels = self.total_train_datasets, self.total_train_event_labels
        elif self.datatype == 'valid':
            total_datasets, total_event_labels = self.total_valid_datasets, self.total_valid_event_labels
        else:
            #test_datasets, test_event_labels = self.process_data()
            total_datasets, total_event_labels = self.total_test_datasets, self.total_test_event_labels
        return total_datasets[index, ], total_event_labels[index,]

    def process_data(self, train_datasets_list=None, train_event_labels_list=None,
                     valid_datasets_list=None, valid_event_labels_list=None,
                     test_datasets_list=None, test_event_labels_list=None):
        #values = np.asarray(self.values, dtype=np.float32)
        #labels = np.asarray(self.labels, dtype=np.int32)
        values = self.values
        labels = self.labels
        if len(values.shape) != 1:
            raise ValueError('Value must be a 1-D array')
        if labels.shape != values.shape:
            raise ValueError('labels shape should be save as values shape')

        if self.datatype == 'train' or self.datatype == 'valid':
            n = int(len(values) * cfg.val_frac)
            train_values, valid_values = values[:-n], values[-n:]
            train_labels, valid_labels = labels[:-n], labels[-n:]
            assert len(train_values) == len(train_labels)
            assert len(valid_labels) == len(valid_values)
            train_size, valid_size = len(train_values), len(valid_values)
            total_train_step, total_valid_step = int(train_size // self.window_size) + 1, \
                                                 int(valid_size // self.window_size) + 1

            train_datasets, train_event_labels = self._window_data_process(total_step=total_train_step,
                                                                         values=train_values,
                                                                         labels=train_labels,
                                                                        datasets=train_datasets_list,
                                                                           event_labels=train_event_labels_list)
            valid_datasets, valid_event_labels = self._window_data_process(total_step=total_valid_step,
                                                                           values=valid_values,
                                                                           labels=valid_labels,
                                                                           datasets=valid_datasets_list,
                                                                           event_labels=valid_event_labels_list)
            return train_datasets, train_event_labels, valid_datasets, valid_event_labels
        else:   ## else would be test datasets.
            test_size = len(values)
            total_test_step = int(test_size // self.window_size) + 1
            test_datasets, test_event_labels = self._window_data_process(total_step=total_test_step,
                                                                         values=values,
                                                                         labels=labels,
                                                                         datasets=test_datasets_list,
                                                                         event_labels=test_event_labels_list
                                                                         )
            return test_datasets, test_event_labels

    def _assert_unique_timestamp(self, df):
        """
        Check if time stamp is unique, otherwise raise valueError.
        """
        df_size = df.shape[0]
        unique_df_size = np.unique(df).shape[0]
        if df_size != unique_df_size:
            raise ValueError('{} file has timestamp that is not unique'.format(self.data_file))

    def _window_data_process(self, total_step, values, datasets, event_labels, labels=None):
        """
        constrcut data set into window_size and window_gap
        Args:
            return: datasets (List) and event_labels(List):
                The internal state of dataset list is
                [[window size long dataset]_{0}, [window size long dataset]_{window-gap},...]
            TODO the event_labels may be further modified because our ultimate goal is to detect \
            point anomlay
        """
        start_point = 0
        for _ in range(total_step):
            end_point = start_point + self.window_size

            if labels is not None:
                step_label = labels[start_point: end_point]

                #### This step make point labels(each point in kpi contains a label, either 0 or 1),
                #### To a event labels(for each window size long kpi data, label 0 or 1 to indicate if
                #### this size long kpi data is anomaly or not.)
                step_event_label = 0
                for index, part_label in enumerate(step_label):
                    if part_label == 1:
                        step_event_label = 1
                        break

                event_labels.append(step_event_label)

            step_value = values[start_point: end_point]
            datasets.append(step_value)
            start_point += self.window_gap

        if labels is not None:
            return datasets, event_labels
        else:
            return datasets

    def load_kpi_csv(self, csv_path):
        """
        Load csv file, check if multiple kpi are contains in csv.
        If more than one kpi, separate and save csv file for each kpi id.
        :param dir: directory name contains csv file.
        :param csv: csv file name.
        :param id_name: Name of row for kpi id, default is the public data.
        :return:
        Currently this function will be terminate and raise a value error if
        more than one kpi are used in a file after csv files separated.
        TODO make this process viable
        """
        def save_csv_id(df, _id, save_csv_path):
            """
            Save kpi according to its id.
            Args:
                csv_type (str): either train or test
                save_csv_path (str): path to save csv file.
            """
            for id in _id:
                df_id = df.loc[df[cfg.kpi_id].values == id]
                csv_file_name = os.path.join(save_csv_path, id + '_' + '.csv')
                df_id.to_csv(csv_file_name)

        kpi_data = pd.read_csv(csv_path)
        unique_id = np.unique(kpi_data[cfg.kpi_id].values)
        if len(unique_id) != 1:
            if self.upper_file is not None:
                if not os.path.exists(self.upper_file):
                    os.makedirs(self.upper_file)
            else:
                if not os.path.exists(self.datapath):
                    os.makedirs(self.datapath)

            print('{} KPI in file {}'.format(len(unique_id), csv_path))
            print('Will be save to csv file for each KPI ID')
            if self.upper_file is not None:
                save_csv_id(kpi_data, unique_id, self.upper_file)
            else:
                save_csv_id(kpi_data, unique_id, self.datapath)
            raise ValueError('Due to the mulitple kpi id, run this program again')
        else:
            return kpi_data

    def retrieve_info_from_csv(self, df, stage):
        """
        This class is design specific for kpi data from iops.ai
        :param df: train or test data
        :param stage: str: train or test
        """
        assert stage in ['train', 'test', 'valid']
        if stage == 'train' or stage == 'valid':
            return df[cfg.timestamp].values, df[cfg.value].values, df[cfg.kpi_id].values, df[cfg.label].values
        else:
            return df[cfg.timestamp].values, df[cfg.value].values, df[cfg.kpi_id].values


def merge_test_label():
    import h5py
    hdf = '/Users/yichen/Desktop/program/kpi-detection/data/test/truth.hdf'
    hfile = h5py.File(hdf, 'r')
    new_csv = {}
    for key in hfile.keys():
        #print('Keys: ',key)
        test_group = hfile[key]
        for group_key in test_group.keys():
            #if group_key == 'axis0':

            print(group_key)
            print(test_group[group_key].value)
            #print(str(test_group[group_key]))
            new_csv[str(test_group[group_key])] = test_group[group_key].value

    #pd.DataFrame.from_dict(new_csv)
    #print(new_csv)
    #.to_csv('/Users/yichen/Desktop/program/kpi-detection/data/test/truth.csv')
    hfile.close()


def extract_unique_id(df):
    """
    This method is fixed, could change to other names.
    """
    unique_id = df['KPI ID'].values
    print(unique_id)

if __name__ == '__main__':
    merge_test_label()





