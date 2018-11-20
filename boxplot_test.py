import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import sklearn

MAIN_PATH = os.getcwd()
datapath = os.path.join(MAIN_PATH, 'data')
train_path = os.path.join(datapath, 'train')


def boxplot(kpi_values, boxplot_size=None):
    """
    Args:
        kpi_values (int):
            values of kpis.
        boxplot_size (int):
            size of window to calculate boxplot value.
    """

    def _boxplot_algo(kpi):
        upper_quantile = np.percentile(kpi, 75)
        lower_quantile = np.percentile(kpi, 25)
        iqr = upper_quantile - lower_quantile
        upper_outlier_bound = upper_quantile + 1.5 * iqr
        lower_outlier_bound = lower_quantile - 1.5 * iqr
        kpi[kpi > upper_outlier_bound] = 1
        kpi[kpi < lower_outlier_bound] = 1
        kpi[kpi != 1] = 0
        kpi_boxplot_labels = kpi
        return kpi_boxplot_labels

    if boxplot_size is not None:
        if boxplot_size > len(kpi_values):
            raise ValueError('boxplot_size cannot larger than kpi_values size')

    kpi_size = len(kpi_values)
    if kpi_size == boxplot_size or boxplot_size is None:
        kpi_labels = _boxplot_algo(kpi_values)
    else:
        num_step = int(kpi_size // boxplot_size) + 1
        start_idx = 0
        for step in range(num_step):
            end_idx = start_idx + boxplot_size
            current_kpi = kpi_values[start_idx: end_idx]
            current_kpi_labels = _boxplot_algo(current_kpi)
            if step == 0:
                kpi_labels = current_kpi_labels
            else:
                kpi_labels.append(current_kpi_labels)
            start_idx += boxplot_size
    return kpi_labels


if __name__ == '__main__':
    example_file_path = os.path.join(train_path, '1c35dbf57f55f5e4_.csv')
    example_file = pd.read_csv(example_file_path, index_col=0)
    values = example_file['value']
    result = boxplot(values)
    print(result)
    print(type(result))
    result.to_csv(os.path.join(MAIN_PATH, 'example_boxplot.csv'))