from os.path import join
from pprint import pprint
from os import listdir


def eval_generated(directory):
    from torch_fidelity import calculate_metrics
    files_real = join(directory, 'real')
    files_pred = join(directory, 'predicated')
    num_files = len(listdir(files_pred))
    metrics = calculate_metrics(input2=files_real,
                                input1=files_pred,
                                fid=True,
                                kid=True,
                                kid_subset_size=num_files,
                                cuda=True)
    print('All our metrics')
    pprint(metrics)
