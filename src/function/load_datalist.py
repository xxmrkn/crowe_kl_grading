import argparse
import os
import pathlib

from utils.parser import get_args

def load_train_data():

    opt = get_args()

    train_files = []
    train_name = []

    p = pathlib.Path(f'').glob(f'train*.txt')
    print(p)

    for i in p:
        train_files.append(os.path.join(f'k{opt.fold}', i.name))

    for j in range(len(train_files)):
        for i in range(int(opt.fold)):
            if str(i) in train_files[j]:
                train_name.append(os.path.join(opt.datalist_path,f'datalist{opt.datalist}', train_files[j]))
    print('train',train_name)

    if len(train_name)!=4:
        raise RuntimeError(f'read fail. there are no text files.') 

    return train_name

def load_valid_data():

    opt = get_args()

    valid_files = []
    valid_name = []

    p = pathlib.Path(f'').glob(f'valid*.txt')
    print(p)

    for i in p:
        valid_files.append(os.path.join(f'k{opt.fold}', i.name))

    for j in range(len(valid_files)):
        for i in range(int(opt.fold)):
            if str(i) in valid_files[j]:
                valid_name.append(os.path.join(opt.datalist_path,f'datalist{opt.datalist}', valid_files[j]))
    print('valid',valid_name)

    if len(valid_name)!=4:
        raise RuntimeError(f'read fail. there are no text files.') 

    return valid_name


def load_test_data():

    opt = get_args()

    test_files = []
    test_name = []

    p = pathlib.Path(f'').glob(f'test*.txt')
    print(p)

    for i in p:
        test_files.append(os.path.join(f'k{opt.fold}', i.name))

    for j in range(len(test_files)):
        for i in range(int(opt.fold)):
            if str(i) in test_files[j]:
                test_name.append(os.path.join(opt.datalist_path,f'datalist{opt.datalist}', test_files[j]))
    print('test',test_name)

    if len(test_name)!=4:
        raise RuntimeError(f'read fail. there are no text files.') 

    return test_name
