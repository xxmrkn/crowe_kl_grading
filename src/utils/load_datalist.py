import argparse
import os
import pathlib

from utils.argparser import get_args

class LoadData(object):
    
    opt = get_args()
    train_files = []
    train_name = []
    valid_files = []
    valid_name = []
    test_files = []
    test_name = []
    
    @classmethod
    def load_train_data(cls) -> list[str]:

        p = pathlib.Path(
                f'{cls.opt.base_path}/datalist/datalist{cls.opt.datalist}/k{cls.opt.fold}'
            ).glob(f'train*.txt')
        
        for i in p:
            cls.train_files.append(os.path.join(f'k{cls.opt.fold}', i.name))

        for j in range(len(cls.train_files)):
            for i in range(int(cls.opt.fold)):
                if str(i) in cls.train_files[j]:
                    cls.train_name.append(os.path.join(cls.opt.datalist_path,
                                                       f'datalist{cls.opt.datalist}',
                                                       cls.train_files[j]))

        if len(cls.train_name)!=4:
            raise RuntimeError(f'Read fail. There are no text files.') 

        return cls.train_name

    @classmethod
    def load_valid_data(cls) -> list[str]:

        p = pathlib.Path(
                f'{cls.opt.base_path}/datalist/datalist{cls.opt.datalist}/k{cls.opt.fold}'
            ).glob(f'valid*.txt')

        for i in p:
            cls.valid_files.append(os.path.join(f'k{cls.opt.fold}', i.name))

        for j in range(len(cls.valid_files)):
            for i in range(int(cls.opt.fold)):
                if str(i) in cls.valid_files[j]:
                    cls.valid_name.append(os.path.join(cls.opt.datalist_path,
                                                       f'datalist{cls.opt.datalist}',
                                                       cls.valid_files[j]))

        if len(cls.valid_name)!=4:
            raise RuntimeError(f'Read fail. There are no text files.') 

        return cls.valid_name

    @classmethod
    def load_test_data(cls, datalist_num) -> list[str]:

        p = pathlib.Path(f'{datalist_num}/k{cls.opt.fold}').glob(f'test*.txt')
        #print(p)

        for i in p:
            cls.test_files.append(os.path.join(f'k{cls.opt.fold}', i.name))

        for j in range(len(cls.test_files)):
            for i in range(int(cls.opt.fold)):
                if str(i) in cls.test_files[j]:
                    cls.test_name.append(os.path.join(cls.opt.datalist_path,
                                                      f'datalist{datalist_num}',
                                                      cls.test_files[j]))
        #print('test', cls.test_name)

        if len(cls.test_name)!=4:
            raise RuntimeError(f'Read fail. There are no text files.') 

        return cls.test_name
