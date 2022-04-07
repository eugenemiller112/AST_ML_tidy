from preprocessing import *
from postprocessing import *
from model import *
import csv
import time


def main():
    jitter_settings = {'lag': 3,
                       'crop': 100,
                       'upsample': 100}

    seg_settings = {'crop': 200,
                    'min_sigma': 10,
                    'max_sigma': 50,
                    'num_sigma': 50,
                    'threshold': .000001,
                    'overlap': 0,
                    'radius': 5,
                    'min_size': 30,
                    'block_size': 3}

    process('/Volumes/External/20220325_KanTitration', 1, 0, seg_settings=seg_settings,
            jitter_settings=jitter_settings, test_jitter=True, test_seg=True)


if __name__ == '__main__':
    main()
