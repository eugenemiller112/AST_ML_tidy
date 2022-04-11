from preprocessing import *
from postprocessing import *
from model import *
import csv
import time


def main():
    exp_list = ['20220325_KanTitration']

    res = dict([['20220325_KanTitration', 'H1']])

    sus = dict([['20220325_KanTitration', 'G1']])

    perfect_shuffle('/Volumes/External',
    '/Users/eugenemiller/Desktop/Modeling-3-25-22/0ug',
    exp_list,
    res,
    sus)


if __name__ == '__main__':
    main()
