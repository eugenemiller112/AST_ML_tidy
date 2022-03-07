from preprocessing import *
from postprocessing import *
from model import *
import csv
import time


def main():
    titration_pred(save_path='/Users/eugenemiller/Desktop/Lab/824Kan',
                   model_path='/Users/eugenemiller/Desktop/Lab/Data-0ug-25ug-Titration/model',
                   csv_path='/Users/eugenemiller/Desktop/Lab/Data-0ug-25ug-Titration/0ug25ugtrain_on_BW.csv')

    titration_pred(save_path='/Users/eugenemiller/Desktop/Lab/0112Kan_Data',
                   model_path='/Users/eugenemiller/Desktop/Lab/Data-0ug-25ug-Titration/model',
                   csv_path='/Users/eugenemiller/Desktop/Lab/Data-0ug-25ug-Titration/0ug25ugtrain_on_MNTH.csv')


if __name__ == '__main__':
    main()
