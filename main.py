from preprocessing import *
from postprocessing import *
from model import *
import csv
import time


def main():
    model, test_dat, test_lab, history = generate_RNN(
        test_dir='/Users/eugenemiller/Desktop/Lab/Data-0ug-25ug-Titration/test',
        val_dir='/Users/eugenemiller/Desktop/Lab/Data-0ug-25ug-Titration/valid',
        train_dir='/Users/eugenemiller/Desktop/Lab/Data-0ug-25ug-Titration/train'
    )

if __name__ == '__main__':
    main()
