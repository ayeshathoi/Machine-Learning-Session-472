import argparse
from processor_1805062 import *

# use argparse to get the input file

parser = argparse.ArgumentParser()
parser.add_argument('dataset', metavar='dataset', type=int, help='the dataset number')
parser.add_argument('alpha', metavar='alpha', type=float, help='the alpha value')
parser.add_argument('epoch', metavar='epoch', type=int, help='the epoch value')
parser.add_argument('Feature_Count', metavar='Feature_Count', type=int, help='the Feature Count value')
args = parser.parse_args()


input_file_1 = './Datasets/Dataset1/WA_Fn-UseC_-Telco-Customer-Churn.csv'
input_file_dataset_2_train = './Datasets/Dataset2/adult.data'
input_file_dataset_2_test = './Datasets/Dataset2/adult.test'
input_file_dataset_3 = './Datasets/Dataset3/creditcard.csv'

# Take input which dataset you want to run
dataset_number = args.dataset
if dataset_number == 1:
    Processor.Telco_preprocessing(input_file_1, args.alpha, args.epoch, args.Feature_Count)
elif dataset_number == 2:
    Processor.adultPreprocessing(input_file_dataset_2_train, input_file_dataset_2_test, args.alpha, args.epoch, args.Feature_Count)
elif dataset_number == 3:
    Processor.creditCardPreprocessing(input_file_dataset_3, args.alpha, args.epoch, args.Feature_Count)
