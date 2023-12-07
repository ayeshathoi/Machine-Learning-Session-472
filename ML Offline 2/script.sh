#!/bin/bash

# DATASET_num Alpha Epoch FEATURE_COUNT
python3 main_1805062.py 1 100 1000 5 >> dataset1_report.txt
python3 main_1805062.py 2 100 1000 15 >> dataset2_report.txt
python3 main_1805062.py 3 100 1000 15 >> dataset3_report.txt
