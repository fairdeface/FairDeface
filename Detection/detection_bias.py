
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
from os import listdir, makedirs
from os.path import isdir, join
import argparse
import numpy as np
import matplotlib.transforms as transforms
from csv import writer


parser = argparse.ArgumentParser()  # Add an argument
parser.add_argument('--file', type=str, required=True,
                    help='path to the file')  # Parse the argument
parser.add_argument('--epsilons', type=str, required=False, default='.2, .15, .1, .05, .02',
                    help='enter epsilon values seperate with a comma')  # Parse the argument
args = parser.parse_args()
detection_path = args.file
task='detection'

epsilons=np.array(list(args.epsilons.replace(" ", "").split(",")), dtype=float)

f = open(detection_path, 'rb')
detection = pickle.load(f)
f.close()
plt.close('all')

line_styles=['-', '--', '-.', ':','dotted'  ]

def mean_detection_passing_calculator(rates, term):
    if term=='fail_ratio':
        fail_total=0
        sum_total=0
        means_detection = {}
        for dataset in rates.keys():
            summ_fails = 0
            summ_all = 0
            for demo in rates[dataset].keys():
                summ_fails += len(rates[dataset][demo]['fails'])
                fail_total+=len(rates[dataset][demo]['fails'])
                summ_all += rates[dataset][demo]['num_pics']
                sum_total+=rates[dataset][demo]['num_pics']
            means_detection[dataset] = round( 100*(1-(summ_fails/summ_all)), 2)
        means_total= round( 100*(1-(fail_total/sum_total)), 2)
        return means_total, means_detection
    elif term=='miss_ratio':
        means_passing = {}
        pass_total=0
        sum_total=0
        for dataset in rates.keys():
            summ_all = 0
            summ_pass=0
            for demo in rates[dataset].keys():
                summ_all += rates[dataset][demo]['num_orgs']
                sum_total += rates[dataset][demo]['num_orgs']
                summ_pass += rates[dataset][demo]['num_pics']
                pass_total +=rates[dataset][demo]['num_pics']
            means_passing[dataset] = round( 100*summ_pass/summ_all, 2)
        means_total= round( 100*pass_total/sum_total, 2)

        return means_total, means_passing

   

def sort_demos(total_demos, rate_type):
    new_demos = {}
    for demo in total_demos.keys():
        if type(total_demos[demo]) == dict:
            new_demos[demo] = total_demos[demo][rate_type]
    new_demos = sorted(new_demos.items(), key=lambda x: x[1],  reverse=False)
    new_demos = [d[0] for d in new_demos]
    return new_demos




    

def bias_measure(detection, obf_method, term):
        directory='bias'
        if not isdir(directory):
            makedirs(directory)
    
        csv_datasets=directory +"/" +  task + '-'+ obf_method+ '-bias.csv'
        with open(csv_datasets, 'w') as csv_f:
            writer_object = writer(csv_f)
            writer_object.writerow(['epsilons=' +str(epsilons) ])
            csv_f.close()
        csv_f=open(csv_datasets, 'a')
        writer_object = writer(csv_f)
        writer_object.writerow([obf_method])

        models = list(detection.keys())
   
        datasets = list(detection[models[0]].keys())
        first_row=[""]
        second_row=[""]
        row_datasets=[""]    
        row_datasets.append(['Dataset Bias'])
        row_datasets.append([""])
        for dataset in datasets:
            first_row.append(dataset)
            row_datasets[-1].append(dataset)
            demos = list(detection[models[0]][dataset].keys())
            demos = sort_demos(detection[models[0]][dataset], 'fail_ratio')
            # print(first_row)
            first_row.extend([""]*(len(demos)-1))



                    
        writer_object.writerow(first_row)
        # writer_object.writerow(second_row)
        bias_eps_demo={}
        bias_eps_dataset={}
     
        for model in detection.keys():
            
            row_datasets.append([model])
            row=[model]

            biases = {}
            bias_eps_demo[model]={}
            bias_eps_dataset[model]={}
            demo_rows=[""]
            # datasets = list(detection[model].keys())
            # num_cols= len(datasets)
            # num_rows= int(np.ceil(len(datasets)/2))
            for dataset in datasets:
                
                bias_eps_demo[model][dataset]={}
                bias_eps_dataset[model][dataset]={}
                for epsilon in epsilons:
                     bias_eps_dataset[model][dataset][epsilon]=0
                biases[dataset] = {}

                demos = list(detection[model][dataset].keys())
                demos = sort_demos(detection[model][dataset], term)
                demo_rows.extend(demos)
                num_demos = len(demos)
                dataset_biass = np.empty((num_demos, num_demos))
                Demos = []
                for D in demos:
                    if "_" in D:
                        D = ''.join([x[0].upper() for x in D.split('_')])
                    Demos.append(D)

                i = 0

                for demo in demos:
                    # print(detection[model][dataset][demo][term])
                    biases[dataset][demo] = {}
                    bias_eps_demo[model][dataset][demo]={}
                    for epsilon in epsilons:
                        bias_eps_demo[model][dataset][demo][epsilon]=0
                    for demo2 in demos:
                        
                        biases[dataset][demo][demo2] = (
                            detection[model][dataset][demo][term]-detection[model][dataset][demo2][term])/100
                        for epsilon in epsilons:
                            bias_eps_demo[model][dataset][demo][epsilon]+= (biases[dataset][demo][demo2] < -epsilon )
                            bias_eps_dataset[model][dataset][epsilon]+= (biases[dataset][demo][demo2] < -epsilon)

                            # biases[dataset][demo][demo2] = float("NaN")
                            # print('devided by zero', model, dataset, demo2)

                    bias=[]
                    for epsilon in epsilons:
                        bias_eps_demo[model][dataset][demo][epsilon]/=num_demos-1
                        # row.append(bias_eps_demo[model][dataset][demo][epsilon])
                        bias.append(int(np.round(100*bias_eps_demo[model][dataset][demo][epsilon])))
                    row.append(bias)     
                    bias_dataset=[]
                for epsilon in epsilons:
                    bias_eps_dataset[model][dataset][epsilon]/=(num_demos*(num_demos-1))/2  
                    bias_dataset.append(int(np.round(100*bias_eps_dataset[model][dataset][epsilon])))
           
                row_datasets[-1].append(bias_dataset)
            writer_object.writerow(demo_rows) 

            writer_object.writerow(row) 

        for row in row_datasets:
            writer_object.writerow(row)
               
        csv_f.close()

for obf_method in detection.keys():
    bias_measure(detection[obf_method], obf_method, 'fail_ratio')
