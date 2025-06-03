
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
identification_path = args.file


term_dict= {'all': 'Scenario1_All', 'each': 'Scenario2_Each'}

# scores_file = 'Identification_Scores_CIAGAN_000016.pkl'
# fig_style = 'all'

with open(identification_path, 'rb') as f:
    scores= pickle.load(f)
    f.close()

task='identification'

epsilons=np.array(list(args.epsilons.replace(" ", "").split(",")), dtype=float)

f = open(identification_path, 'rb')
identification = pickle.load(f)
f.close()
plt.close('all')

line_styles=['-', '--', '-.', ':','dotted'  ]

def mean_score_calculator(rates, term, method):
        sum_all=0
        sum_rates=0
        means_dataset={}
        for dataset in rates.keys():
            sum_dataset=0
            sum_rates_dataset=0
            for demo in rates[dataset].keys():
                if type(rates[dataset][demo])==dict:
                    
                    score=rates[dataset][demo][term]['test']
                    if method!='Original':
                        score=100-score
                        
                    sum_rates_dataset+=score
                    sum_dataset+=1
                    
                    sum_rates+=score
                    sum_all+=1

            means_dataset[dataset]= sum_rates_dataset/sum_dataset
        means_total=sum_rates/sum_all
        return means_total, means_dataset

def scores_filter(scores):
    methods = list(scores.keys())
    for method in methods:
        datasets= list(scores[method].keys())
        models= list(scores[method][datasets[0]].keys()) 
        for dataset in datasets:
            for model in models:
                if scores[method][dataset][model]['done']==0:
                    del scores[method][dataset][model]
    return scores


def sort_demos(total_demos, term):
    new_demos= {}
    for demo in total_demos.keys():
        if type(total_demos[demo])==dict:
            # print(total_demos)
            new_demos[demo]= total_demos[demo][term]['test']
    new_demos = sorted(new_demos.items(), key=lambda x:x[1],  reverse=False)
    new_demos = [d[0] for d in new_demos]

    return new_demos




    

def bias_measure(identification, obf_method, term):
        directory='bias'
        if not isdir(directory):
            makedirs(directory)
    
        csv_datasets=directory +"/" +  task + '-'+ obf_method+ '-'+ term + '-bias.csv'
        with open(csv_datasets, 'w') as csv_f:
            writer_object = writer(csv_f)
            writer_object.writerow(['epsilons=' +str(epsilons) ])
            csv_f.close()
        csv_f=open(csv_datasets, 'a')
        writer_object = writer(csv_f)
        writer_object.writerow([obf_method])

        models = list(identification.keys())
   
        datasets = list(identification[models[0]].keys())
        first_row=[""]
        second_row=[""]
        row_datasets=[""]    
        row_datasets.append(['Dataset Bias'])
        row_datasets.append([""])
        for dataset in datasets:
            first_row.append(dataset)
            row_datasets[-1].append(dataset)
            # print(identification[dataset])
            demos = list(identification[models[0]][dataset].keys())
            demos = sort_demos(identification[models[0]][dataset], term)
            for demo in demos:
                second_row.append(demo)
                first_row.append("")



                    
        writer_object.writerow(first_row)
        # writer_object.writerow(second_row)
        bias_eps_demo={}
        bias_eps_dataset={}
        for model in identification.keys():
            
            row_datasets.append([model])
            row=[model]

            biases = {}
            bias_eps_demo[model]={}
            bias_eps_dataset[model]={}

            # datasets = list(identification[model].keys())
            # num_cols= len(datasets)
            # num_rows= int(np.ceil(len(datasets)/2))
            sorted_demos=[""]

            for dataset in datasets:
                
                bias_eps_demo[model][dataset]={}
                bias_eps_dataset[model][dataset]={}
                for epsilon in epsilons:
                     bias_eps_dataset[model][dataset][epsilon]=0
                biases[dataset] = {}

                demos = list(identification[model][dataset].keys())
                demos = sort_demos(identification[model][dataset], term)
                sorted_demos+=demos
                num_demos = len(demos)
                dataset_biass = np.empty((num_demos, num_demos))
                Demos = []
                for D in demos:
                    if "_" in D:
                        D = ''.join([x[0].upper() for x in D.split('_')])
                    Demos.append(D)

                i = 0

                for demo in demos:
                    biases[dataset][demo] = {}
                    bias_eps_demo[model][dataset][demo]={}
                    for epsilon in epsilons:
                        bias_eps_demo[model][dataset][demo][epsilon]=0
                    for demo2 in demos:
                        try:

                            biases[dataset][demo][demo2] = (
                                -identification[model][dataset][demo][term]["test"]+identification[model][dataset][demo2][term]["test"])/100
                            for epsilon in epsilons:
                                bias_eps_demo[model][dataset][demo][epsilon]+= int(biases[dataset][demo][demo2] < -epsilon)
                                bias_eps_dataset[model][dataset][epsilon]+= int(biases[dataset][demo][demo2] < -epsilon)
                        except Exception as e: 
                            print(e)
                            
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
            writer_object.writerow(sorted_demos) 

            writer_object.writerow(row) 


        for row in row_datasets:
            writer_object.writerow(row)
               
        csv_f.close()
scores=scores_filter(scores)    
# print(scores)
for obf_method in scores.keys():
    for term in ['all', 'each']:    
        bias_measure(scores[obf_method], obf_method, term)
