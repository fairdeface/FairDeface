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

from shutil import rmtree
import pickle
import numpy as np
from PIL import Image
from os.path import join, isdir, isfile, basename
from os import listdir, remove, walk, mkdir, makedirs
import os
import json, pickle
import requests
from random import shuffle, choice
import argparse
parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('--obfuscation_file', type=str, required=True, help='The path to the obfuscated attribute file')# Parse the argument
parser.add_argument('--epsilons', type=str, required=False, default='.2, .15, .1, .05, .02',
                    help='enter epsilon values seperate with a comma')  # Parse the argument
parser.add_argument('--original_file', type=str, required=False, help='The path to the original attribute file')# Parse the argument

args = parser.parse_args()

# fig_style = args.fig_style

attributes_all_file=args.obfuscation_file
with open(attributes_all_file, 'rb') as my_file:
    attributes_all = pickle.load(my_file)
    my_file.close()
method = list(attributes_all.keys())[0]
if method!='CIAGAN':
    attributes_all_ORG=args.original_file
    with open(attributes_all_ORG, 'rb') as my_file:
        attributes_all_org = pickle.load(my_file)
        my_file.close()

line_styles=['-', '--', '-.', ':','dotted'  ]

def mean_rate_calculator(rates, attribute_comparison, attribute):
        sum_all=0
        sum_rates=0
        means_dataset={}
        for dataset in rates.keys():
            sum_dataset=0
            sum_rates_dataset=0
            for demo in rates[dataset].keys():
                if type(rates[dataset][demo])==dict:
                    num_pairs= len(attribute_comparison[dataset][demo][attribute])
                    score=rates[dataset][demo]['mean']*num_pairs

                        
                    sum_rates_dataset+=score
                    sum_dataset+=num_pairs
                    
                    sum_rates+=score
                    sum_all+=num_pairs

            means_dataset[dataset]= sum_rates_dataset/sum_dataset
        means_total=sum_rates/sum_all
        return means_total, means_dataset        


attribute_results = {}


attributes=['gender', 'emotion', 'race', 'age']

    

attribute_comparison = {}

org_pics ={}
obf_pics = {}
org_obf = {}
method = list(attributes_all.keys())[0]
for dataset in attributes_all[method].keys():
    attribute_results[dataset]= {}
    org_pics[dataset] ={}
    obf_pics[dataset] = {}
    org_obf[dataset] = {}
    
    attribute_comparison[dataset] = {}
    for demo in attributes_all[method][dataset].keys():
        attribute_results[dataset][demo]= {}
        org_pics[dataset][demo] = []
        obf_pics[dataset][demo] = []
        org_obf[dataset][demo] = {}
        attribute_comparison[dataset][demo] = {}
        
        for attribute in attributes:
            attribute_comparison[dataset][demo][attribute] = []
        
        
        for person in attributes_all[method][dataset][demo].keys():
            
            if type(attributes_all[method][dataset][demo][person]) == dict:
                obfs= list(attributes_all[method][dataset][demo][person]['attribute'].keys())
                if method =='CIAGAN':
                    orgs=[org for org in obfs if 'org' in org]
                    obfs=[obf for obf in obfs if 'obf' in obf]
                    
                    pics=[obf for obf in obfs if obf.replace('obf', 'org') in orgs]
                    
                else:
                    orgs= list(attributes_all_org['Original'][dataset][demo][person]['attribute'].keys())
                    pics=[org for org in orgs if org in obfs]
                        
                

                for pic in pics:

                    org_obf[dataset][demo][pic] = {}
                    org_obf[dataset][demo][pic]['obf'] = attributes_all[method][dataset][demo][person]['attribute'][pic]

                    if method!='CIAGAN':
                        org_obf[dataset][demo][pic]['org'] = attributes_all_org['Original'][dataset][demo][person]['attribute'][pic]
                    else:
                        org_obf[dataset][demo][pic]['org'] = attributes_all[method][dataset][demo][person]['attribute'][pic.replace('obf', 'org')]

                    for attribute in attributes:
                        if attribute == 'age':
                            difference = abs(org_obf[dataset][demo][pic]['org'][attribute]- org_obf[dataset][demo][pic]['obf'][attribute])
                            similarity= 1- difference/abs(org_obf[dataset][demo][pic]['org'][attribute])
                            
                            attribute_comparison[dataset][demo][attribute].append(similarity)
                        else:
                            if org_obf[dataset][demo][pic]['org'][attribute]==org_obf[dataset][demo][pic]['obf'][attribute]:
                                attribute_comparison[dataset][demo][attribute].append(1)
                            else:
                                attribute_comparison[dataset][demo][attribute].append(0)
                                
                # for obf in attributes_all[method][dataset][demo][person]['attribute'].keys():
                #     obf_pics[dataset][demo].append(obf)
        # print("********************************************")
        # print(dataset, demo )
        for attribute in attributes:
            attribute_results[dataset][demo][attribute] = {}
            if attribute == 'age':
              
                mean_age = np.mean(attribute_comparison[dataset][demo][attribute])
                std_age = np.std(attribute_comparison[dataset][demo][attribute])/np.sqrt(len(attribute_comparison[dataset][demo][attribute]))
                
                attribute_results[dataset][demo][attribute]['mean'] = round(100*mean_age, 2)
                attribute_results[dataset][demo][attribute]['std_err'] = round(100*std_age, 2)

                # print("\nage:")
                # print('mean: ', round(mean_age, 1), "std: ", round(std_age, 2))
            else:
                # y = 'mean_' + attribute
                # exec('%s = %d' %(y, np.mean(attribute_comparison[dataset][demo][attribute])))
                # print('\n', attribute)
                # print( "mean: ", round(100*np.mean(attribute_comparison[dataset][demo][attribute]), 1))
                attribute_results[dataset][demo][attribute]['mean'] = round(100*np.mean(attribute_comparison[dataset][demo][attribute]), 1)

attribute_dict={}
for att in attributes:
    attribute_dict[att]={}
    for dataset in attribute_results.keys():
        attribute_dict[att][dataset]={}
        for demo in attribute_results[dataset].keys():
            attribute_dict[att][dataset][demo] = attribute_results[dataset][demo][att]


# parser = argparse.ArgumentParser()  # Add an argument
# parser.add_argument('--file', type=str, required=True,
#                     help='path to the file')  # Parse the argument
# parser.add_argument('--epsilons', type=str, required=False, default='.2, .15, .1, .05, .02',
#                     help='enter epsilon values seperate with a comma')  # Parse the argument
# args = parser.parse_args()
# identification_path = args.file




epsilons=np.array(list(args.epsilons.replace(" ", "").split(",")), dtype=float)




def sort_demos(total_demos):
    new_demos= {}
    for demo in total_demos.keys():
        if type(total_demos[demo])==dict:
            # print( total_demos[demo])
            # print(total_demos)
            new_demos[demo]= total_demos[demo]['mean']
    new_demos = sorted(new_demos.items(), key=lambda x:x[1],  reverse=False)
    new_demos = [d[0] for d in new_demos]

    return new_demos



from os.path import isdir
from csv import writer

from os import makedirs

def bias_measure(attribute_dict, obf_method, attribute):
        directory='bias'
        if not isdir(directory):
            makedirs(directory)
    
        csv_datasets=directory +"/" +  obf_method+ '-'+attribute+ '-bias.csv'
        with open(csv_datasets, 'w') as csv_f:
            writer_object = writer(csv_f)
            writer_object.writerow(['epsilons=' +str(epsilons) ])
            csv_f.close()
        csv_f=open(csv_datasets, 'a')
        writer_object = writer(csv_f)
        writer_object.writerow([obf_method])

        # attributes = list(attribute_dict.keys())
   
        datasets = list(attribute_dict[attribute].keys())
        first_row=[""]
        second_row=[""]
        row_datasets=[""]    
        row_datasets.append(['Dataset Bias'])
        row_datasets.append([""])
        for dataset in datasets:
            first_row.append(dataset)
            row_datasets[-1].append(dataset)
            # print(identification[dataset])
            demos = list(attribute_dict[attribute][dataset].keys())
            demos = sort_demos(attribute_dict[attribute][dataset])
            for demo in demos:
                second_row.append(demo)
                first_row.append("")



                    
        writer_object.writerow(first_row)
        # writer_object.writerow(second_row)
        bias_eps_demo={}
        bias_eps_dataset={}
        for attribute in attribute_dict.keys():
            
            row_datasets.append([attribute])
            row=[attribute]

            biases = {}
            bias_eps_demo[attribute]={}
            bias_eps_dataset[attribute]={}

            # datasets = list(identification[model].keys())
            # num_cols= len(datasets)
            # num_rows= int(np.ceil(len(datasets)/2))
            sorted_demos=[""]

            for dataset in datasets:
                
                bias_eps_demo[attribute][dataset]={}
                bias_eps_dataset[attribute][dataset]={}
                for epsilon in epsilons:
                     bias_eps_dataset[attribute][dataset][epsilon]=0
                biases[dataset] = {}

                demos = list(attribute_dict[attribute][dataset].keys())
                demos = sort_demos(attribute_dict[attribute][dataset])
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
                    bias_eps_demo[attribute][dataset][demo]={}
                    for epsilon in epsilons:
                        bias_eps_demo[attribute][dataset][demo][epsilon]=0
                    for demo2 in demos:
                        try:
                            # print(attribute_dict[attribute][dataset][demo]['mean'])
                            biases[dataset][demo][demo2] = (
                                -attribute_dict[attribute][dataset][demo]['mean']+attribute_dict[attribute][dataset][demo2]['mean'])/100
                            for epsilon in epsilons:
                                bias_eps_demo[attribute][dataset][demo][epsilon]+= int(biases[dataset][demo][demo2] < -epsilon)
                                bias_eps_dataset[attribute][dataset][epsilon]+= int(biases[dataset][demo][demo2] < -epsilon)
                        except Exception as e: 
                            print(e)
                            
                            # biases[dataset][demo][demo2] = float("NaN")
                            # print('devided by zero', model, dataset, demo2)

                    bias=[]
                    for epsilon in epsilons:
                        bias_eps_demo[attribute][dataset][demo][epsilon]/=num_demos-1
                        # row.append(bias_eps_demo[model][dataset][demo][epsilon])
                        bias.append(int(np.round(100*bias_eps_demo[attribute][dataset][demo][epsilon])))
                    row.append(bias)     
                    bias_dataset=[]
                for epsilon in epsilons:
                    bias_eps_dataset[attribute][dataset][epsilon]/=(num_demos*(num_demos-1))/2  
                    bias_dataset.append(int(np.round(100*bias_eps_dataset[attribute][dataset][epsilon])))
           
                row_datasets[-1].append(bias_dataset)
            writer_object.writerow(sorted_demos) 

            writer_object.writerow(row) 


        for row in row_datasets:
            writer_object.writerow(row)
               
        csv_f.close()
# scores=scores_filter(scores)    
# print(scores)
for attribute in attribute_dict.keys():
        bias_measure(attribute_dict, method, attribute)

