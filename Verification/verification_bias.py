#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:07:fontsize 2023

@author: moosavi
"""
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
verification_path = args.file
task='verification'

epsilons=np.array(list(args.epsilons.replace(" ", "").split(",")), dtype=float)
if 'same-photo' in verification_path:
    ch = 'same-photo'
    ylabel = 'Obfuscation Success Rates'
elif 'different-photo' in verification_path:
    ch = 'different-photo'
    ylabel = 'Obfuscation Success Rates'
elif 'same-identity' in  verification_path:
    ch = 'same-identity'
    ylabel = 'True Positive Rates'
elif 'different-identity' in verification_path:
    ch = 'different-identity'
    ylabel = 'True Negative Rates'

def decission_maker(distance, threshold):
    if ch=='same-photo' or ch =='different-photo' or ch=='different-identity':
        return distance>threshold
    elif ch=='same-identity':
        return distance<threshold
    
f = open(verification_path, 'rb')
verification = pickle.load(f)
f.close()
plt.close('all')

distances_demo ={}
epsilons=[.02, .05, .1, .15, .2]

thresholds={'ArcFace':.765, 'Facenet':0.674, 'VGG-Face':.297, 'Dlib':0.078 , 'OpenFace': 0.227, 'DeepID': .017,  'DeepFace': 0.189 }
method= list(verification.keys())[0]
for model in verification[method].keys():
    #print(model)
    distances_demo[model] = {}
    for dataset in verification[method][model].keys():
        distances_demo[model][dataset] = {}
        for demo in verification[method][model][dataset].keys():
            distances_demo[model][dataset][demo] = {}
            distances_demo[model][dataset][demo]['verified']= []
            distances_demo[model][dataset][demo]['distance']= []
            for person in verification[method][model][dataset][demo].keys():
                if type(verification[method][model][dataset][demo][person]) is dict:
                    for ver_pic_results in verification[method][model][dataset][demo][person]['distance']:
                        distances_demo[model][dataset][demo]['verified'].append(ver_pic_results['verified'])
                        distances_demo[model][dataset][demo]['distance'].append(float(ver_pic_results['distance']))
    

datasets=['BFW', 'DemogPairs']
pair_datasets = [d for d in distances_demo[model].keys() if d in datasets ]
#success rates for thresholds 0 to 100                        
success_rates={}                    
total_pairs={}
for model in distances_demo.keys():
    success_rates[model] = {}
    total_pairs[model]={}

    for dataset in pair_datasets:
        total_pairs[model][dataset]={}

        success_rates[model][dataset] = {}
        for demo in distances_demo[model][dataset].keys():
            total_pairs[model][dataset][demo]=len(distances_demo[model][dataset][demo]['distance'])

  
            summ = 0
            for distance in distances_demo[model][dataset][demo]['distance']:
                    summ +=decission_maker(distance, thresholds[model])
            try:
                success_rates[model][dataset][demo] = ( summ/len(distances_demo[model][dataset][demo]['distance']))
            except Exception as e:
                print(e)


#Races
Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian'], "RFW": ['Asian',  'African', 'Caucasian', 'Indian']}
# Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian']}
race_datasets= [d for d in verification[method][model].keys() if d in Races.keys()]

success_rates_race = {}
total_race = {}

for model in distances_demo.keys():
    total_race[model]={}
    sum_race = {}
    for dataset in race_datasets:
        total_race[model][dataset] = {}
        sum_race[dataset]= {}
        for race in Races[dataset]:
            total_race[model][dataset][race] = 0
            sum_race[dataset][race]=0

    success_rates_race[model] = {}
    for dataset in race_datasets:
        success_rates_race[model][dataset] = {}
        for demo in distances_demo[model][dataset].keys(): 
            for race in Races[dataset]:
                success_rates_race[model][dataset][race] = {}
                if race in demo:
                    race_p = race
                    total_race[model][dataset][race_p] += len(distances_demo[model][dataset][demo]['distance'])
       
            for distance in distances_demo[model][dataset][demo]['distance']:
                    sum_race[dataset][race_p] += decission_maker(distance, thresholds[model])

        for race in Races[dataset]:
               try:
                   success_rates_race[model][dataset][race] = sum_race[dataset][race]/total_race[model][dataset][race]
               except Exception as e:
                    print(e)
           
 
#Genders
datasets= {"BFW", "DemogPairs", "CelebA"}
Genders = ["Males", "Females"]

gen_datasets= [d for d in verification[method][model].keys() if d in datasets]
success_rates_Gender = {}
total_Gender = {}

for model in distances_demo.keys():
    
    sum_Gender = {}
    total_Gender[model] = {}
    for dataset in gen_datasets:
        total_Gender[model][dataset] = {}
        sum_Gender[dataset]= {}
        for Gender in Genders:
            total_Gender[model][dataset][Gender] = 0
            sum_Gender[dataset][Gender]=0

    
    success_rates_Gender[model] = {}
    for dataset in gen_datasets:
        success_rates_Gender[model][dataset] = {}
        for Gender in Genders:
            success_rates_Gender[model][dataset][Gender] = {}
            
        for demo in distances_demo[model][dataset].keys(): 
                
            if 'female' in demo or 'Female' in demo:
                Gender = "Females"
            else:
                Gender = "Males"
            total_Gender[model][dataset][Gender] += len(distances_demo[model][dataset][demo]['distance'])
            for distance in distances_demo[model][dataset][demo]['distance']:
                    sum_Gender[dataset][Gender] += decission_maker(distance, thresholds[model])

        for Gender in Genders:
                try:
                    success_rates_Gender[model][dataset][Gender] = sum_Gender[dataset][Gender]/total_Gender[model][dataset][Gender]
                except Exception as e:
                    print(e)



#success rates based on the models suggestions    
success_rates_verified={}                    
for model in distances_demo.keys():
    success_rates_verified[model] = {}
    for dataset in distances_demo[model].keys():
        success_rates_verified[model][dataset] = {}
        for demo in distances_demo[model][dataset].keys():  
            summ = 0
            for verified in distances_demo[model][dataset][demo]['verified']:
                if verified == False:
                    summ+=1
            try:
                success_rates_verified[model][dataset][demo] = summ/len(distances_demo[model][dataset][demo]['verified'])
            except Exception as e:
                print(e)





# #Races Verified
# # Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian'], "RFW": ['Asian',  'African', 'Caucasian', 'Indian']}
# Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian']}

# success_rates_race_verified = {}
# for model in distances_demo.keys():
    
#     sum_race = {}
#     total_race = {}
#     for dataset in race_datasets:
#         total_race[dataset] = {}
#         sum_race[dataset]= {}
#         for race in Races[dataset]:
#             total_race[dataset][race] = 0
#             sum_race[dataset][race]= 0
            
#     success_rates_race_verified[model] = {}
#     for dataset in race_datasets:
#         success_rates_race_verified[model][dataset] = {}
#         for demo in distances_demo[model][dataset].keys(): 
#             for race in Races[dataset]:
#                 success_rates_race_verified[model][dataset][race] = {}
#                 if race in demo:
#                     race_p = race
#                     total_race[dataset][race_p] += len(distances_demo[model][dataset][demo]['distance'])

#             for verified in distances_demo[model][dataset][demo]['verified']:
#                 if verified == False:
#                     sum_race[dataset][race_p]+=1

#         for race in Races[dataset]:
#             try:
#               success_rates_race_verified[model][dataset][race] = sum_race[dataset][race]/total_race[dataset][race]
#             except Exception as e:
#                 print(e)
              

# success_rates_Gender_verified = {}
# for model in distances_demo.keys():
    
#     sum_Gender = {}
#     total_Gender = {}
#     for dataset in gen_datasets:
#         total_Gender[dataset] = {}
#         sum_Gender[dataset]= {}
#         for Gender in Genders:
#             total_Gender[dataset][Gender] = 0
#             sum_Gender[dataset][Gender]= 0

    
#     success_rates_Gender_verified[model] = {}
#     for dataset in gen_datasets:
#         success_rates_Gender_verified[model][dataset] = {}
#         for Gender in Genders:
#             success_rates_Gender_verified[model][dataset][Gender] = {}
            
#         for demo in distances_demo[model][dataset].keys(): 
                
#             if 'female' in demo or 'Female' in demo:
#                 Gender = "Females"
#             else:
#                 Gender = "Males"
#             total_Gender[dataset][Gender] += len(distances_demo[model][dataset][demo]['distance'])
        
#             for verified in distances_demo[model][dataset][demo]['verified']:
#                 if verified == False:
#                     sum_Gender[dataset][Gender]+=1

#         for Gender in Genders:
#             try:
#                 success_rates_Gender_verified[model][dataset][Gender] = sum_Gender[dataset][Gender]/total_Gender[dataset][Gender]                 
#             except Exception as e:
#                 print(e)
               

success_rates_all= {}
success_rates_all['pairs'] = success_rates
success_rates_all['race'] = success_rates_race
success_rates_all['gender'] = success_rates_Gender
# fol="Success_Rates"
# if not isdir(fol):
#     makedirs(fol)
# success_file= join(fol, "Success_rates_" + verification_file)
# with open(success_file, 'wb') as s_f:
#     pickle.dump(success_rates_all,s_f )
#     s_f.close()

def mean_verification_passing_calculator(rates, term):
    if term=='fail_ratio':
        fail_total=0
        sum_total=0
        means_verification = {}
        for dataset in rates.keys():
            summ_fails = 0
            summ_all = 0
            for demo in rates[dataset].keys():
                summ_fails += len(rates[dataset][demo]['fails'])
                fail_total+=len(rates[dataset][demo]['fails'])
                summ_all += rates[dataset][demo]['num_pics']
                sum_total+=rates[dataset][demo]['num_pics']
            means_verification[dataset] = round( 100*(1-(summ_fails/summ_all)), 2)
        means_total= round( 100*(1-(fail_total/sum_total)), 2)
        return means_total, means_verification
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

   

def sort_demos(total_demos):
    new_demos= {}
    for demo in total_demos.keys():
        new_demos[demo]= total_demos[demo]
    new_demos = sorted(new_demos.items(), key=lambda x:x[1],  reverse=True)
    new_demos = [d[0] for d in new_demos]
    return new_demos


directory='bias'
if not isdir(directory):
    makedirs(directory)

csv_datasets=directory +"/" +  task + '-'+ method+ '-'+ ch+'-bias.csv'
with open(csv_datasets, 'w') as csv_f:
    writer_object = writer(csv_f)
    writer_object.writerow(['epsilons=' +str(epsilons) ])
    csv_f.close()
csv_f=open(csv_datasets, 'a')
writer_object = writer(csv_f)
writer_object.writerow([method])
    

def bias_measure(verification, term, ch):
        writer_object.writerow([""])

        writer_object.writerow([term])
        models = list(verification.keys())
   
        datasets = list(verification[models[0]].keys())
        first_row=[""]
        second_row=[""]
        row_datasets=[""]    
        row_datasets.append(['Dataset Bias'])
        row_datasets.append([""])
        for dataset in datasets:
            first_row.append(dataset)
            row_datasets[-1].append(dataset)
            demos = list(verification[models[0]][dataset].keys())
            demos = sort_demos(verification[models[0]][dataset])
            for demo in demos:
                second_row.append(demo)
                first_row.append("")



                    
        writer_object.writerow(first_row)
        writer_object.writerow(second_row)
        bias_eps_demo={}
        bias_eps_dataset={}
     
        for model in verification.keys():
            
            row_datasets.append([model])
            row=[model]

            biases = {}
            bias_eps_demo[model]={}
            bias_eps_dataset[model]={}

            # datasets = list(verification[model].keys())
            # num_cols= len(datasets)
            # num_rows= int(np.ceil(len(datasets)/2))
            for dataset in datasets:
                
                bias_eps_demo[model][dataset]={}
                bias_eps_dataset[model][dataset]={}
                for epsilon in epsilons:
                     bias_eps_dataset[model][dataset][epsilon]=0
                biases[dataset] = {}

                demos = list(verification[model][dataset].keys())
                demos = sort_demos(verification[model][dataset])
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
                               verification[model][dataset][demo])/(verification[model][dataset][demo2])
                            for epsilon in epsilons:
                                bias_eps_demo[model][dataset][demo][epsilon]+= int(biases[dataset][demo][demo2] < (1-epsilon))
                                bias_eps_dataset[model][dataset][epsilon]+= int(biases[dataset][demo][demo2] < (1-epsilon))
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
            writer_object.writerow(row) 

        for row in row_datasets:
            writer_object.writerow(row)
               

# for obf_method in verification.keys():
#     bias_measure(verification[obf_method], obf_method, 'fail_ratio')
bias_measure(success_rates, 'pairs',ch)
bias_measure(success_rates_race, 'race', ch)
bias_measure(success_rates_Gender,'gender', ch)  

csv_f.close()
