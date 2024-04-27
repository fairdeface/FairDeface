#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:54:44 2023

@author: moosavi
"""
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
parser.add_argument('--fig_style', type=str,  default='all',  help=' could be anything but all to show the results for each attribute in a single plot')# Parse the argument
parser.add_argument('--original_file', type=str, required=False, help='The path to the original attribute file')# Parse the argument

args = parser.parse_args()

fig_style = args.fig_style

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
                            attribute_comparison[dataset][demo][attribute].append(difference)
                        else:
                            if org_obf[dataset][demo][pic]['org'][attribute]==org_obf[dataset][demo][pic]['obf'][attribute]:
                                attribute_comparison[dataset][demo][attribute].append(1)
                            else:
                                attribute_comparison[dataset][demo][attribute].append(0)
                                
                # for obf in attributes_all[method][dataset][demo][person]['attribute'].keys():
                #     obf_pics[dataset][demo].append(obf)
        print("********************************************")
        print(dataset, demo )
        for attribute in attributes:
            attribute_results[dataset][demo][attribute] = {}
            if attribute == 'age':
              
                mean_age = np.mean(attribute_comparison[dataset][demo][attribute])
                std_age = np.std(attribute_comparison[dataset][demo][attribute])/np.sqrt(len(attribute_comparison[dataset][demo][attribute]))
                
                attribute_results[dataset][demo][attribute]['mean'] = round(mean_age, 1)
                attribute_results[dataset][demo][attribute]['std_err'] = round(std_age, 2)

                print("\nage:")
                print('mean: ', round(mean_age, 1), "std: ", round(std_age, 2))
            else:
                # y = 'mean_' + attribute
                # exec('%s = %d' %(y, np.mean(attribute_comparison[dataset][demo][attribute])))
                print('\n', attribute)
                print( "mean: ", round(100*np.mean(attribute_comparison[dataset][demo][attribute]), 1))
                attribute_results[dataset][demo][attribute]['mean'] = round(100*np.mean(attribute_comparison[dataset][demo][attribute]), 1)

attribute_dict={}
for att in attributes:
    attribute_dict[att]={}
    for dataset in attribute_results.keys():
        attribute_dict[att][dataset]={}
        for demo in attribute_results[dataset].keys():
            attribute_dict[att][dataset][demo] = attribute_results[dataset][demo][att]
    
import pandas as pd
import pickle
from os import listdir
# attribute = {}
# pkls = listdir(".")
# pkls = [p for p in pkls if '.pkl' in p and 'Attribute_results_' in p]
import numpy as np 
import matplotlib.pyplot as plt 
plt.close('all')

fontsize=13
plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize) 
colors = ['red', 'blue', 'yellow',  'green', 'pink', 'gold', 'purple', 'brown', ]
color_dic = {"white_males": colors[0], "White_Males": colors[0], "black_males": colors[1], "Black_Males": colors[1],  
              "asian_males": colors[2], "Asian_Males": colors[2], "indian_males": colors[3], "Indian_Males": colors[3], 
             
              "white_females": colors[4], "White_Females": colors[4], "indian_females": colors[5], "Indian_Females": colors[5],  
              "black_females": colors[6], "Black_Females": colors[6], "asian_females": colors[7], "Asian_Females": colors[7], 
              "white": colors[0], "White": colors[0], "Caucasian": colors[0], 'indian': colors[3], 'Indian': colors[3],
              'black': colors[1], 'Black': colors[1], 'African': colors[1], 'asian': colors[2], 'Asian': colors[2], 
              'males': colors[0], 'Males': colors[0], "females": colors[1], "Females": colors[1]} 

hatches=['/', '\\', '//', '-', '+', 'x', 'o', 'O', '.', '*']
hatch_dic = {"white_males": hatches[0], "White_Males": hatches[0], "black_males": hatches[1], "Black_Males": hatches[1],  
              "asian_males": hatches[2], "Asian_Males": hatches[2], "indian_males": hatches[3], "Indian_Males": hatches[3], 
             
              "white_females": hatches[4], "White_Females": hatches[4], "indian_females": hatches[5], "Indian_Females": hatches[5],  
              "black_females": hatches[6], "Black_Females": hatches[6], "asian_females": hatches[7], "Asian_Females": hatches[7], 
              "white": hatches[0], "White": hatches[0], "Caucasian": hatches[0], 'indian': hatches[3], 'Indian': hatches[3],
              'black': hatches[1], 'Black': hatches[1], 'African': hatches[1], 'asian': hatches[2], 'Asian': hatches[2], 
              'males': hatches[0], 'Males': hatches[0], "females": hatches[1], "Females": hatches[1]} 

def sort_demos(total_demos):
    new_demos= {}
    for demo in total_demos.keys():
        if type(total_demos[demo])==dict:
            new_demos[demo]= total_demos[demo]['mean']
    new_demos = sorted(new_demos.items(), key=lambda x:x[1],  reverse=True)
    new_demos = [d[0] for d in new_demos]
    return new_demos


race = {'BFW':['White', 'Black', 'Asian', 'Indian'], 'DemogPairs':['White', 'Black', 'Asian']}
# csv_file = "Attributes.csv"
legend_once=1
if fig_style=='all':
        row=1
        num_rows =len(attributes)
        num_cols= 1
        fig, ax = plt.subplots(num_rows, num_cols, figsize=( 10*num_cols, 5*num_rows))

        figname = "Attribute-" + method + "-all_attributes"
        for att in  attribute_dict.keys():
            plt.subplot(num_rows, num_cols,row)
            row+=1
            datasets = list(attribute_dict[att].keys())
            X_axis = np.arange(len(datasets))

        
            k=0
            for dataset in datasets:
                sorted_demos=sort_demos(attribute_dict[att][dataset])
                i = 1/2
                demo_num = len(sorted_demos)
                Total_length = .9
                width_demo = Total_length/(demo_num) 
                for demo in sorted_demos:
                    if att !='age':
                        d=attribute_dict[att][dataset][demo]['mean']
                        plt.bar(k -Total_length/2 + width_demo*i ,  d, width_demo, label = demo, color=color_dic[demo],  hatch=hatch_dic[demo]) 
                        plt.ylim(0, 100)
                    else:
                        d=attribute_dict[att][dataset][demo]['mean']
                        stds=attribute_dict[att][dataset][demo]['std_err']
                        plt.bar(k -Total_length/2 + width_demo*i,  d, yerr=stds, width=width_demo,  label = demo, color=color_dic[demo],  hatch=hatch_dic[demo], align='center', alpha=0.5, ecolor='black', capsize=10) 
              
                    i+=1
                
                k+=1                # plt.bar(X_axis -Total_length/2 + width_demo*i , attribute[att][FR][dataset][demo]["each"]["test"], width_demo, label = demo + "-" + "each") 
            plt.xticks(X_axis, datasets, fontsize=fontsize) 
            if row==num_cols*num_rows+1:
                plt.xlabel("datasets", fontsize=fontsize) 
            if att !='age':

                plt.ylabel("Preserving Rates(%)", fontsize=fontsize-3) 
            else:
                plt.ylabel("Difference Means", fontsize=fontsize-3) 

            plt.title(att)
    
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            means_total, mean_dataset= mean_rate_calculator(attribute_dict[att], attribute_comparison, att)
           
            plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total: "+"{:.0f}".format(means_total)) 
            cl=0
            for dataset in mean_dataset.keys():
                cl+=1
                plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+": {:.0f}".format(mean_dataset[dataset])) 
                # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize-3, loc='lower center', bbox_to_anchor=(.5, 1.06), ncol=6)
            plt.tight_layout()
                

            # rowx+=1
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        # plt.show()
        directory = 'All'
        if not isdir(directory):
            mkdir(directory)
        figname = join(directory, figname)
        plt.savefig(figname +".jpg", dpi=150)
else:
    for att in  attribute_dict.keys():
        datasets = list(attribute_dict[att].keys())
        X_axis = np.arange(len(datasets))
        fig2 = plt.figure(att, figsize=(10, 5) )
        figname = "Attribute-" + method + '-' +  att

    
        k=0
        for dataset in datasets:
            sorted_demos=sort_demos(attribute_dict[att][dataset])
            i = 1/2
            demo_num = len(sorted_demos)
            Total_length = .9
            width_demo = Total_length/(demo_num) 
            for demo in sorted_demos:
                if att !='age':
                    d=attribute_dict[att][dataset][demo]['mean']
                    plt.bar(k -Total_length/2 + width_demo*i ,  d, width_demo, label = demo, color=color_dic[demo],  hatch=hatch_dic[demo]) 
                else:
                    d=attribute_dict[att][dataset][demo]['mean']
                    stds=attribute_dict[att][dataset][demo]['std_err']
                    plt.bar(k -Total_length/2 + width_demo*i,  d, yerr=stds, width=width_demo,  label = demo, color=color_dic[demo],  hatch=hatch_dic[demo], align='center', alpha=0.5, ecolor='black', capsize=10) 
          
                i+=1
            
            k+=1                # plt.bar(X_axis -Total_length/2 + width_demo*i , attribute[att][FR][dataset][demo]["each"]["test"], width_demo, label = demo + "-" + "each") 
        plt.xticks(X_axis, datasets, fontsize=fontsize) 
        plt.xlabel("datasets", fontsize=fontsize) 
        plt.ylabel("Preserving Rates(%)", fontsize=fontsize) 

        means_total, mean_dataset= mean_rate_calculator(attribute_dict[att], attribute_comparison, att)
      
        plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total: "+"{:.0f}".format(means_total)) 
        cl=0
        for dataset in mean_dataset.keys():
            cl+=1
            plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+": {:.0f}".format(mean_dataset[dataset])) 
            # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize-3, loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=5)

        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        plt.tight_layout()
        # plt.show()
        directory = join('Separate', method)
        if not isdir(directory):
            makedirs(directory)
        figname2 = join(directory, figname)
        plt.savefig(figname2 +".jpg", dpi=150)
   
plt.close('all')  
import seaborn as sns     
# sns.set(font_scale=1.4)          
plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize) 
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

my_colors=['red', 'green']
my_cmap= ListedColormap(my_colors)
bounds = [0, .8,  1]
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))
patterns = ['', 'oo', '////', 'XXX', '*']
fairness_thresh=.8
def plot_demographic_parity(attribute_dict):
    # attributes = list(attribute_dict.keys())
    datasets= list(attribute_dict[attributes[0]].keys())

    if fig_style=='all':

        num_cols= len(datasets)
        num_rows =len(attributes)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        figname = "Bias-attribute-" + method +  '-all_attributes' 

        d=1
        for attribute in attributes:
            for dataset in datasets:

                biases={}
                
                demos = list(attribute_dict[attribute][dataset].keys())
                demos=sort_demos(attribute_dict[attribute][dataset])

        
                num_demos =len(demos)
                dataset_biass = np.empty((num_demos, num_demos))
                Demos=[]
                for D in demos:
                    if "_" in D:
                        D = ''.join([x[0].upper() for x in D.split('_')])
                    Demos.append(D)
        
                i=0
        
                for demo in demos:
                    biases[demo]={}
                    for demo2 in demos:
                        try:
                          
                            biases[demo][demo2]=(attribute_dict[attribute][dataset][demo]['mean'])/(attribute_dict[attribute][dataset][demo2]['mean'])
                            if biases[demo][demo2]<=fairness_thresh:
                                print( attribute, dataset, demo, demo2, biases[demo][demo2])
                   
                        except:
                                biases[demo][demo2]= float("NaN")
                                print('devided by zero',attribute, dataset, demo2) 
                                
                    dataset_biass[i,:]= np.array(list(biases[demo].values()))
                    i+=1
                  # x = np.array(biases[att][method][dataset].values())
                dataset_biass = np.round(dataset_biass, 2)
                  # corr= dataset_biass.corr()
                
                  # Getting the Upper Triangle of the co-relation matrix
                matrix = np.triu(dataset_biass)
                annot_kws={
                  # 'size':100,
                'fontstyle':'italic',  
                'color':"k",
                'rotation':"vertical",
                'verticalalignment':'center',
                'backgroundcolor':'w'}
                plt.subplot(num_rows, num_cols, d)
                plt.title(attribute + '-' + dataset, fontsize=fontsize)
        
                    
                d+=1
                sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,  mask= matrix,  cmap=my_cmap,  norm=my_norm) 
        # plt.show()
        plt.tight_layout()
        directory = 'All'
        if not isdir(directory):
            mkdir(directory)
        figname = join(directory, figname)
        plt.savefig(figname +".jpg", dpi=150)
                    
    else:
      
        for attribute in attributes:
            num_cols= len(datasets)
            num_rows =1
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
            figname = "Bias-attribute-" + method +  '-' + attribute
    
            d=1
     
            for dataset in datasets:

                biases={}
                
                demos = list(attribute_dict[attribute][dataset].keys())
                demos=sort_demos(attribute_dict[attribute][dataset])

        
                num_demos =len(demos)
                dataset_biass = np.empty((num_demos, num_demos))
                Demos=[]
                for D in demos:
                    if "_" in D:
                        D = ''.join([x[0].upper() for x in D.split('_')])
                    Demos.append(D)
        
                i=0
        
                for demo in demos:
                    biases[demo]={}
                    for demo2 in demos:
                        try:
                          
                            biases[demo][demo2]=(attribute_dict[attribute][dataset][demo]['mean'])/(attribute_dict[attribute][dataset][demo2]['mean'])
                            if biases[demo][demo2]<=fairness_thresh:
                                print( attribute, dataset, demo, demo2, biases[demo][demo2])
                        
                        except:
                            biases[demo][demo2]= float("NaN")
                            print('devided by zero',attribute, dataset, demo2) 
                                
                    dataset_biass[i,:]= np.array(list(biases[demo].values()))
                    i+=1
                  # x = np.array(biases[att][method][dataset].values())
                dataset_biass = np.round(dataset_biass, 2)
                  # corr= dataset_biass.corr()
                
                  # Getting the Upper Triangle of the co-relation matrix
                matrix = np.triu(dataset_biass)
                annot_kws={
                  'size':100,
                'fontstyle':'italic',  
                'color':"k",
                'rotation':"vertical",
                'verticalalignment':'center',
                'backgroundcolor':'w'}
                plt.subplot(num_rows, num_cols, d)
                plt.title(attribute + '-' + dataset, fontsize=fontsize)
        
                    
                d+=1
                sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,  mask= matrix,  cmap=my_cmap,  norm=my_norm) 
        # plt.show()
            plt.tight_layout()
            directory = join('Separate', method)
            if not isdir(directory):
                makedirs(directory)
            figname2 = join(directory, figname)
            plt.savefig(figname2 +".jpg", dpi=150)


plot_demographic_parity(attribute_dict)