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
parser = argparse.ArgumentParser()  # Add an argument
parser.add_argument('--detection_file', type=str, required=True,
                    help='path to the detection')  # Parse the argument
parser.add_argument('--pass_rates', type=str, required=False,
                    help='choose yes if you want to show passing rates', default='no')  # Parse the argument
parser.add_argument('--fig_style', type=str, required=False,
                    help='choose <seperate> to have a figure for each model', default='all')  # Parse the argument

args = parser.parse_args()
detection_path = args.detection_file
pass_yes_no = args.pass_rates
fig_style = args.fig_style
# detection_path='Detection_CIAGAN_000000.pkl'
# pass_yes_no='yes'
# fig_style='separate'
fontsize = 10
fairness_thresh = .8

colors = ['red', 'blue', 'yellow',  'green',
          'pink', 'gold', 'purple', 'brown', ]
color_dic = {"white_males": colors[0], "White_Males": colors[0], "black_males": colors[1], "Black_Males": colors[1],
             "asian_males": colors[2], "Asian_Males": colors[2], "indian_males": colors[3], "Indian_Males": colors[3],

             "white_females": colors[4], "White_Females": colors[4], "indian_females": colors[5], "Indian_Females": colors[5],
             "black_females": colors[6], "Black_Females": colors[6], "asian_females": colors[7], "Asian_Females": colors[7],
             "white": colors[0], "White": colors[0], "Caucasian": colors[0], 'indian': colors[3], 'Indian': colors[3],
             'black': colors[1], 'Black': colors[1], 'African': colors[1], 'asian': colors[2], 'Asian': colors[2],
             'males': colors[0], 'Males': colors[0], "females": colors[1], "Females": colors[1]}

hatches = ['/', '\\', '//', '-', '+', 'x', 'o', 'O', '.', '*']
hatch_dic = {"white_males": hatches[0], "White_Males": hatches[0], "black_males": hatches[1], "Black_Males": hatches[1],
             "asian_males": hatches[2], "Asian_Males": hatches[2], "indian_males": hatches[3], "Indian_Males": hatches[3],

             "white_females": hatches[4], "White_Females": hatches[4], "indian_females": hatches[5], "Indian_Females": hatches[5],
             "black_females": hatches[6], "Black_Females": hatches[6], "asian_females": hatches[7], "Asian_Females": hatches[7],
             "white": hatches[0], "White": hatches[0], "Caucasian": hatches[0], 'indian': hatches[3], 'Indian': hatches[3],
             'black': hatches[1], 'Black': hatches[1], 'African': hatches[1], 'asian': hatches[2], 'Asian': hatches[2],
             'males': hatches[0], 'Males': hatches[0], "females": hatches[1], "Females": hatches[1]}

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


def plot_main(detection, obf_method, fig_style, fail_pass):
    if fail_pass == 'fail_ratio':
        Detection_Passing = "Detection"
    elif fail_pass == 'miss_ratio':
        Detection_Passing = "Passing"
    if fig_style == 'all':
        models = list(detection.keys())
        num_rows = len(models)
        num_cols = 1
        fig, ax = plt.subplots(
            num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

        figname = Detection_Passing+"-" + obf_method + "-" + 'all_models'

        Total_length = .9
        d = 0
        for model in detection.keys():
            datasets = list(detection[models[0]].keys())
            d += 1
            plt.subplot(num_rows, num_cols, d)
            plt.title(obf_method + '-' + model)

            J = 0
            # fig2 = plt.figure('Detection Rates for ' + method + '-' + model )
            X_axis = np.arange(len(datasets))
            for dataset in datasets:
                demos = list(detection[model][dataset].keys())
                demos = sort_demos(detection[model][dataset], fail_pass)
                num_demos = len(demos)
                width = Total_length/(len(demos))
                i = 1/2
                for demo in demos:
                    plt.bar(J - Total_length/2 + width*i,  100 -
                            detection[model][dataset][demo][fail_pass], width, hatch=hatch_dic[demo], label=demo, color=color_dic[demo])
                    i += 1

                J += 1

                plt.xticks(X_axis, datasets, fontsize=fontsize)
                plt.xlabel("Datasets", fontsize=fontsize)
                plt.ylabel(Detection_Passing + " Rates", fontsize=fontsize)
                # plt.title("Number of Students in each group")

                # , loc="upper left", ncol=2, bbox_to_anchor=(0.5, 1.0)
         
            means_total, mean_dataset= mean_detection_passing_calculator(detection[model], fail_pass)

            plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total: "+"{:.0f}".format(means_total)) 
            # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
            # plt.text(-1,means_total, "{:.0f}".format(means_total), color=colors[0],   ha="left", va="center")
            cl=0
            for dataset in mean_dataset.keys():
                cl+=1
                plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+": {:.0f}".format(mean_dataset[dataset])) 
                # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
                # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(
                ), fontsize=fontsize-2, loc='center left', bbox_to_anchor=(1, 0.5))
            # manager = plt.get_current_fig_manager()
            # manager.full_screen_toggle()
        plt.tight_layout()
        if fail_pass == 'fail_ratio':
            directory = join('Detection_Rates', 'All')
            
           

        elif fail_pass == 'miss_ratio':
            directory = join('Passing_Rates', 'All')
        if not isdir(directory):
            makedirs(directory)
        figname2 = join(directory, figname)
        plt.savefig(figname2 + ".jpg", dpi=150)
        plt.close('all')
    else:

        Total_length = .9
        for model in detection.keys():

            figname = Detection_Passing+"-" + obf_method + "-" + model
            J = 0
            datasets = list(detection[model].keys())
            # fig2 = plt.figure('Detection Rates for ' + method + '-' + model )
            X_axis = np.arange(len(datasets))
            for dataset in datasets:
                demos = list(detection[model][dataset].keys())
                demos = sort_demos(detection[model][dataset], fail_pass)

                num_demos = len(demos)
                width = Total_length/(len(demos))
                i = 1/2
                for demo in demos:
                    plt.bar(J - Total_length/2 + width*i,  100 -
                            detection[model][dataset][demo][fail_pass], width, hatch=hatch_dic[demo], label=demo, color=color_dic[demo])
                    i += 1

                J += 1

                plt.xticks(X_axis, datasets, fontsize=fontsize)
                plt.xlabel("Datasets", fontsize=fontsize)
                plt.ylabel(Detection_Passing + " Rates", fontsize=fontsize)
  

            means_total, mean_dataset= mean_detection_passing_calculator(detection[model], fail_pass)

            plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total: "+"{:.0f}".format(means_total)) 
            # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
            # plt.text(-1,means_total, "{:.0f}".format(means_total), color=colors[0],   ha="left", va="center")
            cl=0
            for dataset in mean_dataset.keys():
                cl+=1
                plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+": {:.0f}".format(mean_dataset[dataset])) 
                # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
                # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(
                ), fontsize=fontsize-2, loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.title(obf_method + '-' + model)
            if fail_pass == 'fail_ratio':
                directory = join('Detection_Rates', 'Separate', obf_method)
            elif fail_pass == 'miss_ratio':
                directory = join('Passing_Rates', 'Separate', obf_method)

            if not isdir(directory):
                makedirs(directory)
            figname2 = join(directory, figname)
            plt.savefig(figname2 + ".jpg", dpi=150)
            plt.close('all')


for obf_method in detection.keys():
    plot_main(detection[obf_method], obf_method, fig_style, 'fail_ratio')
    if pass_yes_no == 'yes' and obf_method != 'Original':
        plot_main(detection[obf_method], obf_method, fig_style, 'miss_ratio')

plt.close('all')
# disparity maps
my_colors = ['red', 'green']
my_cmap = ListedColormap(my_colors)
bounds = [0, .8,  1]
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))
patterns = ['', 'oo', '////', 'XXX', '*']


def plot_demographic_parity(detection, obf_method, term):
    if fig_style == 'all':
        models = list(detection.keys())
        num_rows = len(models)
        datasets = list(detection[models[0]].keys())
        num_cols = len(datasets)
        fig, ax = plt.subplots(
            num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        if term == 'fail_ratio':
            figname = "Bias-Detection-" + obf_method + "-" + 'all_models'
        elif term == 'miss_ratio':
            figname = "Bias-Passing_Rates-" + obf_method + "-" + 'all_models'

        d = 1
        for model in detection.keys():
            biases = {}

            # datasets = list(detection[model].keys())
            # num_cols= len(datasets)
            # num_rows= int(np.ceil(len(datasets)/2))
            for dataset in datasets:

                biases[dataset] = {}

                demos = list(detection[model][dataset].keys())
                demos = sort_demos(detection[model][dataset], term)

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
                    for demo2 in demos:
                        try:

                            biases[dataset][demo][demo2] = (
                                100-detection[model][dataset][demo][term])/(100-detection[model][dataset][demo2][term])
                            if biases[dataset][demo][demo2] <= fairness_thresh:
                                print(model, dataset, demo, demo2,
                                      biases[dataset][demo][demo2])
                        except:
                            biases[dataset][demo][demo2] = float("NaN")
                            print('devided by zero', model, dataset, demo2)

                    dataset_biass[i, :] = np.array(
                        list(biases[dataset][demo].values()))
                    i += 1
                 # x = np.array(biases[att][method][dataset].values())
                dataset_biass = np.round(dataset_biass, 2)
                # corr= dataset_biass.corr()

                # Getting the Upper Triangle of the co-relation matrix
                matrix = np.triu(dataset_biass)
                annot_kws = {
                    'size': 100,
                    'fontstyle': 'italic',
                    'color': "k",
                    'rotation': "vertical",
                    'verticalalignment': 'center',
                    'backgroundcolor': 'w'}
                plt.subplot(num_rows, num_cols, d)
                plt.title(model + '-' + dataset)

                d += 1
                sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,
                            mask=matrix,  cmap=my_cmap,  norm=my_norm)
        # plt.show()
        plt.tight_layout()
        if term == 'fail_ratio':
            directory = join('Detection_Rates', 'All')
        elif term == 'miss_ratio':
            directory = join('Passing_Rates', 'All')
        if not isdir(directory):
            makedirs(directory)
        figname2 = join(directory, figname)
        plt.savefig(figname2+'.jpg', dpi=150)

    else:

        for model in detection.keys():
            plt.close('all')

            if term == 'fail_ratio':
                figname = "Bias-Detection-" + obf_method + '-' + model
            elif term == 'miss_ratio':
                figname = "Bias-Passing_Rates-" + obf_method + '-' + model

            biases = {}
            d = 1
            datasets = list(detection[model].keys())
            num_rows = 1
            # num_rows= int(np.ceil(len(datasets)/2))
            num_cols = len(datasets)
            fig, ax = plt.subplots(
                num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

            for dataset in datasets:

                biases[dataset] = {}

                demos = list(detection[model][dataset].keys())
                demos = sort_demos(detection[model][dataset], term)

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
                    for demo2 in demos:
                        try:
                            biases[dataset][demo][demo2] = (
                                100-detection[model][dataset][demo][term])/(100-detection[model][dataset][demo2][term])
                            if biases[dataset][demo][demo2] <= fairness_thresh:
                                print(model, dataset, demo, demo2,
                                      biases[dataset][demo][demo2])
                        except:
                            biases[dataset][demo][demo2] = float("NaN")
                            print('devided by zero', model, dataset, demo2)

                    dataset_biass[i, :] = np.array(
                        list(biases[dataset][demo].values()))
                    i += 1
                 # x = np.array(biases[att][method][dataset].values())
                dataset_biass = np.round(dataset_biass, 2)
                # corr= dataset_biass.corr()

                # Getting the Upper Triangle of the co-relation matrix
                matrix = np.triu(dataset_biass)
                annot_kws = {
                    'size': 100,
                    'fontstyle': 'italic',
                    'color': "k",
                    'rotation': "vertical",
                    'verticalalignment': 'center',
                    'backgroundcolor': 'w'}
                plt.subplot(num_rows, num_cols, d)
                plt.title(dataset)

                d += 1
                sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,
                            mask=matrix,  cmap=my_cmap,  norm=my_norm)
            # plt.show()
            plt.tight_layout()
            if term == 'fail_ratio':
                directory = join('Detection_Rates', 'Separate', obf_method)
            elif term == 'miss_ratio':
                directory = join('Passing_Rates', 'Separate', obf_method)
            if not isdir(directory):
                makedirs(directory)
            figname2 = join(directory, figname)
            plt.savefig(figname2+'.jpg', dpi=150)


for obf_method in detection.keys():
    plot_demographic_parity(detection[obf_method], obf_method, 'fail_ratio')
    if pass_yes_no == 'yes' and obf_method != 'Original':
        plot_demographic_parity(
            detection[obf_method], obf_method, 'miss_ratio')
