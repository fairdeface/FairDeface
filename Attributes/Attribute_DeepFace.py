
from deepface import DeepFace
from shutil import rmtree
import pickle
import numpy as np
from PIL import Image
from os.path import join, isdir, isfile, basename
from os import listdir, remove, walk
import os
import json, pickle
import requests
from random import shuffle, choice
import argparse

parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('--method', type=str, required=True)# Parse the argument
parser.add_argument('--datasets_path', type=str,  default='../datasets')# Parse the argument
parser.add_argument('--datasets', type=str, help='a list of datasets seperated by ,',  default='BFW, DemogPairs')# Parse the argument
parser.add_argument('--step', type=int, help='after <step> many successfull requests the results are saved',  default=500)# Parse the argument

# parser.add_argument('--detection_methods', type=str, default='mtcnn', help='choose one or more of these methods (should be separetad with ,): mtcnn, dlib, opencv, ssd, retinaface')# Parse the argument

# parser.add_argument('--method_file_path', type=str, required=True)# Parse the argument
args = parser.parse_args()
method=args.method
datasets_path=args.datasets_path
# method = 'DP1'
attributes_all_file = 'Attribute_' + method + '_.pkl'
datasets = args.datasets.replace(' ','').split(',')

step = args.step

def remove_items(test_list, item): 
  
    # using filter() + __ne__ to perform the task 
    res = list(filter((item).__ne__, test_list)) 
    return res 



def all_attribute_init():
        attribute_all = {}
        attribute_all[method] = {}
        for dataset in datasets:
            attribute_all[method][dataset] = {}
            dataset_p = join(datasets_path, method, dataset)
            demos = listdir(dataset_p)
            demos = [d for d in demos if  isdir(join(dataset_p, d))]
            for demo in demos:
                attribute_all[method][dataset][demo] = {}
                attribute_all[method][dataset][demo]['num_pics'] = 0

                attribute_all[method][dataset][demo]['num_orgs'] = 0
                attribute_all[method][dataset][demo]['fails']= set()
                demo_p = join(dataset_p, demo)
                people = listdir(demo_p)
                people = [p for p in people if isdir(join(demo_p, p))]
                for person in people:
                    attribute_all[method][dataset][demo][person] = {}
                    person_p = join(demo_p, person)
                    pics = listdir(person_p)
                    pics = [ join(dataset, demo, p) for p in pics if 'DS_Store' not in p ]

                    attribute_all[method][dataset][demo][person]['pics'] = set(pics)
                    attribute_all[method][dataset][demo][person]['passed'] = set()
                    attribute_all[method][dataset][demo][person]['fails'] = set()
                    attribute_all[method][dataset][demo][person]['attribute'] = {}
                    org_path = join(datasets_path,'Original', dataset, demo, person)
                    orgs = listdir(org_path)
                    orgs = [ p for p in orgs if 'DS_Store' not in p ]
                    attribute_all[method][dataset][demo]['num_orgs'] += len(orgs)
                    if method=='CIAGAN':
                        attribute_all[method][dataset][demo]['num_pics'] += len(pics)/2
                    else:
                        attribute_all[method][dataset][demo]['num_pics'] += len(pics)
      

        return attribute_all
def clean_folder(folder):
    for root, dirs, files in walk(folder):
        for file in files:
            if 'DS_Store' in file:
                remove(join(root, file))

        for folder in dirs:
            if 'DS_Store' in folder:
                rmtree(join(root, folder))


def check_if_the_file_exist(attributes_all_file):
    i = 4
    pickles = listdir('.')
    pickles = [p for p in pickles if ".pkl" in p and attributes_all_file[:-i] in p ]
    if (len(pickles)==0):
        attributes_all = all_attribute_init()
        return attributes_all, 1
    last_ind = -100
    for p in pickles:
        pi = int( p[len(attributes_all_file[:-i]):-i])
        if pi>last_ind:
           last_ind =pi
           last = p
    print("last_ind =", last_ind)
    flag = 1
    while(flag):
        try:
            with open(last, 'rb') as my_file:
                attributes_all = pickle.load(my_file)
                my_file.close()
                flag = 0
                print("loaded_last_ind =", last_ind)
                for k in range(1, last_ind):
                    file = attributes_all_file[:-i] + format(k, '06d') + attributes_all_file[-i:]
                    if isfile(file):
                        remove(file)
                return attributes_all, last_ind+1

        except:
               remove(last)
               last = attributes_all_file[:-i] + format(last_ind-1, '06d') + attributes_all_file[-i:]
               last_ind = last_ind-1
    




def save_close(attributes_all_file, attributes_all, ind):
    i=4
    file_name = attributes_all_file[:-i] + format(ind, '06d') + attributes_all_file[-i:]
    with  open(file_name, 'wb') as geeky_file:
        pickle.dump(attributes_all, geeky_file)
        geeky_file.close()
        x = ind
        flag = 1
        while (flag):
            try:
                file_name = attributes_all_file[:-i] + format(x, '06d') + attributes_all_file[-i:]
                if isfile(file_name):
                    with open(file_name, 'rb') as my_file:
                        attributes_all = pickle.load(my_file)
                        my_file.close()
                        x = x-1
                        flag = 0
            except:
                flag =1
        for j in range(x+1):
            file_name = attributes_all_file[:-i] + format(x, '06d') + attributes_all_file[-i:]
            if isfile(file_name):
                remove(file_name)
        return ind+1


attributes_all, ind = check_if_the_file_exist(attributes_all_file)

for dataset in datasets:
    # dataset_p = join('dataset', dataset)
    # obfuscated = join('obfuscateds', method, dataset+"_obfuscated")
    for demo in attributes_all[method][dataset].keys():
        print(dataset, demo)
        
        count = 0
        change = 0 
        for person in attributes_all[method][dataset][demo].keys():
            if person not in  ['num_pics', 'num_orgs', 'fails', 'fail_ratio', 'miss_ratio' ]:
                for pic in  attributes_all[method][dataset][demo][person]['pics']:
                    pic_path= join(datasets_path, method, pic)

                    if pic not in attributes_all[method][dataset][demo][person]['passed']:
                        if pic not in attributes_all[method][dataset][demo][person]['fails']:
                            try:
                                 analysis = DeepFace.analyze(img_path = pic_path, actions = ["age", "gender", "emotion", "race"], detector_backend='mtcnn')
                                 attribute = {}
                                 if len(analysis)==1:
                                     att = analysis[0]
                                 # else:
                                 #    attributes_all[method][dataset][demo]['more_than_one'].append(pic)
                                     
                                 attribute['age'] = att['age']
                                 attribute['emotion'] = att['dominant_emotion']
                                 attribute['gender'] = att['dominant_gender']
                                 attribute['race'] = att['dominant_race']
  
                                 # print("passed first:", pic)
                                 attributes_all[method][dataset][demo][person]['attribute'][pic] = attribute
                                 attributes_all[method][dataset][demo][person]['passed'].add(pic)
                                 count += 1
                                 change = 1
                            except Exception as e:
                                  print(e)
                                  attributes_all[method][dataset][demo][person]['fails'].add(pic)
                                  attributes_all[method][dataset][demo]['fails'].add(pic)
                
                        else:

                             try:
                                 analysis = DeepFace.analyze(img_path=pic_path, actions = ["age", "gender", "emotion", "race"], detector_backend='mtcnn')
                                 attribute = {}
                                 if len(analysis)==1:
                                     att = analysis[0]
                                 # else:
                                 #     attributes_all[method][dataset][demo]['more_than_one'].append(pic)
                                     
                                 attribute['age'] = att['age']
                                 attribute['emotion'] = att['dominant_emotion']
                                 attribute['gender'] = att['dominant_gender']
                                 attribute['race'] = att['dominant_race']
                                 
                                 # print("passed first:", pic)
                                 attributes_all[method][dataset][demo][person]['attribute'][pic] = attribute
                                 attributes_all[method][dataset][demo][person]['passed'].add(pic)
                                 attributes_all[method][dataset][demo][person]['fails'].remove(pic)
                                 
                                 attributes_all[method][dataset][demo]['fails'].remove(pic)
                                 count += 1
                                 change = 1
     
                             except Exception as e:
                                x = 1
                        if count%step == 0 and change == 1:
                         ind = save_close(attributes_all_file, attributes_all, ind) 
                         change = 0
  
        lorg = attributes_all[method][dataset][demo]['num_orgs']
        lpics=attributes_all[method][dataset][demo]['num_pics']
        if method !='Original':
            attributes_all[method][dataset][demo]['miss_ratio'] =   round(100.0*(1-lpics/lorg), 2)
            
        lf = len(attributes_all[method][dataset][demo]['fails'])

        attributes_all[method][dataset][demo]['fail_ratio'] =   round(100.0*lf/lpics, 2)
        print(demo)
        print("fail ratio = ", attributes_all[method][dataset][demo]['fail_ratio'])
        if method!='Original':
            print("miss ratio = ", attributes_all[method][dataset][demo]['miss_ratio'])
        # print("The number of photos with more than faces = ", l_more)
        if (change):
            ind = save_close(attributes_all_file, attributes_all, ind) 
#