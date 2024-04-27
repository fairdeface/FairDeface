
from deepface import DeepFace
from shutil import rmtree
import pickle
import numpy as np
from PIL import Image
from os.path import join, isdir, isfile, basename
from os import listdir, remove, walk
import os
import json, pickle
import requests, time
from random import shuffle, choice
import argparse


parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('--method', type=str, required=True, help='Obfuscation method or Original as in directory names in datasets directory')# Parse the argument
# parser.add_argument('--same_photo', type=str, help='should be yes or no. for the Original choice, use --same_identity', required='True')# Parse the argument

parser.add_argument('--datasets_path', type=str,  default='../datasets')# Parse the argument
parser.add_argument('--datasets', type=str,  default='BFW,DemogPairs')# Parse the argument
parser.add_argument('--step', type=int,  default=500)# Parse the argument

parser.add_argument('--FR_models', type=str, default='ArcFace', help='choose one or more of these methods (should be separetad with ,): DeepFace , VGG-Face, Facenet, OpenFace, DeepID, Dlib, ArcFace)# Parse the argument')
parser.add_argument('--detector', type=str, default='mtcnn', help='choose one of these methods: mtcnn, dlib, opencv, ssd, retinaface')# Parse the argument
# parser.add_argument('--same_identity', type=str, help='This is for no obfuscation choice(method==Original) to calculate TPRs or TFRs. should be yes(TPRs) or no(TFRs)', default='yes')# Parse the argument

# parser.add_argument('--method_file_path', type=str, required=True)# Parse the argument
args = parser.parse_args()
method = args.method
detector=args.detector
datasets = args.datasets.split(',')
representation_all_file = "Identification_" + method +"_.pkl"
datasets_path=args.datasets_path
# for d in datasets:
#     representation_all_file+= "_" + d
# representation_all_file+="_.pkl"
if args.FR_models=='all':
    models = [
      "DeepID", 
      "VGG-Face", 
      "Facenet", 
      "Facenet512", 
      "OpenFace", 
      "DeepFace", 
      "Dlib", 
      "ArcFace", 
    ]
else:
    models= args.FR_models.split(',')

# method_path = join("obfuscateds", method)
step = args.step

def all_representation_init():
        representation_all = {}
        representation_all[method] = {}
        for dataset in datasets:
            representation_all[method][dataset] = {}
            for model in models:

                representation_all[method][dataset][model] = {}
                dataset_p = join(datasets_path, method, dataset)
                demos = listdir(dataset_p)
                demos = [d for d in demos if  isdir(join(dataset_p, d))]
                for demo in demos:
                    representation_all[method][dataset][model][demo] = {}
                    representation_all[method][dataset][model][demo]['num_pics'] = 0
    
                    representation_all[method][dataset][model][demo]['num_orgs'] = 0
                    representation_all[method][dataset][model][demo]['fails']= set()
                    demo_p = join(dataset_p, demo)
                    people = listdir(demo_p)
                    people = [p for p in people if isdir(join(demo_p, p))]
                    for person in people:
                        representation_all[method][dataset][model][demo][person] = {}
                        person_p = join(demo_p, person)
                        pics = listdir(person_p)
                        pics = [ join(dataset, demo, person, p) for p in pics if 'DS_Store' not in p ]
    
                        representation_all[method][dataset][model][demo][person]['pics'] = set(pics)
                        representation_all[method][dataset][model][demo][person]['passed'] = set()
                        representation_all[method][dataset][model][demo][person]['fails'] = set()
                        representation_all[method][dataset][model][demo][person]['representations'] = {}

                        org_path = join(datasets_path,'Original', dataset, demo, person)
                        orgs = listdir(org_path)
                        orgs = [ p for p in orgs if 'DS_Store' not in p ]
                        representation_all[method][dataset][model][demo]['num_orgs'] += len(orgs)
                        if method=='CIAGAN':
                            pics = [ p for p in pics if 'org' not in p ]

                        representation_all[method][dataset][model][demo]['num_pics'] += len(pics)
        return representation_all



# datasets = ['BFW', 'RFW', 'DemogPairs', 'CelebA']




def check_if_the_file_exist(representation_all_file):
    i = 4
    pickles = listdir('.')
    pickles = [p for p in pickles if ".pkl" in p and representation_all_file[:-i] in p ]
    if (len(pickles)==0):
        representation_all = all_representation_init()
        return representation_all, 1
    last_ind = 0
    for p in pickles:
        pi = int( p[len(representation_all_file[:-i]):-i])
        if pi>last_ind:
           last_ind =pi
           last = p
    print("last_ind =", last_ind)
    flag = 1
    while(flag):
        try:
            with open(last, 'rb') as my_file:
                representation_all = pickle.load(my_file)
                my_file.close()
                flag = 0
                print("loaded_last_ind =", last_ind)
                for k in range(1, last_ind):
                    file = representation_all_file[:-i] + format(k, '06d') + representation_all_file[-i:]
                    if isfile(file):
                        remove(file)
                return representation_all, last_ind+1

        except:
               remove(last)
               last = representation_all_file[:-i] + format(last_ind-1, '06d') + representation_all_file[-i:]
               last_ind = last_ind-1
    




def save_close(representation_all_file, representation_all, ind):
    i=4
    file_name = representation_all_file[:-i] + format(ind, '06d') + representation_all_file[-i:]
    with  open(file_name, 'wb') as geeky_file:
        pickle.dump(representation_all, geeky_file)
        geeky_file.close()
        x = ind
        flag = 1
        while (flag):
            try:
                file_name = representation_all_file[:-i] + format(x, '06d') + representation_all_file[-i:]
                if isfile(file_name):
                    with open(file_name, 'rb') as my_file:
                        representation_all = pickle.load(my_file)
                        my_file.close()
                        x = x-1
                        flag = 0
            except:
                flag =1
        for j in range(x+1):
            file_name = representation_all_file[:-i] + format(x, '06d') + representation_all_file[-i:]
            if isfile(file_name):
                remove(file_name)
        return ind+1

count2 = 0
representation_all, ind = check_if_the_file_exist(representation_all_file)

for dataset in datasets:
     for model in models:
       # dataset_p = join('dataset', dataset)
            for demo in representation_all[method][dataset][model].keys():
                print(model, dataset, demo)
    
                count = 0
                change = 0 
             
                for person in representation_all[method][dataset][model][demo].keys():
                    if person not in ["num_pics", "fails", "fail_ratio", "miss_ratio", "num_orgs"]:
                        for pic in  representation_all[method][dataset][model][demo][person]['pics']:
                            pic_p = join(datasets_path, method, pic)

                            if pic not in representation_all[method][dataset][model][demo][person]['passed']:
                                if pic not in representation_all[method][dataset][model][demo][person]['fails']:
                                    try:
                                         embedding_objs = DeepFace.represent(img_path = pic_p, model_name = model, detector_backend=detector)
                                         representation_all[method][dataset][model][demo][person]['representations'][pic] = embedding_objs
    
                                         representation_all[method][dataset][model][demo][person]['passed'].add(pic)
                                         count += 1
                                         change = 1
    
                                    except Exception as e:
                                         
                                         representation_all[method][dataset][model][demo][person]['fails'].add(pic)
                                         representation_all[method][dataset][model][demo]['fails'].add(pic)
                                    
                                else:
                                    
                                    try:
                                         embedding_objs = DeepFace.represent(img_path = pic_p, model_name = model, detector_backend=detector)
                                         representation_all[method][dataset][model][demo][person]['representations'][pic] = embedding_objs
    
                                         representation_all[method][dataset][model][demo][person]['passed'].add(pic)
                                         representation_all[method][dataset][model][demo][person]['fails'].remove(pic)
                                         representation_all[method][dataset][model][demo]['fails'].remove(pic)
                                         count += 1
                                         change = 1
                                
                                    except Exception as e:
                                        count2 += 1
    
    
                                if count%step == 0 and change==1:
                                    ind = save_close(representation_all_file, representation_all, ind) 
                                    change = 0
                                # if count2%(20*step) == 0:
                                #     print("count2= ", count2)
          
                lf = len(representation_all[method][dataset][model][demo]['fails'])
                lorg = representation_all[method][dataset][model][demo]['num_pics']
                #lo = lorg  - lno
                
                
                representation_all[method][dataset][model][demo]['fail_ratio'] =   round(100.0*lf/lorg, 2)
                print(demo)
                print(representation_all[method][dataset][model][demo]['fail_ratio'])
                if (change):
                    ind = save_close(representation_all_file, representation_all, ind)
     
            
