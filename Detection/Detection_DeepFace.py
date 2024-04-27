
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
parser.add_argument('--detection_methods', type=str, default='mtcnn', help='choose one or more of these methods (should be separetad with ,): mtcnn, dlib, opencv, ssd, retinaface')# Parse the argument
parser.add_argument('--datasets', type=str, default='BFW,DemogPairs', help='list your intended datasets seperated by comma. Note that you cant update the datasets later. You would need to start from scratch for a new dataset')# Parse the argument
parser.add_argument('--step', type=int, default=500, help='after <step> number of successfully processed photos the results are saved')# Parse the argument

# parser.add_argument('--method_file_path', type=str, required=True)# Parse the argument
args = parser.parse_args()
method = args.method
detectors= args.detection_methods
detectors = detectors.replace(' ','').split(',') #add "mtcnn" and "dlib" later

datasets_path=args.datasets_path
datasets= args.datasets.replace(' ', '').split(',')

if len(detectors)>1:
        detection_all_file = 'Detection_' + method + '_.pkl'
else:
    detection_all_file = 'Detection_' + method + '_'+ detectors[0]+'_.pkl'

step = args.step

def all_detection_init():
        detection_all = {}
        detection_all[method] = {}
        for detector in detectors:
            detection_all[method][detector] = {}
            for dataset in datasets:
                detection_all[method][detector][dataset] = {}
                dataset_p = join(datasets_path, method, dataset)
                demos = listdir(dataset_p)
                demos = [d for d in demos if  isdir(join(dataset_p, d))]
                for demo in demos:
                    detection_all[method][detector][dataset][demo] = {}
                    detection_all[method][detector][dataset][demo]['num_pics'] = 0
    
                    detection_all[method][detector][dataset][demo]['num_orgs'] = 0
                    detection_all[method][detector][dataset][demo]['fails']= set()
                    demo_p = join(dataset_p, demo)
                    people = listdir(demo_p)
                    people = [p for p in people if isdir(join(demo_p, p))]
                    for person in people:
                        detection_all[method][detector][dataset][demo][person] = {}
                        person_p = join(demo_p, person)
                        pics = listdir(person_p)
                        pics = [ join(dataset, demo, person, p) for p in pics if 'DS_Store' not in p ]
    
                        detection_all[method][detector][dataset][demo][person]['pics'] = set(pics)
                        detection_all[method][detector][dataset][demo][person]['passed'] = set()
                        detection_all[method][detector][dataset][demo][person]['fails'] = set()
                        org_path = join(datasets_path,'Original', dataset, demo, person)
                        orgs = listdir(org_path)
                        orgs = [ p for p in orgs if 'DS_Store' not in p ]
                        detection_all[method][detector][dataset][demo]['num_orgs'] += len(orgs)
                        if method=='CIAGAN':
                            detection_all[method][detector][dataset][demo]['num_pics'] += len(pics)/2
                        else:
                            detection_all[method][detector][dataset][demo]['num_pics'] += len(pics)
        return detection_all



def check_if_the_file_exist(detection_all_file):
    i = 4
    pickles = listdir('.')
    pickles = [p for p in pickles if ".pkl" in p and detection_all_file[:-i] in p ]
    if (len(pickles)==0):
        detection_all = all_detection_init()
        return detection_all, 1
    last_ind = 0
    for p in pickles:
        pi = int( p[len(detection_all_file[:-i]):-i])
        if pi>last_ind:
           last_ind =pi
           last = p
    print("last_ind =", last_ind)
    flag = 1
    while(flag):
        try:
            with open(last, 'rb') as my_file:
                detection_all = pickle.load(my_file)
                my_file.close()
                flag = 0
                print("loaded_last_ind =", last_ind)
                for k in range(1, last_ind):
                    file = detection_all_file[:-i] + format(k, '06d') + detection_all_file[-i:]
                    if isfile(file):
                        remove(file)
                return detection_all, last_ind+1

        except:
               last = detection_all_file[:-i] + format(last_ind-1, '06d') + detection_all_file[-i:]
               last_ind = last_ind-1
    




def save_close(detection_all_file, detection_all, ind):
    i=4
    file_name = detection_all_file[:-i] + format(ind, '06d') + detection_all_file[-i:]
    with  open(file_name, 'wb') as geeky_file:
        pickle.dump(detection_all, geeky_file)
        geeky_file.close()
        x = ind
        flag = 1
        while (flag):
            try:
                file_name = detection_all_file[:-i] + format(x, '06d') + detection_all_file[-i:]
                if isfile(file_name):
                    with open(file_name, 'rb') as my_file:
                        detection_all = pickle.load(my_file)
                        my_file.close()
                        x = x-1
                        flag = 0
            except:
                flag =1
        for j in range(x+1):
            file_name = detection_all_file[:-i] + format(x, '06d') + detection_all_file[-i:]
            if isfile(file_name):
                remove(file_name)
        return ind+1

count2 = 0
detection_all, ind = check_if_the_file_exist(detection_all_file)
for detector in detectors:
    for dataset in datasets:
        dataset_p = join(args.datasets_path, dataset)
        for demo in detection_all[method][detector][dataset].keys():
            print(detector, dataset, demo)
           

            count = 0
            change = 0 
            for person in detection_all[method][detector][dataset][demo].keys():
                if person not in ["num_pics", "fails", "fail_ratio", 'num_orgs', 'miss_ratio' ]:
                    for pic in  detection_all[method][detector][dataset][demo][person]['pics']:
                       pic_p=join(datasets_path, method, pic)
                       # print(pic_p)
                       if pic not in detection_all[method][detector][dataset][demo][person]['passed']:
                            
                            if pic not in detection_all[method][detector][dataset][demo][person]['fails']:
                                
                                try:

                                     img = DeepFace.extract_faces(pic_p, detector_backend=detector)

                                     detection_all[method][detector][dataset][demo][person]['passed'].add(pic)
                                     change = 1
                                     count+=1


           
                                except Exception as e:
                                     detection_all[method][detector][dataset][demo][person]['fails'].add(pic)
                                     detection_all[method][detector][dataset][demo]['fails'].add(pic)
                                
                            else:
                                
                                try:
                                     img = DeepFace.extract_faces(pic_p, detector_backend=detector)
                                     detection_all[method][detector][dataset][demo][person]['passed'].add(pic)
                                     detection_all[method][detector][dataset][demo][person]['fails'].remove(pic)
                                     detection_all[method][detector][dataset][demo]['fails'].remove(pic)
                                     count += 1
                                     change = 1
                                except Exception as e:
                                    count2 += 1
                          
                            if count%step == 0 and change == 1:
                                ind = save_close(detection_all_file, detection_all, ind) 
                                change = 0
                            # if count2%(20*step) == 0:
                            #     print("count2= ", count2)

      
            lf = len(detection_all[method][detector][dataset][demo]['fails'])
            l_pics = detection_all[method][detector][dataset][demo]['num_pics']
            # print(lf, l_pics)

            # # l_org = detection_all[method][detector][dataset][demo]['num_orgs']
            
            detection_all[method][detector][dataset][demo]['fail_ratio'] =   round(100.0*lf/l_pics, 2)
            print(demo)
            print('failure rate', detection_all[method][detector][dataset][demo]['fail_ratio'])

            if method != 'Original':
                l_obf = detection_all[method][detector][dataset][demo]['num_pics']
                l_org = detection_all[method][detector][dataset][demo]['num_orgs']

                detection_all[method][detector][dataset][demo]['miss_ratio'] = round(100.0*(1-l_obf/l_org), 2)
                print('missing rate = ', detection_all[method][detector][dataset][demo]['miss_ratio'] )
            
            if (change):
                ind = save_close(detection_all_file, detection_all, ind)

#