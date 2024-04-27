
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
parser.add_argument('--method', type=str, required=True, help='Obfuscation method or Original as in directory names in datasets directory')# Parse the argument
parser.add_argument('--same_photo', type=str, help='should be yes or no. for the Original choice, use --same_identity', required=False)# Parse the argument

parser.add_argument('--datasets_path', type=str,  default='../datasets')# Parse the argument
parser.add_argument('--models', type=str, default='ArcFace', help='choose one or more of these methods (should be separetad with ,): DeepFace , VGG-Face, Facenet, OpenFace, DeepID, Dlib, ArcFace)# Parse the argument')
parser.add_argument('--detector', type=str, default='mtcnn', help='choose one of these methods: mtcnn, dlib, opencv, ssd, retinaface')# Parse the argument
parser.add_argument('--same_identity', type=str, help='This is for no obfuscation choice(method==Original) to calculate TPRs or TFRs. should be yes(TPRs) or no(TFRs)', required=False)# Parse the argument
parser.add_argument('--step', type=int, default=500, help='after <step> number of successfully processed photos the results are saved')# Parse the argument
parser.add_argument('--datasets', type=str, default='BFW,DemogPairs', help='list your intended datasets seperated by comma. Note that you cant update the datasets later. You would need to start from scratch for a new dataset')# Parse the argument

# parser.add_argument('--method_file_path', type=str, required=True)# Parse the argument
args = parser.parse_args()
method = args.method
models= args.models
models = models.split(',') #add "mtcnn" and "dlib" later

datasets_path=args.datasets_path

datasets= args.datasets.replace(' ', '').split(',')

chs=['same-photo', 'different-photo', 'same-identity',  'different-identity']
if method!='Original':
    original_p=join(args.datasets_path, 'Original')
    if args.same_photo=='yes':
        ch =chs[0]
    else:
        ch=chs[1]
else:
    if args.same_identity=='yes':
        ch =chs[2]
    else:
        ch=chs[3]

# verification_all_file = 'Verification_' + method + "_" + ch +'_.pkl'

# if len(models)>1:
verification_all_file = 'Verification_' + method + "_" + ch +'_.pkl'
# else:
#     verification_all_file = 'Verification_' + method+ "_" + ch  + '_'+ models[0]+'_.pkl'

step = args.step

# face verification verification = DeepFace.verify('img1.jpg', 'img2.jpg', model_name = models[1])


def all_verification_init():
        verification_all = {}
        verification_all[method] = {}
        for model in models:
            verification_all[method][model] = {}
            for dataset in datasets:
                verification_all[method][model][dataset] = {}
                dataset_p = join(datasets_path, method, dataset)
                demos = listdir(dataset_p)
                demos = [d for d in demos if  isdir(join(dataset_p, d))]
                for demo in demos:
                    verification_all[method][model][dataset][demo] = {}
                    verification_all[method][model][dataset][demo]['num_pics'] = 0
    
                    verification_all[method][model][dataset][demo]['num_orgs'] = 0
                    verification_all[method][model][dataset][demo]['fails']= set()
                    demo_p = join(dataset_p, demo)
                    people = listdir(demo_p)
                    people = [p for p in people if isdir(join(demo_p, p))]
                    for person in people:
                        verification_all[method][model][dataset][demo][person] = {}
                        person_p = join(demo_p, person)
                        pics = listdir(person_p)
                        pics = [ join(dataset, demo, person, p) for p in pics if 'DS_Store' not in p ]
    
                        verification_all[method][model][dataset][demo][person]['pics'] = set(pics)
                        verification_all[method][model][dataset][demo][person]['passed'] = set()
                        verification_all[method][model][dataset][demo][person]['fails'] = set()
                        verification_all[method][model][dataset][demo][person]['distance'] = []

                        org_path = join(datasets_path,'Original', dataset, demo, person)
                        orgs = listdir(org_path)
                        orgs = [ p for p in orgs if 'DS_Store' not in p ]
                        verification_all[method][model][dataset][demo]['num_orgs'] += len(orgs)
                        if method=='CIAGAN':
                            pics = [ p for p in pics if 'org' not in p ]

                        verification_all[method][model][dataset][demo]['num_pics'] += len(pics)
                     
        return verification_all

def clean_folder(folder):
    for root, dirs, files in walk(folder):
        for file in files:
            if 'DS_Store' in file:
                remove(join(root, file))

        for folder in dirs:
            if 'DS_Store' in folder:
                rmtree(join(root, folder))


def check_if_the_file_exist(verification_all_file):
    i = 4
    pickles = listdir('.')
    pickles = [p for p in pickles if ".pkl" in p and verification_all_file[:-i] in p ]
    if (len(pickles)==0):
        verification_all = all_verification_init()
        return verification_all, 1
    last_ind = 0
    for p in pickles:
        pi = int( p[len(verification_all_file[:-i]):-i])
        if pi>last_ind:
           last_ind =pi
           last = p
    print("last_ind =", last_ind)
    flag = 1
    while(flag):
        try:
            with open(last, 'rb') as my_file:
                verification_all = pickle.load(my_file)
                my_file.close()
                flag = 0
                print("loaded_last_ind =", last_ind)
                for k in range(1, last_ind):
                    file = verification_all_file[:-i] + format(k, '06d') + verification_all_file[-i:]
                    if isfile(file):
                        remove(file)
                return verification_all, last_ind+1

        except:
               remove(last)
               last = verification_all_file[:-i] + format(last_ind-1, '06d') + verification_all_file[-i:]
               last_ind = last_ind-1
    




def save_close(verification_all_file, verification_all, ind):
    i=4
    file_name = verification_all_file[:-i] + format(ind, '06d') + verification_all_file[-i:]
    with  open(file_name, 'wb') as geeky_file:
        pickle.dump(verification_all, geeky_file)
        geeky_file.close()
        x = ind
        flag = 1
        while (flag):
            try:
                file_name = verification_all_file[:-i] + format(x, '06d') + verification_all_file[-i:]
                if isfile(file_name):
                    with open(file_name, 'rb') as my_file:
                        verification_all = pickle.load(my_file)
                        my_file.close()
                        x = x-1
                        flag = 0
            except:
                flag =1
        for j in range(x+1):
            file_name = verification_all_file[:-i] + format(x, '06d') + verification_all_file[-i:]
            if isfile(file_name):
                remove(file_name)
        return ind+1


verification_all, ind = check_if_the_file_exist(verification_all_file)
for model in models:
    for dataset in datasets:
        if method!='Original':
            original_d = join(original_p, dataset)
        dataset_p = join(datasets_path, method, dataset)
        for demo in verification_all[method][model][dataset].keys():
            # verification_all[method][model][dataset][demo]['no_obf_all'] = list(set(verification_all[method][model][dataset][demo]['no_obf_all']))
            print(model, dataset, demo)

               
            count = 0
            change = 0 
            for person in verification_all[method][model][dataset][demo].keys():
                if person not in ["num_pics", "fails", "fail_ratio", 'num_orgs', 'miss_ratio']:
                    for pic in  verification_all[method][model][dataset][demo][person]['pics']:
                        pic_p = join(datasets_path, method, pic)
                        orgs_p= join(datasets_path, 'Original', dataset, demo, person)
                        if method!='Original':
                            if ch ==chs[0]:#same-photo

                                if method == 'CIAGAN':
                                    pic2=pic_p.replace('obf', 'org')
                     
                                else: 
                                  pic2= join(orgs_p, basename(pic))
             
                            elif ch ==chs[1]:
                                if method != 'CIAGAN':
                                   orgs =  listdir(orgs_p)
                                   if len(orgs)>1:
                                       orgs = [join(orgs_p, org) for org in orgs if org!=basename(pic) ]
                                       pic2=choice(orgs)
                                else:
                                    pic_org=pic_p.replace('obf', 'org')
                                    obfs=verification_all[method][model][dataset][demo][person]['pics'] 
                                    orgs=[]
                                    for obf in obfs:
                                        if obf!=pic:
                                            orgs.append(obf.replace('obf', 'org'))
                                    if len(orgs)>1:

                                        pic_chosen=choice(orgs)  
                                        pic2=join(datasets_path, method,pic_chosen)

                        else: #Face Recognition
                            demo_p=join(datasets_path, method, dataset, demo)
                            if ch==chs[2]:#same-identity
                                orgs_p= join(demo_p, person)
                                orgs =  listdir(orgs_p)
                                if len(orgs)>1:
                                    orgs = [join(orgs_p, org) for org in orgs if org!=basename(pic) ]
                                    pic2=choice(orgs)
                            elif ch==ch[3]:
                                people = listdir(demo_p)
                                people= [p for p in people if isdir(join(demo_p, p)) and p!=person]
                                not_success=1
                                while(not_success):
                                    person2=choice(people)
                                    person2_p= join(demo_p, person2)
                                    p2_pics= listdir(person2_p)
                                    if len(p2_pics)>0:
                                        not_success=0
                                        pic2=choice(p2_pics)
                                        pic2=join(person2_p, pic2)

                        
                        if pic not in verification_all[method][model][dataset][demo][person]['passed']:
                            if pic not in verification_all[method][model][dataset][demo][person]['fails'] or pic not in verification_all[method][model][dataset][demo]['fails']:
                                    # print(pic)
                                try:
                                     verification = DeepFace.verify( pic_p, pic2, model_name=model, detector_backend=args.detector)
                                     # print("yes first")
      
                                     # print("passed first:", pic)
                                     verification_all[method][model][dataset][demo][person]['distance'].append(verification)
                                     verification_all[method][model][dataset][demo][person]['passed'].add(pic)
                                     count+=1
                                     change=1
           
                                except Exception as e:
                                     # print('no first')
                                     verification_all[method][model][dataset][demo][person]['fails'].add(pic)
                                     verification_all[method][model][dataset][demo]['fails'].add(pic)
                                     # print(e)
                    
                            else:
     
                                 try:
                                     verification = DeepFace.verify( pic_p, pic2, model_name=model, detector_backend=args.detector)
                                     
                                     
                                     verification_all[method][model][dataset][demo]['fails'].remove(pic)
                                     
                                     verification_all[method][model][dataset][demo][person]['fails'].remove(pic)
                                     # print("yes second")
         
                                     verification_all[method][model][dataset][demo][person]['distance'].append(verification)
                                     verification_all[method][model][dataset][demo][person]['passed'].add(pic)
                                     
                                     count += 1
                                     change = 1
         
           
                                 except Exception as e:
                                     # print('no second')
                                    # if  dataset=="CelebA":
                                       # print(e) 
                                       x = 1
                            if count%step == 0 and change == 1:
                             ind = save_close(verification_all_file, verification_all, ind) 
                             change = 0
      
            lf = len(verification_all[method][model][dataset][demo]['fails'])
            l_org = verification_all[method][model][dataset][demo]['num_orgs']
            l_pic = verification_all[method][model][dataset][demo]['num_pics']

            verification_all[method][model][dataset][demo]['fail_ratio'] =   round(100.0*lf/l_pic, 2)
            print(demo)
            print('failure rate', verification_all[method][model][dataset][demo]['fail_ratio'])

            if method != 'Original':
                verification_all[method][model][dataset][demo]['miss_ratio'] = round(100.0*(l_org-l_pic)/l_org, 2)
                print('missing rate = ', verification_all[method][model][dataset][demo]['miss_ratio'] )
            if (change):
                ind = save_close(verification_all_file, verification_all, ind) 
#