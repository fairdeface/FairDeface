
import pickle
from os.path import join,  isfile, basename
from os import listdir, remove
import os
from random import sample

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse


parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('--method', type=str, required=True, help='Obfuscation method or Original as in directory names in datasets directory')# Parse the argument
parser.add_argument('--obfuscation_file', type=str,  help='path to the identification file')# Parse the argument
parser.add_argument('--original_file', type=str, required=True, help='path to the original file')# Parse the argument
parser.add_argument('--train_test_ratio', type=str, required=False, help='path to the original file', default=.7)# Parse the argument
parser.add_argument('--scenarios', type=str, required=False, help='choose 1 when training each demographic seperately, or 2 when training all demographics together, or', default='1,2')# Parse the argument

args = parser.parse_args()

method = args.method
train_test_ratio=args.train_test_ratio
original_file = args.original_file
scenarios=args.scenarios.split(',')
if method!='Original':
    obfuscation_file=args.obfuscation_file
else:
    obfuscation_file=args.original_file


# method= 'DP1'
# original_file='Identification_Original_000026.pkl'
# obfuscation_file='Identification_DP1_000048.pkl'
# train_test_ratio=.7
# scenarios=['1', '2']

scores_all_file = "Identification_Scores_" + method + '_.pkl'
def all_score_init():    
    scores_all = {}

    scores_all[method] = {}
    for dataset in representation_all[method].keys():
        scores_all[method][dataset] = {}

        for model in representation_all[method][dataset].keys():
            scores_all[method][dataset][model] = {}
      
            scores_all[method][dataset][model]['done'] = 0
            for demo in representation_all[method][dataset][model].keys():
                scores_all[method][dataset][model][demo] = {}
                scores_all[method][dataset][model][demo]['all'] = {}
                scores_all[method][dataset][model][demo]['each'] = {}

    return scores_all



    # return scores_all
def check_if_the_file_exist(scores_all_file):
    i = 4
    pickles = listdir('.')
    pickles = [p for p in pickles if ".pkl" in p and scores_all_file[:-i] in p ]
    if (len(pickles)==0):
        scores_all = all_score_init()
        return scores_all, 1
    last_ind = 0
    for p in pickles:
        pi = int( p[len(scores_all_file[:-i]):-i])
        if pi>last_ind:
            last_ind =pi
            last = p
    print("last_ind =", last_ind)
    flag = 1
    while(flag):
        try:
            with open(last, 'rb') as my_file:
                scores_all = pickle.load(my_file)
                my_file.close()
                flag = 0
                print("loaded_last_ind =", last_ind)
                for k in range(1, last_ind):
                    file = scores_all_file[:-i] + format(k, '06d') + scores_all_file[-i:]
                    if isfile(file):
                        remove(file)
                return scores_all, last_ind+1

        except:
                remove(last)
                last = scores_all_file[:-i] + format(last_ind-1, '06d') + scores_all_file[-i:]
                last_ind = last_ind-1
    



      
def load_results(resultsf):

        file_to_read = open(resultsf, "rb")
        results = pickle.load(file_to_read)
        file_to_read.close()
        return results


def save_close(scores_all_file, scores_all, ind):
    i=4
    file_name = scores_all_file[:-i] + format(ind, '06d') + scores_all_file[-i:]
    with  open(file_name, 'wb') as geeky_file:
        pickle.dump(scores_all, geeky_file)
        geeky_file.close()
        x = ind
        flag = 1
        while (flag):
            try:
                file_name = scores_all_file[:-i] + format(x, '06d') + scores_all_file[-i:]
                if isfile(file_name):
                    with open(file_name, 'rb') as my_file:
                        scores_all = pickle.load(my_file)
                        my_file.close()
                        x = x-1
                        flag = 0
            except:
                flag =1
        for j in range(x+1):
            file_name = scores_all_file[:-i] + format(x, '06d') + scores_all_file[-i:]
            if isfile(file_name):
                remove(file_name)
        return ind+1


def SVM_Method_Dataset(method, dataset, model, scores, ind, representation_all_people_train_test,  representation_demo_based_train_test, scenarios):
    # representation_all_people_file = "representation_all_people_" + method + "_" + dataset + ".pkl"
    # representation_demo_based_file = "representation_demo_based_" + method + "_" + dataset + ".pkl"
    for scenario in scenarios:

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        # label encode targets
        out_encoder = LabelEncoder()
        model_svm = SVC(kernel='linear', probability=True)
        
        if scenario=='1':
            print("each demographic trained and tested for each demographic")
            demos = set(representation_demo_based_train_test.keys())- {'trainx', 'trainy'}

            for demo in demos:
                trainX_all = representation_demo_based_train_test[demo]['trainx']
                trainy_all = representation_demo_based_train_test[demo]['trainy']
                # trainX_all = in_encoder.transform(trainX_all)
                # out_encoder.fit(trainy_all)
                # trainy_all = out_encoder.transform(trainy_all)
                model_svm.fit(trainX_all, trainy_all)
                # predict
                yhat_train = model_svm.predict(trainX_all)
                # score
                score_train = accuracy_score(trainy_all, yhat_train)
                print('Accuracy for ' + model + ':  train=%.3f' % ( score_train*100))
                scores[method][dataset][model][demo]['each']['train'] = round(score_train*100, 1)


                yhat_test = model_svm.predict(representation_demo_based_train_test[demo]['testx'])
                # score 
                
                score_test = accuracy_score(representation_demo_based_train_test[demo]['testy'], yhat_test)
                # summarize
                print('Accuracy for ' + demo + ':  test=%.3f' % ( score_test*100))
                scores[method][dataset][model][demo]['each']['test'] = round(score_test*100, 1)
        if scenario=='2':
            print("all peaple trained, tested for each demographic")
            trainX_all = representation_all_people_train_test['trainx']
            trainy_all = representation_all_people_train_test['trainy']
            # trainX_all = in_encoder.transform(trainX_all)
            # out_encoder.fit(trainy_all)
            # trainy_all = out_encoder.transform(trainy_all)
            model_svm.fit(trainX_all, trainy_all)
            # predict
            yhat_train = model_svm.predict(trainX_all)
            # score
            score_train = accuracy_score(trainy_all, yhat_train)
            print('Accuracy for ' + model + ':  train=%.3f' % ( score_train*100))
        
            demos = set(representation_all_people_train_test.keys())- {'trainx', 'trainy'}
            for demo in demos:
                yhat_test = model_svm.predict(representation_all_people_train_test[demo]['testx'])
                # score
                scores[method][dataset][model][demo]['all']['train'] = round(score_train*100, 1)
        
                score_test = accuracy_score(representation_all_people_train_test[demo]['testy'], yhat_test)
                # summarize
                print('Accuracy for ' + demo + ':  test=%.3f' % ( score_test*100))
                scores[method][dataset][model][demo]['all']['test'] = round(score_test*100, 1)
                ind = save_close(scores_all_file, scores, ind)

    return scores, ind

count2 = 0
embed_means={}
embed_stds = {}
embed_means_mean={}
embed_stds_mean = {}
with open(obfuscation_file, 'rb') as my_file:
    representation_all = pickle.load(my_file)
    my_file.close()
    
with open(original_file, 'rb') as my_file:
    representation_all_org = pickle.load(my_file)
    my_file.close()

representation_all_people = {}
representation_demo_based = {}

scores, ind = check_if_the_file_exist(scores_all_file)
for dataset in scores[method].keys():
    for model in scores[method][dataset].keys():
            if scores[method][dataset][model]['done']==0 and model in representation_all_org['Original'][dataset].keys():
                
                representation_all_people= {}
                representation_demo_based= {}
                # dataset_p = join('dataset', dataset)
                for demo in representation_all[method][dataset][model].keys():
                    representation_demo_based[demo] = {}

                    for person in representation_all[method][dataset][model][demo].keys():

                        if person not in ["num_pics", "fails", "fail_ratio"]:
                            representation_all_people[person] = {}
                            representation_demo_based[demo][person]={}
                            for pic in  representation_all[method][dataset][model][demo][person]['passed']:
                                embeddings = representation_all[method][dataset][model][demo][person]['representations'][pic]

                                if len(embeddings)==1:
        
                                    embedding = embeddings[0]['embedding']
                                    representation_all_people[person][pic] = embedding
                                    representation_demo_based[demo][person][pic] = embedding


        
                representation_all_people_train_test = {}
                representation_all_people_train_test['trainx'] = []
                representation_all_people_train_test['trainy'] = []
                representation_demo_based_train_test= {}

                for demo in representation_all[method][dataset][model].keys():
                    representation_all_people_train_test[demo] = {}
                    representation_all_people_train_test[demo]['testx'] = []
                    representation_all_people_train_test[demo]['testy'] = []
                    obf_demo=representation_all[method][dataset][model][demo]
                    if method!='CIAGAN':
                        org_demo=representation_all_org['Original'][dataset][model][demo]
   
            
                    for person in obf_demo.keys(): 
                        if type(obf_demo[person])==dict:
                            reps = obf_demo[person]['representations']
                            if method!='CIAGAN':
                                reps_org = org_demo[person]['representations']
            
                            else:
                                to_be_deleted=set()
                                reps_org={}
                                for pic in reps.keys():
                                    if 'org' in pic:
                                        reps_org[pic]= reps[pic]
                                        to_be_deleted.add(pic)
                                for pic in to_be_deleted:
                                    del reps[pic]
                            ylabel = [join(demo, person)]
                            
                            total_num = len(reps)
                            
                                        
                            train_num = round(total_num*train_test_ratio)
                         
                            allkeys = sorted(reps)
                            train_pics = sample(allkeys, train_num)
                            test_pics = list(set(allkeys) - set(train_pics))
                            if len(test_pics)>0:
                                # allkeys_org = sorted(reps_org)
                                # dirn= os.path.dirname(allkeys_org[0])
                                # org_dic ={}
                                # for p in allkeys:
                                #     org_dic[p]= join(dirn, basename(p)[:-4]+ allkeys_org[0][-4:])
                
                                for pic in train_pics:
                                    representation_all_people_train_test['trainx'].append(reps[pic][0]['embedding'])
                                    representation_all_people_train_test['trainy'].append(ylabel)
                                    if method!='Fawkes-High' and method!='Original':
                                        if method=='CIAGAN':
                                            pic=pic.replace('obf', 'org')
                                        
                                        if pic in reps_org.keys():

                                            representation_all_people_train_test['trainx'].append(reps_org[pic][0]['embedding'])
                                            representation_all_people_train_test['trainy'].append(ylabel)
                                  
                             
                                if method=='Fawkes-High':
                                    for pic in test_pics:
                                        if pic in reps_org.keys():
                                            
                                            representation_all_people_train_test[demo]['testx'].append(reps_org[pic][0]['embedding'])
                                            representation_all_people_train_test[demo]['testy'].append(ylabel)
                                else:
                                     
                                    for pic in test_pics:
                                         representation_all_people_train_test[demo]['testx'].append(reps[pic][0]['embedding'])
                                         representation_all_people_train_test[demo]['testy'].append(ylabel)
                     

                      
                    representation_demo_based_train_test[demo] = {}
                    representation_demo_based_train_test[demo]['trainx'] = []
                    representation_demo_based_train_test[demo]['trainy'] = []
                    representation_demo_based_train_test[demo]['testx'] = []
                    representation_demo_based_train_test[demo]['testy'] = []
                    for person in obf_demo.keys(): 
                        if type(obf_demo[person])==dict:
                            
                            reps = obf_demo[person]['representations']
                            if method!='CIAGAN':
                                reps_org = org_demo[person]['representations']
                            else:
                                reps_org={}
                                for pic in reps.keys():
                                    if 'org' in pic:
                                        reps_org[pic]= reps[pic]
                                        del reps[pic]
                            
                            ylabel = [join(demo, person)]
                            
                            total_num = len(reps)
                            train_num = round(total_num*train_test_ratio)
    
    
                            allkeys = sorted(reps)
                            train_pics = sample(allkeys, train_num)
                            test_pics = list(set(allkeys) - set(train_pics))
                            if len(test_pics)>0:

                                for pic in train_pics:
                                    representation_demo_based_train_test[demo]['trainx'].append(reps[pic][0]['embedding'])
                                    representation_demo_based_train_test[demo]['trainy'].append(ylabel)
                                   
                                    if method!='Fawkes-High' and method!='Original':
                                        if method=='CIAGAN':
                                            pic=pic.replace('obf', 'org')
                                        if pic in reps_org.keys():
                                            representation_demo_based_train_test[demo]['trainx'].append(reps_org[pic][0]['embedding'])
                                            representation_demo_based_train_test[demo]['trainy'].append(ylabel)

    
        
                                if method=='Fawkes-High':
                                    for pic in test_pics:
                                        if pic in reps_org.keys():

                                            representation_demo_based_train_test[demo]['testx'].append(reps_org[pic][0]['embedding'])
                                            representation_demo_based_train_test[demo]['testy'].append(ylabel)
                                else:
                                    for pic in test_pics:
                                        representation_demo_based_train_test[demo]['testx'].append(reps[pic][0]['embedding'])
                                        representation_demo_based_train_test[demo]['testy'].append(ylabel)
        
                            
                           
  
                scores, ind = SVM_Method_Dataset(method, dataset, model, scores, ind,  representation_all_people_train_test,  representation_demo_based_train_test, scenarios )
               
                
               
                
               
                scores[method][dataset][model]['done']=1
                ind=save_close(scores_all_file, scores, ind)


