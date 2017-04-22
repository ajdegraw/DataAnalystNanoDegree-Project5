""" The goal of this file is to get a simple understanding of some data using
univariate plots with matplotlib or the pandas packages."""

import pandas as pd
import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from exploratoryPlots import load_data, make_plots, remove_outliers, write_pkl, transform_features
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


features_list = ['poi',
                 'salary',
                 'to_messages',
                 #'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'shared_receipt_with_poi',
                 #'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 #'loan_advances',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 #'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 'from_poi_to_this_person']



features, labels = load_data(features_list)
features_df = pd.DataFrame(features, columns = features_list[1:]) 
#Transform features
features_df = transform_features(features_df, features_list)
features = features_df.values
#Remove outliers
no_features, no_labels = remove_outliers(features, labels, lo = 0.00, hi = 1.00)
""" I am finding that the POI's are being removed at a higher rate when the ouliers
are removed than the non-POI's are.  So the analysis will be done with the outliers
left in the data as this could be one of the characteristics that is highly correlated
with being labelled POI - whether or not the observation is considered an outlier
in at least one feature. """
no_features_df = pd.DataFrame(no_features, columns = features_list[1:])
########################################################################################
"""Here I am doing a modeling where the outliers have been removed from the dataset.
I am still going to use StratifiedKFold as the labels poi only number 18 out
of 145, so a balanced data sample is needed to train on, else the majority 
classifier will be favored"""


def train_multiple(features, labels):
    """train_multiple: accepts input of a features array or dataframe "features" 
    and the corresponding binary vector of "labels".  
    
    A manual grid search is performed for SVC using stratified cross validation
    
    prints out best classifier based on avg precision and recall scores
    
    returns a list of models with performance measures and parameters that 
    meet a minimum performance threshold. """
    
    
    #Using stratified K fold because there are not many poi's in the dataset
    n_splits = 30

    skf = StratifiedKFold(n_splits = n_splits, random_state = 4, shuffle = False)
    skf.get_n_splits(features, labels)

    ss = StandardScaler()
    pca = PCA()

    features = np.array(features)
    labels = np.array(labels)

    # Good value: C = 21.216
    Cs = [0.5, 1, 10, 20, 50]
    kernels = ['linear', 'sigmoid', 'rbf']
    # Iteratively scale, apply PCA and train SVC before scoring with recall and precision
    all_performances = []
    K = range(3,len(features[0]))
    best_score = 0
    
    for C in Cs:
        for kernel in kernels:
            for k in K:
                sum_precision = 0 
                sum_recall = 0
        
                for train_index, test_index in skf.split(features, labels):
                    #print("Building model")
                    
                    #split the features and labels data with skf
                    features_train, features_test = features[train_index], features[test_index]
                    labels_train, labels_test = labels[train_index], labels[test_index]

                    #standardize the features in features_train and features_test
                    ss.fit(features_train)
                    features_train = ss.transform(features_train)
                    features_test = ss.transform(features_test)

                    #Use PCA 
                    pca.fit(features_train[:,0:k])
                    features_train = pca.transform(features_train[:,0:k])
                    features_test = pca.transform(features_test[:,0:k])
                
                    #train and test the SVC model
                    #clf = SVC(C = C, kernel = kernel)
                    clf = SVC(class_weight = 'balanced', C = C, kernel = kernel)
                    #clf = RandomForestClassifier()
                    #clf = AdaBoostClassifier()
                    #clf = LogisticRegression(penalty = 'l2', class_weight = 'balanced')
                    clf.fit(features_train, labels_train)
                    # predict and score on the testing set for my "accuracy" of the model
                    preds = clf.predict(features_test[:,0:k])
                    recall = recall_score(preds, labels_test)
                    precision = precision_score(preds, labels_test)
                    sum_precision += precision
                    sum_recall += recall

                    # construct a dict from a new dataframe and send to tester.py 
                    # function for evaluation          
                    df = pd.DataFrame(pca.transform(features_train))
                    df.index = range(len(features_train))#, inplace=True)
                    # after you create features, the column names will be your new features
                    # create a list of column names:
                    df['poi'] = list(labels_train)
           
                    df = df[['poi'] + range(0,k)]
                    new_features_list = df.columns.values

                    # create a dictionary from the dataframe
                    df_dict = df.to_dict('index')

                all_performances.append({'clf':clf,
                                     'K_features':k,
                                 'C':C, 
                                 'kernel':kernel, 
                                 'Avg_precision': sum_precision / n_splits,
                                 'Avg_recall': sum_recall / n_splits,
                                 'tester_score': test_classifier(clf, df_dict, new_features_list)})
    
                if (sum_precision + sum_recall) / n_splits > best_score:
                    best_score = (sum_precision + sum_recall) / n_splits
                    best_model = {'clf':clf, 'kernel': kernel, 'C':C, 'k_features': k}

    #The arrays and data frame creation are for the purpose of inspecting performance  
    clf_ = []
    recall_ = []
    prec_ = []
    C_ = []
    K_ = []
    kernel_ = []
    avg_precision_ = []
    avg_recall_ = []

    for ii in range(len(all_performances)):
        avg_precision_ = avg_precision_ + [all_performances[ii]['Avg_precision']]
        avg_recall_ = avg_recall_ + [all_performances[ii]['Avg_recall']]
        recall_ = recall_ + [all_performances[ii]['tester_score']['recall']]
        prec_ = prec_ + [all_performances[ii]['tester_score']['precision']]
        clf_ = clf_ + [all_performances[ii]['clf']]
        kernel_ = kernel_ + [all_performances[ii]['kernel']]
        C_ = C_ + [all_performances[ii]['C']]
        K_ = K_ + [all_performances[ii]['K_features']]
    all_perf_df = pd.DataFrame({'clf':clf_, 'K_features':K_, 'C':C_, 
        'precision':prec_, 'recall': recall_, 'kernel':kernel_,
        'avg_precision':avg_precision_, 'avg_recall':avg_recall_})        
        
    #Skim the cream off the top
    good_models = all_perf_df[(all_perf_df['avg_precision'] >= 0.33) & (all_perf_df['avg_recall'] >= 0.3)]
    print(good_models)
    good_models.to_csv("good_models_transformed_features.csv",header=True, index = False)
    print("Best Average recall and precision score on Stratified CV is {}".format(best_model))
    #return all model performances
    return(all_perf_df)


def train_best_model(features, labels, C, kernel, k_best):
    """
    train_best_model: input - features: array or dataframe
    labels: corresponding class labels
    C, kernel, k_best: tuning parameters to build SVC model
    
    """
    #Using stratified K fold because there are not many poi's in the dataset
    n_splits = 30

    skf = StratifiedKFold(n_splits = n_splits, random_state = 4, shuffle = False)
    skf.get_n_splits(features, labels)

    ss = StandardScaler()
    pca = PCA()

    features = np.array(features)
    labels = np.array(labels)

    sum_precision = 0
    sum_recall = 0
   

    for train_index, test_index in skf.split(features, labels):
                    
        #split the features and labels data with skf
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        
        #standardize the features in features_train and features_test
        ss.fit(features_train)
        features_train = ss.transform(features_train)
        features_test = ss.transform(features_test)

        #Use PCA 
        pca.fit(features_train[:,0:k_best])
        features_train = pca.transform(features_train[:,0:k_best])
        features_test = pca.transform(features_test[:,0:k_best])
                
        #train and test the SVC model
        clf = SVC(class_weight = 'balanced', C = C, kernel = kernel)
        clf.fit(features_train, labels_train)
        #predict and score on the testing set for my "accuracy" of the model
        preds = clf.predict(features_test[:,0:k_best])
        recall = recall_score(preds, labels_test)
        precision = precision_score(preds, labels_test)
        sum_precision += precision
        sum_recall += recall
        # construct a dict from a new dataframe and send to tester.py 
        # function for evaluation          
        df = pd.DataFrame(pca.transform(features_train))
        df.index = range(len(features_train))#, inplace=True)
        # after you create features, the column names will be your new features
        # create a list of column names:
        df['poi'] = list(labels_train)
         
        df = df[['poi'] + range(0,k_best)]
        new_features_list = df.columns.values

        # create a dictionary from the dataframe
        df_dict = df.to_dict('index')

    write_pkl(clf, df)
    recall = sum_recall / n_splits
    precision = sum_precision / n_splits
    return(clf, recall, precision)
    
    
    
    
    
    

    






###ADD new feature that is the total number of negative positions in stock or salary
#features_df['net_negative'] = (features_df['restricted_stock_deferred'] < 0) * 1.0 \
#        + (features_df['restricted_stock'] < 0) * 1.0 + (features_df['deferral_payments'] < 0 ) * 1.0 \
#        + (features_df['total_stock_value'] < 0) * 1.0
#          
#features_df['net_negative_sum'] = features_df['restricted_stock_deferred'] \
#        + features_df['restricted_stock'] + features_df['deferral_payments'] \
#        + features_df['total_stock_value']
    
    
    
#all_perf_df = train_multiple(features, labels)
train_best_model(features = features, labels = labels, C = 20.0, kernel = 'linear', k_best = 14)


         
#features_df = transform_features(features_df, features_list)
#features = features_df.values
#all_perf_df = train_multiple(features, labels)
#train_best_model(features = features, labels = labels, C = 1.0, kernel = 'linear', k_best = 15)        