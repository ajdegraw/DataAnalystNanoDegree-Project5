""" The goal of this file is to get a simple understanding of some data using
univariate plots with matplotlib or the pandas packages."""

import pandas as pd
import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


def load_data(features_list = ['poi',
                     'salary',
                     'to_messages',
                     'deferral_payments',
                     'total_payments',
                     'exercised_stock_options',
                     'bonus',
                     'restricted_stock',
                     'shared_receipt_with_poi',
                     'restricted_stock_deferred',
                     'total_stock_value',
                     'expenses',
                     'loan_advances',
                     'from_messages',
                     'other',
                     'from_this_person_to_poi',
                     'director_fees',
                     'deferred_income',
                     'long_term_incentive',
                     'from_poi_to_this_person']):
    #Loads data from supplied pickle file and returns arrays of features, labels

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        
    #pop TOTAL row so that it is not used in training or testing
    data_dict.pop('TOTAL') 
                            
    ### Extract features and labels from dataset for local testing
    data = featureFormat(data_dict, features_list, sort_keys = True)

    labels, features = targetFeatureSplit(data)
    return features, labels

def make_plots(features_df, features_list):
    import matplotlib.pyplot as plt
    for feat in features_list[1:]:
        fig, (ax1, ax2) = plt.subplots(2)
    
        ax1.boxplot(features_df[feat], showbox = True)
        #ax1.title(feat)
        
        ax2.hist(features_df[feat], bins = 30)
        #ax2.title(feat)
        plt.suptitle(feat)
        plt.show()

def count_missing_values():
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        
    #pop TOTAL row so that it is not used in training or testing
    data_dict.pop('TOTAL') 
    missing_data = {'salary':0,
                    'to_messages':0,
                    'deferral_payments':0,
                    'total_payments':0,
                    'exercised_stock_options':0,
                    'bonus':0,
                    'restricted_stock':0,
                    'shared_receipt_with_poi':0,
                    'restricted_stock_deferred':0,
                    'total_stock_value':0,
                    'expenses':0,
                    'loan_advances':0,
                    'from_messages':0,
                    'other':0,
                    'from_this_person_to_poi':0,
                    'poi':0,
                    'director_fees':0,
                    'deferred_income':0,
                    'long_term_incentive':0,
                    'email_address':0,
                    'from_poi_to_this_person':0}
    
    for key in missing_data:
        for name in data_dict.keys():
            if data_dict[name][key] == 'NaN':
                missing_data[key] += 1
    return(missing_data)

def remove_outliers(features, labels, lo = 0.05, hi = 0.95):
    """Going to remove the outliers in each feature:
    features: scaled or unscaled features with observations arranged in rows
    lo: the percentile for cut-off of outlier condition
    hi: the percentile for cut-off of outlier condition"""


    features = np.array(features)
    labels = np.array(labels)
    non_outlier_idx = range(len(features))
    percentile_hi = np.percentile(features, hi * 100, axis = 0)
    percentile_lo = np.percentile(features, lo * 100, axis = 0)
    for ii in range(len(features[0])):
        for jj in range(len(features)):
            if ((features[jj][ii] < percentile_lo[ii]) or (features[jj][ii] > percentile_hi[ii])):
                non_outlier_idx[jj] = -1
    
    non_outlier_idx = [ii for ii in non_outlier_idx if ii != -1]
    n_o_features = features[non_outlier_idx]
    n_o_labels = labels[non_outlier_idx]
    return n_o_features, n_o_labels


def write_pkl(clf, features_labels_df):#, labels):
    #add lables as poi column to dataframe that will be converted to dict and
    # written to pkl file
    #df = features_df
    #df['poi'] = list(labels)
    #df = df[['poi',0,1,2,3,4,5,6,7,8]]#9,10,11,12,13,14,15,16,17,18]]

    #print(df.columns.values)
    #print(sum(df['poi']))
    #print(new_features_list)
    # create a dictionary from the dataframe
    df = features_labels_df
    new_features_list = df.columns.values
    df_dict = df.to_dict('index')

    #write dictionary to pkl file using provided function
    dump_classifier_and_data(clf, df_dict, new_features_list)
    return df_dict




features_list = ['poi',
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'shared_receipt_with_poi',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 'from_poi_to_this_person']





###################################################
""" A different direction to go in is to see if I can transform the features
using log or sqrt or other function so that the outliers are less extreme when
compared to the bulk of the data.  I will overwrite the current dataframe as 
the original data can be easily reloaded with the function I build.  Some features
it could be important to preserve the negative values as for something like
restricted_stock this could mean that the person has a short position in the stock,
that is they benefit from a decrease in stock value.  This could be important"""

def transform_features(features_df, features_list):
    #transform and look at plots again: log transform
    features_df['deferred_income'] = np.abs(features_df['deferred_income'])
    #features_df.fillna(features_df.median())
    for feat in features_list:
        if feat not in ['poi','restricted_stock_deferred','restricted_stock','deferral_payments', 'total_stock_value']:
            try:
                features_df[feat] = features_df[feat]
                features_df[feat] = np.log10(features_df[feat] + 1)
            except:
                continue
        else:
            continue
    return features_df
    



if __name__ == "main":
    
    
    features_list = ['poi',
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'shared_receipt_with_poi',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 'from_poi_to_this_person']    
    
    
    features, labels = load_data()
    features_df = pd.DataFrame(features, columns = features_list[1:]) 
    print("With outliers:")
    make_plots(features_df, features_list[1:])
    """I am finding that most of the features have some extreme outliers that may need
    to be removed from the data before analysis so that the algorithm is more robust.
    Note: Several of those that benefited the most from the scandal were financial outliers"""
    ################################################    
    """These feature transformations do not depend on the actual values that could end
    up in the train vs. test sets, so the entire data set is transformed
    prior to any training train/test splitting or Kfold splitting"""
    features_df = transform_features(features_df, features_list)
    print("Transformed Features.")
    ################################################
    print("With outliers:")
    make_plots(features_df, features_list[1:])
    ####################################################
    #remove outliers and re-analyze
    ####################################################
    no_features, no_labels = remove_outliers(features, labels, lo = 0.05, hi = 0.95)
    """ I am finding that the POI's are being removed at a higher rate when the ouliers
    are removed than the non-POI's are.  So the analysis will be done with the outliers
    left in the data as this could be one of the characteristics that is highly correlated
    with being labelled POI - whether or not the observation is considered an outlier
    in at least one feature. """
    no_features_df = pd.DataFrame(no_features, columns = features_list[1:])
    # and replot
    print("Without Outliers:")
    make_plots(no_features_df, features_list[1:])
    ###################################################
    ###################################################
    """ A different direction to go in is to see if I can transform the features
    using log or sqrt or other function so that the outliers are less extreme when
    compared to the bulk of the data.  I will overwrite the current dataframe as 
    the original data can be easily reloaded with the load_data.  
    
    Note:In some features
    it could be important to preserve the negative values as for something like
    restricted_stock this could mean that the person has a short position in the stock,
    that is they benefit from a decrease in stock value.  This could be important"""