#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:40:41 2022

@author: sebastianquintero
"""

#  Harmonized Crop Data
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sn
import string
from treelib import Node, Tree

from descriptiveAnalysis import *
import nltk

from nltk.stem import WordNetLemmatizer 






# Crop ID Metrics
harmonizedSystem = pd.read_csv('hs6_2017_list.csv')
# Earth Stat FAO Data
cropData = pd.read_csv("earthstat_crop_list.csv")


# This set contains every category of crops from the harmonized data set, specifically the description preceding the semicolon in the third column

categories = {'Degras', 'Cereal', 'Seeds of herbaceous plants', 'Fruit, palm hearts', 'Animal fats and oils and their fractions', 
 'Chocolate and other food preparations containing cocoa', 'Vegetable preparations', 'Sugar confectionery', 'Malt', 'Sugar cane', 'Pig fat', 
 'Cereal straw and husks', 'Nuts, edible', 'Flours and meals', 'Cereal groats and meal', 'Wheat gluten', 'Nuts', 'Cocoa', 'Flakes, granules and pellets', 
 'Prepared foods obtained by the swelling or roasting of cereals or cereal products (e.g. corn flakes)', 'Vegetables, alliaceous', 
 'Vegetable saps and extracts', 'Flowers, cut', 'Vegetables, brassica', 'Pectic substances', 'Fruit', 
 'Vegetable materials of a kind used primarily for plaiting', 'Soya beans', 'Fats of bovine animals, sheep or goats', 
 'Ground-nuts', 'Hop cones', 'Waxes, other than vegetable, n.e.c. in heading no. 1521', 'Jams, jellies, marmalades, purees and pastes', 
 'Poppy straw', 'Locust beans (carob)', 'Vegetable fats and oils and their fractions', 'Fruit, nuts and other edible parts of plants', 
 'Cocoa beans', 'Starch', 'Jams, fruit jellies, marmalades, fruit or nut puree and fruit or nut pastes', 
 'Flours and meals of oil seeds or oleaginous fruits', 'Cereals', 'Oil seeds and oleaginous fruits', 'Nuts and other seeds', 
 'Foliage, branches and other parts of plants, without flowers or flower buds, and grasses, mosses and lichens', 
 'Animal or vegetable fats and oils and their fractions', 'Glycerol, crude', 'Seed', 'Vegetables and mixed vegetables', 
 'Plants, live', 'Food preparations', 'Vegetable mixtures', 'Peel', 'Seaweeds and other algae', 'Poultry fat', 'Vegetables, leguminous', 
 'Coffee', 'Tea, black', 'Spices', 'Jams, fruit jellies, marmalades, purees and pastes', 'Seeds', 'Margarine', 
 'Vegetable waxes (other than triglycerides)', 'Oil seeds', 'Mucilages and thickeners', 'Lard stearin, lard oil, oleostearin, oleo-oil and tallow oil',
 'Sugar beet', 'Vegetables, root', 'Vegetable oils', 'Vegetables', 'Fruit, edible', 'Vegetable products', 'Cereal flour', 
 'Vegetable roots and tubers', 'Flour, meal and powder', 'Fats and oils and their fractions', 'Cereal grains', 'Tallow', 'Oils of fish', 'Coca leaf', 
 'Seeds of forage plants', 'Juice', 'Juices', 'Chocolate & other food preparations containing cocoa', 'Sugars', 'Tea, green', 
 'Chicory roots (Chicorium intybus sativum)', 'Cereal flours'}



# This dictionary maps every crop FAO name from earth stat dataset to its correponding crop group, which can be used when matching IDs from the harmonized Data set

cropGroupDict = {
'Fiber' : ['Manila Fibre (Abaca)', 'Agave Fibres Nes', 'Coir', 'Seed cotton', 'Fibre Crops Nes', 'Flax fibre and tow', 'Hemp Tow Waste', 'Jute', 'Other Bastfibres', 'Kapok Fibre', 'Kapokseed in Shell', 'Ramie', 'Sisal'] ,
'Forage' : ['alfalfa', 'beetfor', 'cabbagefor', 'carrotfor', 'clover', 'fornes', 'grassnes', 'legumenes', 'maizefor', 'mixedgrass', 'oilseedfor', 'ryefor', 'sorghumfor', 'swedefor', 'turnipfor', 'vegfor'] ,
'Treenuts' : ['Almonds, with shell', 'Brazil nuts, with shell', 'Cashew nuts, with shell', 'Chestnuts', 'Hazelnuts, with shell', 'Nuts, nes', 'Pistachios', 'Walnuts, with shell'] ,
'OtherCrops' : ['Anise, badian, fennel, corian', 'Arecanuts', 'Chicory roots', 'Cinnamon (canella)', 'Cloves', 'Cocoa beans', 'Coffee, green', 'Ginger', 'Gums Natural', 'Hops', 'Kolanuts', 'Mat_', 'Nutmeg, mace and cardamoms', 'Pepper (Piper spp.)', 'Peppermint', 'Chillies and peppers, dry', 'Pyrethrum,Dried', 'Natural rubber', 'Spices, nes', 'Tea', 'Tobacco, unmanufactured', 'Vanilla'] ,
'Fruit' : ['Apples', 'Apricots', 'Avocados', 'Bananas', 'Berries Nes', 'Blueberries', 'Carobs', 'Cashewapple', 'Cherries', 'Citrus fruit, nes', 'Cranberries', 'Currants', 'Dates', 'Figs', 'Fruit Fresh Nes', 'Gooseberries', 'Grapes', 'Grapefruit (inc. pomelos)', 'Kiwi fruit', 'Lemons and limes', 'Mangoes, mangosteens, guavas', 'Oranges', 'Papayas', 'Peaches and nectarines', 'Pears', 'Persimmons', 'Pineapples', 'Plantains', 'Plums and sloes', 'Quinces', 'Raspberries', 'Sour cherries', 'Stone fruit, nes', 'Strawberries', 'Tangerines, mandarins, clem', 'Fruit, tropical fresh nes'] ,
'Vegetables&Melons' : ['Artichokes', 'Asparagus', 'Cabbages and other brassicas', 'Carrots and turnips', 'Cauliflowers and broccoli', 'Chillies and peppers, green', 'Cucumbers and gherkins', 'Eggplants (aubergines)', 'Garlic', 'Beans, green', 'Leguminous vegetables, nes', 'Maize, green', 'Onions (inc. shallots), green', 'Peas, green', 'Lettuce and chicory', 'Other melons (inc.cantaloupes)', 'Mushrooms and truffles', 'Okra', 'Onions, dry', 'Pumpkins, squash and gourds', 'Spinach', 'String beans', 'Tomatoes', 'Vegetables fresh nes', 'Watermelons'] ,
'Pulses' : ['Bambara beans', 'Beans, dry', 'Broad beans, horse beans, dry', 'Chick peas', 'Cow peas, dry', 'Lentils', 'Lupins', 'Peas, dry', 'Pigeon peas', 'Pulses, nes', 'Vetches'] ,
'Cereals' : ['Barley', 'Buckwheat', 'Canary seed', 'Cereals, nes', 'Fonio', 'Maize', 'Millet', 'Mixed grain', 'Oats', 'Popcorn', 'Quinoa', 'Rice, paddy', 'Rye', 'Sorghum', 'Triticale', 'Wheat'] ,
'Roots&Tubers' : ['Cassava', 'Potatoes', 'Roots and Tubers, nes', 'Sweet potatoes', 'Taro (cocoyam)', 'Yams', 'Yautia (cocoyam)'] ,
'Oilcrops' : ['Castor oil seed', 'Coconuts', 'Groundnuts, with shell', 'Hempseed', 'Karite Nuts (Sheanuts)', 'Linseed', 'Melonseed', 'Mustard seed', 'Oil palm fruit', 'Oilseeds, Nes', 'Olives', 'Poppy seed', 'Rapeseed', 'Safflower seed', 'Sesame seed', 'Soybeans', 'Sunflower seed', 'Tung Nuts'] ,
'SugarCrops' : ['Sugar beet', 'Sugar cane', 'Sugar crops, nes']
            }


auxiliary_data_2017 = {'edible': 76, 'seed': 62, 'ground': 40, 'whether or not containing added sugar': 36, 'prepared or preserved': 29, 'whether or not refined': 21, 'powder':20, 'beans': 18, 'other than seed': 13, 'potato': 11, 'roasted': 11, 'maize (corn)': 10, 'oats': 10, 'potatoes': 10, 'wheat': 9, 'grape': 9, 'uncooked or cooked by steaming or boiling in water': 9, 'mushrooms': 8, 'apple': 8, 'vegetables': 8, 'fit for human consumption': 7, 'rice': 7, 'citrus': 7, 'orange': 6, 'not roasted': 6, 'paste': 6, 'grapefruit': 5, 'fructose': 5, 'of a kind used primarily in perfumery': 5, 'cereals': 5, 'black': 5, 'mushrooms of the genus Agaricus': 5, 'flowers and buds of a kind suitable for bouquets or ornamental purposes': 5, 'cherries': 4, 'pineapple': 4, 'worked': 4, 'olives': 4, 'rye': 4, 'cucumbers and gherkins': 4, 'peas (pisum sativum)': 4, 'decaffeinated': 4, 'apples': 4, 'homogenised': 4, 'wheat and meslin': 4, 'tomato': 4, 'pasta': 4, 'low erucic acid rape or colza oil and its fractions': 4, 'apricots': 3, 'watermelons': 3, 'strawberries': 3, 'of maize (corn)': 3}

def transformData(hs_data, cropData):
    '''
    Converts csv files into subscriptable dictionary data for the harmonized system, and formats the earthsat FAO data into a dictionary sorted by crop Groups

    '''


    hs_data = pd.read_csv(hs_data)
    # skeleton for harmizonied Dataframe
    transformedData = {
        'ID': [],
        'Category': [],
        'Description': []
    }
    

    # set of all crop categories in harmonized set
    crop_types = set()
    #set of all crop descriptions following semicolon in harmonized set
    description = set()
    descriptions = set()
    # set of all auxillary data (preparation characteristics) in harmonized set
    auxillaryData = set()
    
    for index, row in hs_data.iterrows():
        # add crop ID to dict
        transformedData['ID'].append(row['code'])
        if ';' in row['desc']:
            # we are splitting crop category and description by the semicolon found in each entry, and splitting each property to be visualized in a matrix
            temp = row['desc'].split(';')

            # temp[0] is each crop category that precedes semicolon
            crop_types.add(temp[0])
            transformedData['Category'].append(temp[0])
            
            
            # split each unique description property separated by commas
            desc = temp[1].split(',')
            
            #print('description:', desc[-1][1:])
            
            # add auxillary data of preparation details into set
            auxillaryData.add(desc[-1][1:])
            
            # remove spaces in front
            for i in range(len(desc)):
                desc[i] = desc[i][1:]
                
            # add each description to set of all crop descriptions
            descriptions.add(desc[0])
            
            # primary description - first phrase that follows semicolon - priority key for matching between data sets
            primary = desc[0]
            transformedData['Description'].append(temp[1])
            
            
        else:
            # when description is not separated by semicolon, treat entire entry as category
            transformedData['Category'].append(row['desc'])
            transformedData['Description'].append('N/A')
            
            
            
    for elem in descriptions:
        if elem == '':
            continue
        # create a column for each unique description property
        transformedData[elem] = [] 
        
    for index, row in hs_data.iterrows():
        for description in descriptions:
            # ignore empty entries
            if description == '':
                continue
            # add 1 to col if crop has description, 0 otherwise
            if description in row['desc']:
                transformedData[description].append(1)
            else:
                transformedData[description].append(0)
        
    #print(new_dict)
    
    #EARTHSTAT DATA
    #Crop dictionary - grouping crop types to list of crop entries
    cropGroups = {}
    for index, row in cropData.iterrows():
        if row['GROUP'] not in cropGroups:
            cropGroups[row['GROUP']] = [(row['Cropname_FAO'],row['Cropname_FAO_original'])]
        else:
            cropGroups[row['GROUP']].append((row['Cropname_FAO'],row['Cropname_FAO_original']))
            
    #print('auxillary data:', auxillaryData) 
    
    return transformedData, cropGroups



def searchID(cropID, harmonizedData):
    '''
    given an inoput of an ID, returns the crops category, descriptions, and auxillary data
    
    '''
    for index, row in harmonizedData.iterrows():
        if cropID == row['ID']:
            return row['Description']
    return None


def matchData(earthStatDict, harmonizedData, groups = ['Vegetables&Melons']):
    '''
    Will match each crop ID to an associated FAO cropname and crop group. Matching approach TBD
    
    Sample Model on Subset Data- Agenda:
        Find all FAO cropnames in the Category "Vegetables&Melons"
        Iterate through each cropname and find a description in harmonized set that matches
        Link crop ID with respective FAO cropname in a data structure
        
        
    Parameters:
        
        earthstatDict: Dictionary containing the crop Groups as Keys and its list of corresponding FAO cropnames as values
        harmonizedData: dataframe of each crop ID with binary encoding of descriptions
    '''
    
    matchingCSV = {
        'Group': [],
        'FAO Cropname': [],
        'Crop ID': [],
        'Description': [],
        'Auxiliary Info': []
        }
    

    

    # currently dealing with issues of duplicate nodes, so we are filtering for only unique IDs for now
    currentIDs = set()
    
    tree = {}
    harmonizedData = pd.DataFrame.from_dict(harmonizedData)
    # will create tree visualization for each crop group
    for group in groups:
        
        tree[group] = {}

        # list will hold all crop IDs whose descriptions match the cropname 
        for cropname in earthStatDict[group]:
            tree[group][cropname] = []
    
    print(tree)
        
    
    for index, row in harmonizedData.iterrows():
        for group in groups:
            for cropname in tree[group]:
                #if 'Vegetable' in row['Category']:
                #print('cropname', cropname, 'description:', row['Description'])
                if cropnamesMatch(cropname[0], row['Description']):
                    tree[group][cropname].append(row['ID'])
                    currentIDs.add(row['ID'])
                    
                    matchingCSV['Group'].append(group)
                    matchingCSV['FAO Cropname'].append(cropname[1])
                    matchingCSV['Crop ID'].append(row['ID'])
                    
                    desc = row['Description']
                    if ',' not in desc:
                        matchingCSV['Description'].append(desc)
                        matchingCSV['Auxiliary Info'].append('N/A')
                    else:
                        desc = desc.split(',')
                        matchingCSV['Description'].append(desc[0])
                        matchingCSV['Auxiliary Info'].append(''.join(desc[1:]))



                
                    
                    # Creating binary map for auxiliary data - to be used for descriptive analysis on MFN rates
                    

    #extraneous = harmonizedData.assign(result=harmonizedData['ID'].isin(matchingData['Crop ID']).astype(int))
    #extraneous = pd.Series(harmonizedData['ID'].isin(matchingData['Crop ID']).values.astype(int), harmonizedData.ID.values)

    excluded_data = {
        'ID': [],
        'Description': []
    }

    #raw_hs_2017 = pd.read_csv('hs6_2017_list.csv')
    for index, row in harmonizedData.iterrows():
        match_found = False
        if row['ID'] not in matchingCSV['Crop ID']:
            for group in groups:
                if group == 'Vegetables&Melons':
                    if 'Vegetables' in row['Category'] or ('Vegetable' in row['Category'] and 'oil' not in row['Category']) or 'Melons' in row['Category']:
                        match_found = True
                elif group == 'Oilcrops' and ('Oil' in row['Category'] or 'oils' in row['Category']):
                    match_found = True
                elif group == 'Treenuts' and 'Nuts' in row['Category']:
                    match_found == True
                elif group in row['Category']:
                    match_found = True

                if match_found:
                    currentIDs.add(row['ID'])
                    
                    matchingCSV['Group'].append(group)
                    matchingCSV['FAO Cropname'].append('NA')
                    matchingCSV['Crop ID'].append(row['ID'])
                    
                    desc = row['Description']
                    if ',' not in desc:
                        matchingCSV['Description'].append(desc)
                        matchingCSV['Auxiliary Info'].append('NA')
                    else:
                        desc = desc.split(',')
                        matchingCSV['Description'].append(desc[0])
                        matchingCSV['Auxiliary Info'].append(''.join(desc[1:]))
                    match_found = False

            if row['ID'] not in matchingCSV['Crop ID']:
                excluded_data['ID'].append(row['ID'])
                excluded_data['Description'].append(row['Category'] + row['Description'])

    excluded_data = pd.DataFrame.from_dict(excluded_data)
    excluded_data.to_csv('../Output_Data/excluded_crops_2017.csv')

    matched_ids = matchingCSV['Crop ID']

    
    #top_auxiliary_data = auxiliary_frequency(harmonizedData, matched_ids, length=10)

    top_auxiliary_data = ['edible', 'modified', 'fresh', 'frozen', 'shelled', 'dried', 'chilled','seed', 'ground', 'preserved']
    print('top auxiliary desc: ', top_auxiliary_data)

    for desc in top_auxiliary_data:
        matchingCSV[desc] = []

    for aux in matchingCSV['Auxiliary Info']:
        for desc in top_auxiliary_data:
            if cropnamesMatch(desc, aux):
                matchingCSV[desc].append(1)
            else:
                matchingCSV[desc].append(0)

    
    matchingData = pd.DataFrame.from_dict(matchingCSV)
    return matchingData, top_auxiliary_data


    
    
def cropnamesMatch(cropFAO, cropDescription):
    '''
    See if the FAO cropname substrings exists as a substring with the crop desription in the harmonized data set, and is therefore a valid match
    Parameters:
        cropFAO - cropname from earthstat, str
        cropDescription - crop description from harmonized system
    
    '''
    #  make all terms lowercase as matching is case sensistive
    crop = cropFAO.lower()
    
    
    description = cropDescription
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in description:
        if ele in punc:
            description = description.replace(ele, "")
    
    # add lemmitization and stemming

    description = description.split(' ')
    '''
    temp = []
    # Lemmatize Single Word - reduces plural word to singular form
    for i in range(len(description)):
        temp.append(WordNetLemmatizer().lemmatize(description[i]))

    description = temp
    
    crop_temp = []
    for i in range(len(crop)):
        crop_temp.append(WordNetLemmatizer().lemmatize(crop[i]))
    
    crop = crop_temp
    
    print('description:', description)
    print('cropname:', crop)
    '''
    match = False

    # differentiate between multi-word vs singular cropFAO names
    if ',' in crop:
        crop = crop.split(',')
        #print('new cropname:', crop)
        # check if each substring is in the crop description
        for sub in crop:
            if sub in description:
                return True
    else:
        if crop in description:
                return True
    
    
    #Lemmatize every word in both groups
    
    # used to verify function is working correctly
    #if match:
    #    print(cropFAO, cropDescription)
    return match


def auxiliary_frequency(harmonizedData, ids, length):
    """
    Finds most common auxiliary data in a harmonized data set and outputs the top 50 entries in ascending order

    Input: Panda Dataframe - Harmonized Dataset

    Output: Dictionary
    """
    for index, row in harmonizedData.iterrows():
        if row['ID'] not in ids:
            harmonizedData = harmonizedData.drop(labels=index, axis=0)
    
    names = []
    frequencies = []
    for name, elements in harmonizedData.iteritems():
        if name == 'ID' or name == 'Category' or name == 'Description':
            continue
        else:
            #print('name:', name)
            #print('freq:', harmonizedData[name].sum())
            names.append(name)
            frequencies.append(harmonizedData[name].sum())

    # Selection Sorting Algorithm to Sort Frequency List
    A = frequencies
    B = names
    # Traverse through all array elements
    for i in range(len(A)):
        
        # Find the minimum element in remaining 
        # unsorted array
        max_idx = i
        for j in range(i+1, len(A)):
            if A[max_idx] < A[j]:
                max_idx = j
                
        # Swap the found minimum element with 
        # the first element        
        A[i], A[max_idx] = A[max_idx], A[i]
        B[i], B[max_idx] = B[max_idx], B[i]

    frequencyDict = {}
    for i in range(len(names)):
        frequencyDict[names[i]] = frequencies[i]

    filtered_frequencies = {}

    for name in frequencyDict:
        if len(filtered_frequencies) > length:
            break
        filtered_frequencies[name] = frequencyDict[name]

    return filtered_frequencies



def confusion_matrix(auto_df, manual_df):
    '''
    Builds confusion matrix comparing results from matching algorithm that that of manually coded matching data

    Inputs: Matching CSV from algorithm, Matching CSV from manual work

    '''
    auto_df = pd.read_csv(auto_df)
    manual_df = pd.read_csv(manual_df)
    total_1997_ids = set()
    total = 0
    num_matched = 0
    num_mismatched = 0
    same_id_num = 0
    for idex_1, row_1 in manual_df.iterrows():
        total += 1
        total_1997_ids.add(row_1['hs6_1996'])
        for index_2, row_2 in auto_df.iterrows():
            if row_2['Crop ID'] == row_1['hs6_1996']:
                same_id_num += 1
                if row_2['Group'] == row_1['group']:
                    num_matched += 1
                else:
                    num_mismatched += 1
    print('Total Number of IDs:', total)
    print('Number of Same IDs:', same_id_num)
    print('Number of Correct Matches:', num_matched)
    print('Number of Incorrect Mathches:', num_mismatched)
    print('Crop Matching Algorithm Accuracy:', float(num_matched/(num_matched+num_mismatched)))
    print('Total IDs in manual data:', len(total_1997_ids))




#def isolate_unmatched_crops(hs_data, matching_data):





def run():
    # Harmonized Data set to be used 
    hs_file = input("Enter the harmonized data set:")
    # hs6_2017_list.csv
    # hs6_1996_list.csv
    # Region Crop Group Auxiliary Data

    # Transforms files to dataframes
    harmonizedData, cropGroupDict = transformData(hs_file, cropData)
    harmonizedData = pd.DataFrame.from_dict(harmonizedData)

    # Save harmonized data file in correspondance to its eyar
    hs_file = hs_file.split('_')
    year = hs_file[1]
    file_name = 'harmonizedData' + year + '.csv'
    harmonizedData.to_csv('../Output_Data/'+file_name)

    # Match harmonized IDs to FAO Crop Groups
    groups = cropGroupDict.keys()
    matchingData, auxiliary_data = matchData(cropGroupDict, harmonizedData, groups)
    file_name = 'tariff_prod_match_merge_hs6_' + year + '.csv'
    print(file_name)

    #Save matching data
    matchingData.to_csv('../Output_Data/' + file_name)
    #print(auxiliary_frequency(harmonizedData))

    # Determine Desriptive Analysis to Be Applied on Data - Analyze by Region, Crop Group or Auxiliary Data
    tariffData = pd.read_csv('wits_tariff_2018.csv')
    category = input('Matching Process Succesful. Select Filter to Run Descriptive Analysis: Region, Crop Group, Auxiliary Data: ')
    if category == 'Region' or category == 'region'or category == '1':
        regionTariffDict = buildRegionData(tariffData)
        print(regionTariffDict)
    elif category == 'Crop Group' or category == 'crop group' or category == '2':
        buildCropGroupData(tariffData, matchingData)
    elif category == 'Auxiliary Data' or category == 'auxiliary data' or category == '3':
        buildAuxiliaryData(matchingData, tariffData, auxiliary_data)
    else:
        print('No Filter Selected. Terminating Task...')
    





if __name__ == '__main__':
    run()
    #print(cropnamesMatch('peas, dry',' peas (pisum sativum), shelled or unshelled, fresh or chilled'))
    #print('running now')
    #lemmatizer = WordNetLemmatizer()

    
    #print(lemmatizer.lemmatize("rocks")) # rocks --> rock



    #confusion_matrix('../Output_Data/tariff_prod_match_merge_hs6_1996.csv', "tariff_prod_match_merge_hs6_1997.csv")
    #harmonizedData, cropGroupDict = transformData('hs6_2017_list.csv', cropData)

    #groups = cropGroupDict.keys()
    #harmonizedData = pd.DataFrame.from_dict(harmonizedData)
    #groups = ['Fiber', 'Forage', 'Treenuts', 'OtherCrops']
    #matchData(cropGroupDict, harmonizedData, groups)
        
    
    #print(cropnamesMatch('Cauliflowers broccoli', 'cauliflowers, broccoli and other delicacies'))
    
    
    #cropGroupData = pd.DataFrame.from_dict(cropGroupData)

    
    # Driver code to test above

    #total = data['of a kind used primarily in perfumery'].sum()
    #print(total)
    #data.to_csv('harmonizedData.csv')
    #print(data)
    


















