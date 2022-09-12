#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:09:16 2022

@author: sebastianquintero
"""
#import pip._vendor.requests
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
#import scipy.stats as stats
#from scipy.stats import norm
import seaborn as sn
import string
#from treelib import Node, Tree
import tabula
#import plotly
#import plotly.graph_objects as go



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





def findCropGroup(cropID, matchingData):
    '''
    Using matchingData.csv, find the crop group of the crop ID input and return it
    
    '''
    for index, row in matchingData.iterrows():
        if row['Crop ID'] == cropID:
           return row['Group']
    print(cropID)
    raise Exception('Crop ID not found')


def findAuxData(cropID, matchingData, auxiliaryData):
    '''
    Using matchingData.csv, find the list of applicable auxiliary data p of the crop ID input and return it
    
    '''
    result = []
    for index, row in matchingData.iterrows():
        if row['Crop ID'] == cropID:
            for aux in auxiliaryData:
                try:
                    if row[aux] == 1:
                        result.append(aux)
                except:
                    continue
            #print(row)
            return result
    raise Exception('Crop ID not found')


def buildCountryDict():
    '''
    constructs country to council region dictionary to group each country by region for descriptive analysis.
    
    '''
    
    # Parsing data from pdf as basis for regional division
    file1 = "https://www.theisn.org/wp-content/uploads/2020/11/List_of_countries_and_corresponding_Council_regions.pdf"
    countryDict = {}
    regions = set()
    
    # Parse data from all 6 pages of the pdf
    for i in range(1,6):
        table = tabula.read_pdf(file1,pages=i)
        countryData = table[0]
        
        for index, row in countryData.iterrows():
            countryDict[row['Country']] = row['Council Region']
            regions.add(row['Council Region'])
            

    return countryDict, list(regions)



def buildRegionData(tariffData):
    
    #countryDict, regions = buildCountryDict()
    countryDict = {'Afghanistan': 'South Asia', 'Albania': 'Eastern & Central Europe', 'Algeria': 'Africa', 'American Samoa': 'Australia, New Zealand & Polynesia', 'Andorra': 'Mediterranean Europe', 'Angola': 'Africa', 'Antigua and Barbuda': 'North America', 'Argentina': 'Latin America', 'Armenia': 'NIS & Russia', 'Aruba': 'North America', 'Australia': 'Australia, New Zealand & Polynesia', 'Austria': 'Continental Europe', 'Azerbaijan': 'NIS & Russia', 'The Bahamas': 'North America', 'Bahrain': 'Middle East', 'Bangladesh': 'South Asia', 'Barbados': 'North America', 'Belarus': 'NIS & Russia', 'Belgium': 'Continental Europe', 'Belize': 'Latin America', 'Benin': 'Africa', 'Bermuda': 'North America', 'Bhutan': 'South Asia', 'Bolivia': 'Latin America', 'Bosnia and Herzegovina': 'Eastern & Central Europe', 'Botswana': 'Africa', 'Brazil': 'Latin America', 'Brunei': 'South East Asia', 'Bulgaria': 'Eastern & Central Europe', 'Burkina Faso': 'Africa', 'Burundi': 'Africa', 'Cabo Verde': 'Africa', 'Cambodia': 'South East Asia', 'Cameroon': 'Africa', 'Canada': 'North America', 'Cayman Islands': 'North America', 'Central African Republic': 'Africa', 'Chad': 'Africa', 'Channel Islands': 'Continental Europe', 'Chile': 'Latin America', 'China': 'East Asia', 'Colombia': 'Latin America', 'Comoros': 'Africa', 'Congo': 'Africa', 'Costa Rica': 'Latin America', "Côte D'Ivoire": 'Africa', 'Croatia': 'Eastern & Central Europe', 'Cuba': 'Latin America', 'Curaçao': 'Latin America', 'Cyprus': 'Eastern & Central Europe', 'Czech Republic': 'Eastern & Central Europe', "Dem. People's Rep. Korea": 'North Asia', 'Dem. Rep. Congo': 'Africa', 'Denmark': 'Scandinavia', 'Djibouti': 'Africa', 'Dominica': 'Latin America', 'Dominican Republic': 'Latin America', 'Ecuador': 'Latin America', 'Egypt': 'Africa', 'El Salvador': 'Latin America', 'Equatorial Guinea': 'Africa', 'Eritrea': 'Africa', 'Estonia': 'Eastern & Central Europe', 'Ethiopia': 'Africa', 'Faeroe Islands': 'Continental Europe', 'Fiji': 'Australia, New Zealand & Polynesia', 'Finland': 'Scandinavia', 'France': 'Continental Europe', 'French Guiana': 'Latin America', 'French Polynesia': 'Australia, New Zealand & Polynesia', 'Gabon': 'Africa', 'The Gambia': 'Africa', 'Georgia': 'NIS & Russia', 'Germany': 'Continental Europe', 'Ghana': 'Africa', 'Greece': 'Mediterranean Europe', 'Greenland': 'North America', 'Grenada': 'North America', 'Guadeloupe (French)': 'Latin America', 'Guam': 'Australia, New Zealand & Polynesia', 'Guatemala': 'Latin America', 'Guinea': 'Africa', 'Guinea-Bissau': 'Africa', 'Guyana': 'Latin America', 'Haiti': 'Latin America', 'Honduras': 'Latin America', 'Hong Kong SAR, China': 'East Asia', 'Hungary': 'Eastern & Central Europe', 'Iceland': 'Scandinavia', 'India': 'South Asia', 'Indonesia': 'South East Asia', 'Iran': 'Middle East', 'Iraq': 'Middle East', 'Ireland': 'Continental Europe', 'Isle of Man': 'Continental Europe', 'Israel': 'Mediterranean Europe', 'Italy': 'Mediterranean Europe', 'Jamaica': 'North America', 'Japan': 'North Asia', 'Jordan': 'Middle East', 'Kazakhstan': 'NIS & Russia', 'Kenya': 'Africa', 'Kiribati': 'Australia, New Zealand & Polynesia', 'Korea': 'North Asia', 'Kosovo': 'Eastern & Central Europe', 'Kuwait': 'Middle East', 'Kyrgyz Republic': 'NIS & Russia', 'Lao PDR': 'South East Asia', 'Latvia': 'Eastern & Central Europe', 'Lebanon': 'Middle East', 'Lesotho': 'Africa', 'Liberia': 'Africa', 'Libya': 'Africa', 'Liechtenstein': 'Continental Europe', 'Lithuania': 'Eastern & Central Europe', 'Luxembourg': 'Continental Europe', 'Macao SAR, China': 'East Asia', 'Macedonia': 'Eastern & Central Europe', 'Madagascar': 'Africa', 'Malawi': 'Africa', 'Malaysia': 'South East Asia', 'Maldives': 'South Asia', 'Mali': 'Africa', 'Malta': 'Mediterranean Europe', 'Marshall Islands': 'Australia, New Zealand & Polynesia', 'Martinique (French)': 'Latin America', 'Mauritania': 'Africa', 'Mauritius': 'Africa', 'Mexico': 'Latin America', 'Micronesia': 'Australia, New Zealand & Polynesia', 'Moldova': 'Eastern & Central Europe', 'Monaco': 'Mediterranean Europe', 'Mongolia': 'East Asia', 'Montenegro': 'Eastern & Central Europe', 'Morocco': 'Africa', 'Mozambique': 'Africa', 'Myanmar': 'South East Asia', 'Namibia': 'Africa', 'Nepal': 'South Asia', 'Netherlands': 'Continental Europe', 'New Caledonia': 'Australia, New Zealand & Polynesia', 'New Zealand': 'Australia, New Zealand & Polynesia', 'Nicaragua': 'Latin America', 'Niger': 'Africa', 'Nigeria': 'Africa', 'Northern Mariana Islands': 'Australia, New Zealand & Polynesia', 'Norway': 'Scandinavia', 'Oman': 'Middle East', 'Pacific Islands': 'Australia, New Zealand & Polynesia', 'Pakistan': 'South Asia', 'Palau': 'Australia, New Zealand & Polynesia', 'Panama': 'Latin America', 'Papua New Guinea': 'Australia, New Zealand & Polynesia', 'Paraguay': 'Latin America', 'Peru': 'Latin America', 'Philippines': 'South East Asia', 'Poland': 'Eastern & Central Europe', 'Portugal': 'Mediterranean Europe', 'Puerto Rico': 'Latin America', 'Qatar': 'Middle East', 'Reunion (French)': 'Africa', 'Romania': 'Eastern & Central Europe', 'Russia': 'NIS & Russia', 'Rwanda': 'Africa', 'Samoa': 'Australia, New Zealand & Polynesia', 'San Marino': 'Mediterranean Europe', 'São Tomé and Principe': 'Africa', 'Saudi Arabia': 'Middle East', 'Senegal': 'Africa', 'Serbia': 'Eastern & Central Europe', 'Seychelles': 'Africa', 'Sierra Leone': 'Africa', 'Singapore': 'South East Asia', 'Sint Maarten (Dutch part)': 'North America', 'Slovak Republic': 'Eastern & Central Europe', 'Slovenia': 'Eastern & Central Europe', 'Solomon Islands': 'Australia, New Zealand & Polynesia', 'Somalia': 'Africa', 'South Africa': 'Africa', 'South Sudan': 'Africa', 'Spain': 'Mediterranean Europe', 'Sri Lanka': 'South Asia', 'St. Kitts and Nevis': 'North America', 'St. Lucia': 'North America', 'St. Martin (French part)': 'North America', 'St. Vincent and the Grenadines': 'North America', 'Sudan': 'Africa', 'Suriname': 'Latin America', 'Swaziland': 'Africa', 'Sweden': 'Scandinavia', 'Switzerland': 'Continental Europe', 'Syrian Arab Republic': 'Middle East', 'Taiwan': 'East Asia', 'Tajikistan': 'NIS & Russia', 'Tanzania': 'Africa', 'Thailand': 'South East Asia', 'Timor-Leste': 'South East Asia', 'Togo': 'Africa', 'Tonga': 'Australia, New Zealand & Polynesia', 'Trinidad and Tobago': 'North America', 'Tunisia': 'Africa', 'Turkey': 'Eastern & Central Europe', 'Turkmenistan': 'NIS & Russia', 'Turks and Caicos Islands': 'North America', 'Tuvalu': 'Australia, New Zealand & Polynesia', 'Uganda': 'Africa', 'Ukraine': 'NIS & Russia', 'United Arab Emirates': 'Middle East', 'United Kingdom': 'Continental Europe', 'United States': 'North America', 'Uruguay': 'Latin America', 'Uzbekistan': 'NIS & Russia', 'Vanuatu': 'Australia, New Zealand & Polynesia', 'Vatican': 'Mediterranean Europe', 'Venezuela': 'Latin America', 'Vietnam': 'South East Asia', 'Virgin Islands': 'North America', 'West Bank & Gaza (Palestinian Authority)': 'Middle East', 'Yemen': 'Middle East', 'Zambia': 'Africa', 'Zimbabwe': 'Africa'}
    regions = list(set(countryDict.values()))

    colors = []
    prev_nation = None
    hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]

    
    regionDict = {}
    regionColors = []
    for region in regions:
        regionDict[region] = []
    
    for index, row in tariffData.iterrows():
        
        #if index >200000:
         #   break
        #mfn.append(row['mfn_rate'])
        if row['name'] not in countryDict:
            continue
        currentRegion = countryDict[row['name']]
        regionDict[currentRegion].append(row['mfn_rate'])
        
        if index % 100000 == 0:
            print('row ' + str(index) + '/700,000 parsed')
        '''
        
        
        if prev_nation != None and row['iso3code'] == prev_nation:
            colors.append(hexadecimal[0])
            
            
            
        else:
            
            average = sum(mfn)/len(mfn)
            average_mfn.append(average)
            
            mfn = []
            
            hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
            
            country_colors.append(hexadecimal[0])
            colors.append(hexadecimal[0])
            countries.append(row['iso3code'])
        '''
            

    regions = []
    avgMFN = []
    
    for region in regionDict:
        if len(regionDict[region]) > 0:
            regionDict[region] = round(sum(regionDict[region])/len(regionDict[region]), 2)
            
        else:
            regionDict[region] = 0
        hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        regionColors.append(hexadecimal[0])
        
    print(regionDict)
    sorted_dict = {}
    sorted_keys = sorted(regionDict, key=regionDict.get)  # [1, 3, 2]
    
    for w in sorted_keys:
        sorted_dict[w] = regionDict[w]

    for region in sorted_dict:
        regions.append(region)
        avgMFN.append(sorted_dict[region])
    
    def add_value_label(x_list,y_list):
        for i, v in enumerate(y_list):
            plt.text(v, i, str(v), color='black', fontweight='bold', fontsize=12, ha='left', va='center')
    
    plt.barh(regions, avgMFN, color=regionColors)
    plt.ylabel('Regions')
    plt.xlabel('Average MFN rate (%)')
    plt.suptitle('*The MFN Rate of X represents X% ad valorem tariff imposed to \n products belonging to a certain product ID. For example, 12 % ad \n valorem tariff means, if 100 dollars worth of products are imported, \n the government collects 12 dollars in tariff revenue for them.', va='bottom', fontsize=12, y=0.2, x=0.7)
    plt.title('Average Tariff MFN Rates by Region, 2018', fontsize=20)
    
    figure = plt.gcf()
    add_value_label(regions, avgMFN)
    figure.set_size_inches(15, 10)
    plt.savefig('Tariff Model.png', bbox_inches='tight', dpi=1080)
    print(regionColors)
    
    
    
    
    
    
    return sorted_dict




def buildCropGroupData(tariffData, matchingData):
    print('running...')
    hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
    groups = ['Fiber', 'Forage', 'Treenuts', 'OtherCrops', 'Fruit', 'Vegetables&Melons', 'Pulses', 'Cereals', 'Roots&Tubers', 'Oilcrops', 'SugarCrops']
    
    #matchingData = pd.read_csv('matchingData.csv')
    cropIDs = set(matchingData['Crop ID'].tolist())
    groupDict = {}
    groupColors =[]
    
    for group in groups:
        groupDict[group] = []
    
    for index, row in tariffData.iterrows():
        
        #if index >200000:
         #   break
        #mfn.append(row['mfn_rate'])
        
        if row['hs6'] not in cropIDs:
            continue
        group = findCropGroup(row['hs6'], matchingData)
        groupDict[group].append(row['mfn_rate'])
        if index % 100000 == 0:
            print('row ' + str(index) + '/700,000 parsed')
            
    groups = []
    avgMFN = []
    
    for group in groupDict:
        if len(groupDict[group]) > 0:
            groupDict[group] = round(sum(groupDict[group])/len(groupDict[group]), 2)
            
        else:
            groupDict[group] = 0
            
        hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        groupColors.append(hexadecimal[0])
        
    sorted_dict = {}
    sorted_keys = sorted(groupDict, key=groupDict.get)
    
    for w in sorted_keys:
        sorted_dict[w] = groupDict[w]
        
    print(sorted_dict)
    for group in sorted_dict:
        groups.append(group)
        avgMFN.append(sorted_dict[group])
    
    def add_value_label(x_list,y_list):
        for i, v in enumerate(y_list):
            plt.text(v, i, str(v), color='black', fontweight='bold', fontsize=12, ha='left', va='center')
    
    plt.barh(groups, avgMFN, color=groupColors)
    plt.ylabel('Crop Groups')
    plt.xlabel('Average MFN rate (%)')
    plt.suptitle('*The MFN Rate of X represents X% ad valorem tariff imposed to \n products belonging to a certain product ID. For example, 12 % ad \n valorem tariff means, if 100 dollars worth of products are imported, \n the government collects 12 dollars in tariff revenue for them.', va='bottom', fontsize=12, y=0.15, x=0.7)
    plt.title('Average Tariff MFN Rates by Crop Group, 2018', fontsize=20)
    
    figure = plt.gcf()
    add_value_label(groups, avgMFN)
    figure.set_size_inches(15, 10)
    plt.savefig('Tariff Model - Crop Groups.png', bbox_inches='tight', dpi=1080)    
    
    return


def buildAuxiliaryData(matchingData, tariffData, auxiliaryData):
    '''
    Plot average MFN rates sorted by auxiliary data of matching csv file.

    '''
    print('running...')

    past_ids = set()

    unique_IDs = set()
    hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
    
    #matchingData = pd.read_csv('matchingData.csv')
    cropIDs = set(matchingData['Crop ID'].tolist())
    groupDict = {}
    groupColors =[]
    
    for group in auxiliaryData:
        groupDict[group] = []
    
    for index, row in tariffData.iterrows():
        
        #if index >200000:
         #   break
        #mfn.append(row['mfn_rate'])
        unique_IDs.add(row['hs6'])
        if row['hs6'] not in cropIDs:
            continue

        if row['hs6'] in past_ids:
            continue
            
        past_ids.add(row['hs6'])
        aux = findAuxData(row['hs6'], matchingData, auxiliaryData)
        for element in aux:
            groupDict[element].append(row['mfn_rate'])
        if index % 100000 == 0:
            print('row ' + str(index) + '/700,000 parsed')
            
    groups = []
    avgMFN = []
    

    for group in groupDict:
        if len(groupDict[group]) > 0:
            groupDict[group] = round(sum(groupDict[group])/len(groupDict[group]), 2)
            
        else:
            groupDict[group] = 0
            
        hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        groupColors.append(hexadecimal[0])
        
    sorted_dict = {}
    sorted_keys = sorted(groupDict, key=groupDict.get)
    
    for w in sorted_keys:
        sorted_dict[w] = groupDict[w]
        
    print(sorted_dict)
    for group in sorted_dict:
        groups.append(group)
        avgMFN.append(sorted_dict[group])
    
    def add_value_label(x_list,y_list):
        for i, v in enumerate(y_list):
            plt.text(v, i, str(v), color='black', fontweight='bold', fontsize=12, ha='left', va='center')
    
    plt.barh(groups, avgMFN, color=groupColors)
    plt.ylabel('Auxiliary Data')
    plt.xlabel('Average MFN rate (%)')
    plt.suptitle('*The MFN Rate of X represents X% ad valorem tariff imposed to \n products belonging to a certain product ID. For example, 12 % ad \n valorem tariff means, if 100 dollars worth of products are imported, \n the government collects 12 dollars in tariff revenue for them.', va='bottom', fontsize=12, y=0.15, x=0.7)
    plt.title('Average Tariff MFN Rates by Crop Group, 2018', fontsize=20)
    
    figure = plt.gcf()
    add_value_label(groups, avgMFN)
    figure.set_size_inches(15, 10)
    plt.savefig('Tariff Model - Auxiliary Data.png', bbox_inches='tight', dpi=1080)   

    matching_ids = set()
    for index, row in matchingData.iterrows():
        if row['Crop ID'] in unique_IDs:
            matching_ids.add(row['Crop ID'])
    print('Number of UNIQUE IDS in tariff line data: ', len(unique_IDs))
    print('Number of Similar IDs between Tariff Line and Matching Data: ', len(matching_ids))
    return

'''
def choropleth(tariffData):

    countryDf = {
        'Code': [],
        'Avg MFN': [],
        'Country': []
        }
    colors = []
    prev_nation = None
    
    mfn = []
    countries = {}
    for index, row in tariffData.iterrows():
        
        #if index >200000:
         #   break
        #mfn.append(row['mfn_rate'])
        current_nation = row['name']
        countries[row['iso3code']] = row['name']
        if prev_nation == None or current_nation == prev_nation:
            mfn.append(row['mfn_rate'])
            prev_nation = row['name']
        elif current_nation != prev_nation:
            avg = round(sum(mfn)/len(mfn), 2)
            print('average:', avg)
            countryDf['Code'].append(row['iso3code'])
            countryDf['Avg MFN'].append(avg)
            countryDf['Country'].append(row['name'])
            prev_nation = current_nation
            mfn = []
            
        if index % 100000 == 0:
            print('row ' + str(index) + '/700,000 parsed')


    
    
    fig = go.Figure(data=go.Choropleth(
        locations = np.array(countryDf['Code']),
        z = np.array(countryDf['Avg MFN']),
        text = np.array(countryDf['Country']),
        colorscale = 'Blues',
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_tickprefix = '$',
        colorbar_title = 'GDP<br>Billions US$',
    ))
    
    fig.update_layout(
        title_text='2014 Global GDP',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        annotations = [dict(
            x=0.55,
            y=0.1,
            xref='paper',
            yref='paper',
            text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
                CIA World Factbook</a>',
            showarrow = False
        )]
    )
    
    fig.show()
    
    df = pd.DataFrame.from_dict(countryDf)
    print(df)
    df.to_csv('country_mfn.csv')
    print(countries)
    print('length:', len(countries))
    #fig.savefig('MFN Map.png', bbox_inches='tight', dpi=1080)
    return countries


'''

    
if __name__ == '__main__':
    
    # Crop ID Metrics
    tariffData = pd.read_csv('wits_tariff_2018.csv')
    regionTariffDict = buildRegionData(tariffData)
    print(regionTariffDict)
    countryDict = buildCountryDict()
    print(countryDict)
    df ={
        'Name': [],
        'Avg MFN':[],
        'Region': []
        
        }
    
    #countries = choropleth(tariffData)
    for country in countries:
        try:
            df['Avg MFN'].append(regionTariffDict[countryDict[0][countries[country]]])
            df['Region'].append(countryDict[0][countries[country]])
            df['Name'].append(country)
        except:
            pass
        
    df1 = pd.DataFrame.from_dict(df)
    print(df1)
    df1.to_csv('region_mfn.csv')


    