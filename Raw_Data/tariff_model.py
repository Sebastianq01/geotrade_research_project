import pandas as pd
import numpy as np

from descriptiveAnalysis import *
from cropData import *




def prepare_data(match_data, tariff_data, polyarchy_data, production_data, trade_data, gdp_industry_data):
    match_data = pd.read_csv(match_data)
    tariff_data = pd.read_csv(tariff_data)
    polyarchy_data = pd.read_csv(polyarchy_data)
    polyarchy_data = polyarchy_data.loc[polyarchy_data['year'] > 2020, ['country_name', 'country_text_id', 'v2x_polyarchy']]
    polyarchy_data.to_csv('../Output_Data/polyarchy_data.csv')

    production_data = pd.read_csv(production_data)
    trade_data = pd.read_csv(trade_data)
    gdp_industry_data = pd.read_csv(gdp_industry_data)
    data = {
        'Crop ID': [],
        'FAO Cropname': [],
        'Crop Group': [],
        'Country': [],
        'iso3code': [],
        'gdp per capita': [],
        'percent gdp from agriculture': [],
        'democracy level': [],
        'production quantity': [],
        'import trade value': [],
        'export trade value': [],
        'edible': [],
        'modified': [],
        'fresh': [],
        'frozen': [],
        'shelled': [],
        'dried': [],
        'chilled': [],
        'seed': [],
        'ground': [],
        'preserved': [],
        'mfn': []
    }

    gdp_dict = {}
    for index, row in gdp_industry_data.iterrows():
        if row['Country Name'] not in gdp_dict:
            gdp_dict[row['Country Name']] = [row['2018 [YR2018]']]
        else:
            gdp_dict[row['Country Name']].append(row['2018 [YR2018]'])

    for index, row in tariff_data.iterrows():
        if index % 17500 == 0:
            percent = round(index/700000,2)*100
            print(str(percent) + '%')
        if row['hs6'] in set(match_data['Crop ID']):
            for i, r in match_data.iterrows():
                if row['hs6'] == r['Crop ID']:
                    data['Crop ID'].append(r['Crop ID'])
                    data['FAO Cropname'].append(r['FAO Cropname'])
                    data['Crop Group'].append(r['Group'])
                    data['Country'].append(row['name'])
                    data['edible'].append(r['edible'])
                    data['modified'].append(r['modified'])
                    data['fresh'].append(r['fresh'])
                    data['frozen'].append(r['frozen'])
                    data['shelled'].append(r['shelled'])
                    data['dried'].append(r['dried'])
                    data['chilled'].append(r['chilled'])
                    data['seed'].append(r['seed'])
                    data['ground'].append(r['ground'])
                    data['preserved'].append(r['preserved'])
                    data['mfn'].append(row['mfn_rate'])

                    if r['FAO Cropname'] in set(production_data.loc[production_data['Area'] == row['name']]['Item']):
                        for i2, r2 in production_data.loc[production_data['Area'] == row['name']].iterrows():
                            if r2['Item'] == r['FAO Cropname']:
                                data['production quantity'].append(r2['Value'])
                                break
                    else:
                        data['production quantity'].append('NA')
                    break
            
            if row['iso3code'] in set(polyarchy_data['country_text_id']):
                for i, r in polyarchy_data.iterrows():
                    if row['iso3code'] == r['country_text_id']:
                        data['iso3code'].append(row['iso3code'])
                        data['democracy level'].append(r['v2x_polyarchy'])
                        break
            else:
                data['iso3code'].append('NA')
                data['democracy level'].append('NA')

            commodity_code = 0
            if int(str(row['hs6'])[0:2]) > 21:
                commodity_code = int(str(row['hs6'])[0])
            else:
                commodity_code = int(str(row['hs6'])[0:2])
            
            import_found = False
            export_found = False
            for i, r in trade_data.loc[trade_data['Reporter'] == row['name']].iterrows():
                if commodity_code == r['Commodity Code'] and row['name'] == r['Reporter'] and r['Trade Flow'] == 'Import':
                    data['import trade value'].append(r['Trade Value (US$)'])
                    import_found = True
                elif commodity_code == r['Commodity Code'] and row['name'] == r['Reporter'] and r['Trade Flow'] == 'Export':
                    data['export trade value'].append(r['Trade Value (US$)'])
                    export_found = True

                if import_found and export_found:
                    break
            if not import_found:
                data['import trade value'].append('NA')
            if not export_found:
                data['export trade value'].append('NA')

            if  row['name'] not in gdp_dict or gdp_dict[row['name']][0] == '..':
                data['percent gdp from agriculture'].append('NA')
            else:
                data['percent gdp from agriculture'].append(gdp_dict[row['name']][0])
            if row['name'] not in gdp_dict or gdp_dict[row['name']][1] == '..':
                data['gdp per capita'].append('NA')
            else:
                data['gdp per capita'].append(gdp_dict[row['name']][1])
            

            

            
            
            

            


    output = pd.DataFrame.from_dict(data)

    output.to_csv('../Output_Data/tariff_model_data.csv')


def filter_country_data(tariff_data, category, auxiliary1, auxiliary2):
    mfn_dict = {}
    tariff_data = pd.read_csv(tariff_data)
    for index, row in tariff_data.iterrows():
        if str(row['FAO Cropname']) == 'nan':
            continue
        
        if category in row['FAO Cropname'] and (row[auxiliary1] == 1 or row[auxiliary2] == 1):
            if row['Country'] not in mfn_dict:
                mfn_dict[row['Country']] = []
            mfn_dict[row['Country']].append(row['mfn'])

    avg = 'Average MFN: ' + auxiliary + ' ' + category
    df = {
        'Country': [],
        avg: [],
        'Set size': []
    }
    for country in mfn_dict:
        df['Country'].append(country)
        df['Set size'].append(len(mfn_dict[country]))
        df[avg].append(round(sum(mfn_dict[country])/len(mfn_dict[country]),2))
    print(df)
    output = pd.DataFrame.from_dict(df)
    output.to_csv('../Output_Data/filtered_country_data.csv')



if __name__ == '__main__':
    print('Building Data...')
    prepare_data('../Output_Data/tariff_prod_match_merge_hs6_2017.csv', 'wits_tariff_2018.csv', 'Country_Year_V-Dem_Core_CSV_v12/V-Dem-CY-Core-v12.csv', 'FAOSTAT_data_prod_quant_2018.csv', 'comtrade_2018.csv', 'gdp_and_industry.csv')
    #filter_country_data('../Output_Data/tariff_model_data.csv', 'berries', 'preserved', 'fresh')

    '''
    output = pd.read_csv('../Output_Data/tariff_model_data.csv')
    result = output[(output['production quantity'] > 0) & (output['democracy level'] > 0)]
    result.to_csv('../Output_Data/tariff_model_data_FILTERED.csv')
    '''

    
    




