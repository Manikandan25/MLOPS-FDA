import pandas as pd
import numpy as np


def preprocess_data(df):
    df = df.drop(columns = ['FEINumberRecall','RecallingFirmName','RecallEventID','RecallEventClassification','RefusalFEINumber',
    'RefusedDate','AnalysisDone','OutbreakLevel','Id','ImportingCountry'])
    a = df[df['OriginCountry'].isin(['-'])]['OriginCountry']
    df['OriginContinent'] = df['OriginContinent'].fillna(a)
    df['PrimaryProcessingOriginContinent'] = df['PrimaryProcessingOriginContinent'].fillna(a)
    b = df[df['OriginContinent'].isna()]
    b['OriginContinent'][b['OriginCountry'].isin(['Faroe Islands', 'Gibraltar'])] = 'Europe'
    b['OriginContinent'][b['OriginCountry'].isin(['American Samoa', 'French Polynesia', 'New Caledonia', 'Tokelau'])] = 'Oceania'
    b['OriginContinent'][b['OriginCountry'].isin(['Hong Kong'])] = 'Asia'
    b['OriginContinent'][b['OriginCountry'].isin(['Greenland', 'British Virgin Islands', 'Turks and Caicos Islands', 'U.S. Virgin Islands'])] = 'North America'
    b['OriginContinent'][b['OriginCountry'].isin(['Reunion', 'Cape Verde', 'Saint Helena'])] = 'Africa'
    b['PrimaryProcessingOriginContinent'][b['OriginCountry'].isin(['Faroe Islands', 'Gibraltar'])] = 'Europe'
    b['PrimaryProcessingOriginContinent'][b['OriginCountry'].isin(['American Samoa', 'French Polynesia', 'New Caledonia', 'Tokelau'])] = 'Oceania'
    b['PrimaryProcessingOriginContinent'][b['OriginCountry'].isin(['Hong Kong'])] = 'Asia'
    b['PrimaryProcessingOriginContinent'][b['OriginCountry'].isin(['Greenland', 'British Virgin Islands', 'Turks and Caicos Islands', 'U.S. Virgin Islands'])] = 'North America'
    b['PrimaryProcessingOriginContinent'][b['OriginCountry'].isin(['Reunion', 'Cape Verde', 'Saint Helena'])] = 'Africa'
    df['OriginContinent'] = df['OriginContinent'].fillna(b['OriginContinent'])
    df['PrimaryProcessingOriginContinent'] = df['PrimaryProcessingOriginContinent'].fillna(b['PrimaryProcessingOriginContinent'])
    sec_na = df[df['SecondaryProcessingOriginCity'].isna()]
    c = sec_na[(sec_na['PrimaryProcessingOriginCity'] == 'NA') & (sec_na['SecondaryProcessingOriginCity'].isna())]['SecondaryProcessingOriginCity'].fillna('NA')
    sec_na['SecondaryProcessingOriginCity'] = sec_na['SecondaryProcessingOriginCity'].fillna(value = c)
    d = sec_na[sec_na['ProductType'].isin(['Frozen','Raw'])]['OriginCity']
    sec_na['SecondaryProcessingOriginCity'] = sec_na['SecondaryProcessingOriginCity'].fillna(d)
    mask = sec_na['SecondaryProcessingOriginCity'].isna() 
    ind = sec_na['SecondaryProcessingOriginCity'].loc[mask][sec_na['SecondaryProcessingOriginCountry'] == 'Canada'].sample(frac=0.3).index
    sec_na.loc[ind, 'SecondaryProcessingOriginCity']=sec_na.loc[ind, 'SecondaryProcessingOriginCity'].fillna('Vancouver')
    sec_na['SecondaryProcessingOriginCity'] = sec_na[sec_na['SecondaryProcessingOriginCountry'] == 'Canada']['SecondaryProcessingOriginCity'].fillna("Brampton")
    mask = sec_na['SecondaryProcessingOriginCity'].isna() 
    ind = sec_na['SecondaryProcessingOriginCity'].loc[mask][sec_na['SecondaryProcessingOriginCountry'] == 'Japan'].sample(frac=0.5).index
    sec_na.loc[ind, 'SecondaryProcessingOriginCity']=sec_na.loc[ind, 'SecondaryProcessingOriginCity'].fillna('Kagoshima')
    sec_na['SecondaryProcessingOriginCity'] = sec_na[sec_na['SecondaryProcessingOriginCountry'] == 'Japan']['SecondaryProcessingOriginCity'].fillna("Ojima")
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Mexico'])] = "Ensenada"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Chile'])] = "SANTIAGO"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Ecuador'])] = "Puerto Lopez"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Indonesia'])] = "Medan"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['India'])] = "Mumbai"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Thailand'])] = "Amphur Muang"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['United States'])] = "Shanghai"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['New Zealand'])] = "Mosgiel"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Honduras'])] = "Choluteca"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Spain'])] = "Madrid"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Ireland'])] = "Madrid"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Hong Kong'])] = "Kwai Chung"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Vietnam'])] = "Ho Chi Minh"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Peru'])] = "Piura"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Nicaragua'])] = "Managua"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Sri Lanka'])] = "Colombo"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['South Korea'])] = "Busan"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Denmark'])] = "Vinderup"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Philippines'])] = "Las Pinas"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Norway'])] = "Oslo"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Germany'])] = "Wallersdorf"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['France'])] = "Lyon"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Grenada'])] = "St. George"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Guatemala'])] = "Flores"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Italy'])] = "Genova"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Uruguay'])] = "Montevideo"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Guyana'])] = "Georgetown"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Venezuela'])] = "Cumana"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Latvia'])] = "Riga"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Morocco'])] = "Agadir"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Taiwan'])] = "Kaohsiung"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['China'])] = "Zhanjiang"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Iceland'])] = "Reykjavík"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Portugal'])] = "Matosinhos"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Greece'])] = "Keramoti"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Brazil'])] = "Itaoca"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['-'])] = "-"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Russia'])] = "Murmansk"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Australia'])] = "Mackay"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Turkey'])] = "Mugla"
    sec_na['SecondaryProcessingOriginCity'][sec_na['SecondaryProcessingOriginCountry'].isin(['Panama'])] = "BRISAS DE AMADOR"
    sec_na['SecondaryProcessingOriginCity'] = sec_na['SecondaryProcessingOriginCity'].fillna('Other')
    df['SecondaryProcessingOriginCity'] = df['SecondaryProcessingOriginCity'].fillna(sec_na['SecondaryProcessingOriginCity'])
# Filling OriginContinent with OriginCountry's continent
# Europe = Faroe Islands, Gibraltar
# Oceania = American Samao, FrenchPolynesia, New Caledonia, Tokelau
# Asia = Hong Kong
# north america = Greenland, British Virgin Islands, Turks and Caicos Islands, U.S. Virgin Islands
# Africa = Reunion, Cape Verde, Saint Helena
    df['SecondaryProcessingOriginContinent'][df['SecondaryProcessingOriginCountry'].isin(['Faroe Islands', 'Gibraltar'])] = 'Europe'
    df['SecondaryProcessingOriginContinent'][df['SecondaryProcessingOriginCountry'].isin(['American Samoa', 'French Polynesia', 'New Caledonia', 'Tokelau'])] = 'Oceania'
    df['SecondaryProcessingOriginContinent'][df['SecondaryProcessingOriginCountry'].isin(['Hong Kong'])] = 'Asia'
    df['SecondaryProcessingOriginContinent'][df['SecondaryProcessingOriginCountry'].isin(['Greenland', 'British Virgin Islands', 'Turks and Caicos Islands', 'U.S. Virgin Islands'])] = 'North America'
    df['SecondaryProcessingOriginContinent'][df['SecondaryProcessingOriginCountry'].isin(['Reunion', 'Cape Verde', 'Saint Helena'])] = 'Africa'
    df['SecondaryProcessingOriginContinent'][df['SecondaryProcessingOriginCountry'].isin(['-'])] = '-'
    df['OriginCity'] = df['OriginCity'].fillna('Others')
    df['PrimaryProcessingOriginCity'] = df['PrimaryProcessingOriginCity'].fillna('Others')
    df['StorageCondition'] = df['StorageCondition'].fillna('others')
    df['PackagingType'] = df['PackagingType'].fillna('others')
    corr_matrix = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
    high_corr_var=np.where(corr_matrix>0.8)
    high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
    final_df = df.drop(['ShipmentID','ArrivalDate', 'SubmissionDate','ManufacturerFEINumber','FilerFEINumber','PrimaryProcessedDateTime',
                   'SecondaryProcessedDTTM','CatchDTTM','OriginCountry','PrimaryProcessingOriginCountry','SecondaryProcessingOriginCountry',
                   'FishCommonName','SourceSubCategory','PrimaryProcessingOriginCity','SecondaryProcessingOriginCountry',
                   'SecondaryProcessingOriginContinent','LastInspectionStatus','InspectActive','PackagingType',
                   'FinalDispositionDate', 'PrimaryProcessingOriginContinent'], axis = 1)
    return final_df


data=pd.read_csv("raw_data/seafood_imports.csv")
data=preprocess_data(data)
data.to_csv("preprocessed_data/seafood_imports.csv")
