import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd
from category_encoders.one_hot import OneHotEncoder
import requests
import json
import geopandas as gpd
import openai
from tqdm import trange
import warnings
import time

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

class DP(object):
    def __init__(self,
                 housing_columns,
                 address_columns,
                 price_column,
                 property_data,
                 ordinal_columns=None,
                 ordinal_rankings=None,
                 reorganize_columns=None,
                 reorganize_codes=None
                 ):
        self.housing = housing_columns
        self.address = address_columns
        self.price = price_column
        self.all_columns = housing_columns + address_columns + price_column
        self.data = property_data
        self.ordinal_columns = ordinal_columns
        self.ordinal_rankings = ordinal_rankings
        self.reorganize_columns = reorganize_columns
        self.reorganize_codes = reorganize_codes

    def clean_data(self):
        # delete empty values
        data = self.data[self.all_columns].dropna()

        # delete the values with 0 price
        data = data[data[self.price[0]] != 0]

        # delete non-sense values
        nonsense_characters = ['/', '-']
        data = data[~data.apply(lambda x: x.isin(nonsense_characters)).any(axis=1)]

        # delete column with only one unique value
        single_value_columns = []

        for column in self.all_columns:
            column_unique_value = len(data[column].unique())
            if column_unique_value == 1:
                single_value_columns.append(column)
                data.drop(columns=single_value_columns, inplace=True, axis=1)

        for column in single_value_columns:
            self.all_columns.remove(column)

        data = self.data[self.all_columns]

        # Reorganize the columns
        if self.reorganize_columns:
            data = column_reorganization(data, self.reorganize_codes, self.reorganize_columns)

        # check whether the values in the column are text-most and numerical-most
        number_columns = []
        text_columns = []

        for column in self.all_columns:
            str_count = 0
            non_str_count = 0

            for value in data[column]:
                if type(value) is str:
                    str_count += 1
                else:
                    non_str_count += 1

            if str_count > 0.9 * len(data):  # column with text values
                text_columns.append(column)

            else:
                number_columns.append(column)

        # number columns: use boxplot to detect and delete outliers
        for number_column in number_columns:
            if number_column in self.housing + self.price:
                q1 = data[number_column].quantile(0.25)
                q3 = data[number_column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 0.5 * iqr
                upper_bound = q3 + 0.5 * iqr
                data = data[(data[number_column] <= upper_bound) & (
                        data[number_column] >= lower_bound)].reset_index().drop(columns='index')

                column_unique_value = len(data[number_column].unique())
                if column_unique_value == 1:
                    data.drop(columns=number_column, inplace=True)

        # text columns: data encoding for text columns
        for text_column in text_columns:
            column_unique_value = len(data[text_column].unique())

            # delete columns with single value
            if column_unique_value == 1:
                data.drop(columns=text_column, inplace=True)

            # label encoding
            if column_unique_value == 2:
                data[text_column + '_encoding'] = data.apply(
                    lambda x: 0 if x[text_column] == data[text_column].unique()[0] else 1,
                    axis=1)
                data.drop(columns=text_column, inplace=True)

            # one-hot encoding
            elif 3 <= column_unique_value <= 5:
                encoder = OneHotEncoder().fit(data[text_column])
                encoded_data = encoder.transform(data[text_column])
                encoded_data.columns = [text_column + '_' + str(i) for i in data[text_column].unique()]
                data = pd.concat([data, encoded_data], axis=1)
                data.drop(columns=text_column, inplace=True)
            else:
                continue

        # ordinal columns
        # reorganize columns
        if self.ordinal_columns:
            data = column_transformation(data=data,
                                         columns=self.ordinal_columns,
                                         rankings=self.ordinal_rankings)

        # Price in the last column
        data = data[[col for col in data.columns if col != self.price[0]] + [self.price[0]]]
        return data

# ordinal encoding: transform ordinal columns and reorganize
def column_transformation(data, rankings, columns):
    for i, column in enumerate(columns):
        data = data[data[column].isin(rankings[i])]
        data[column] = data[column].apply(lambda x: rankings[i].index(x))
    return data

def gpt_transformation(message):
    # Deepseek API will be supported
    openai.api_type = "azure"
    openai.api_base = "https://hkust.azure-api.net"
    openai.api_version = "2023-05-15"
    openai.api_key = "132de8db856847a69cdd44107437d1a7"

    response = openai.ChatCompletion.create(
        engine='gpt-35-turbo',
        temperature=1,
        messages=[
            {"role": "system", "content": 'I need you to write a Python function called reorganize with two parameters data and column. '
                                          'Please show the Python codes without any further explanation.'},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0]["message"]["content"]

# reorganize target column using GPT-based codes
def column_reorganization(data, codes, columns):
    for i, column in enumerate(columns):
        exec(codes[i][0])
    # code = 'data[column] = data[column].apply(lambda x: "freehold" if "freehold" in x.lower() else "leasehold")'
    return data


def address_transformation(data, street, city, property=None, google_api_key=None):
    if property:
        unique_addresses = data[[street, property]].drop_duplicates().reset_index(drop=True)
    else:
        unique_addresses = data[[street]].drop_duplicates().reset_index(drop=True)

    for i in trange(len(unique_addresses)):

        if property:
            address_query = '{},{},{}'.format(unique_addresses.loc[i, property],
                                              unique_addresses.loc[i, street], city)
        else:
            address_query = '{},{}'.format(unique_addresses.loc[i, street], city)

        if google_api_key: # Use Google Map API
            try:
                url = "https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}".format(address_query,
                                                                                                             google_api_key)
                r = json.loads(requests.get(url).text)
                latitude = r['geometry']['location']['lat']
                longitude = r['geometry']['location']['lng']
            except Exception:
                latitude = ''
                longitude = ''

        else: # Use Nominatim Map API
            try:
                url = "https://nominatim.openstreetmap.org/search?amenity={}&street={}&city={}&format=json".format(unique_addresses.loc[i, property],
                                                                                                                   unique_addresses.loc[i, street], city)
                headers = {'Referer':'https://nominatim.org/',
                           'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'}
                r = json.loads(requests.get(url=url, headers=headers).text)
                latitude = r[0]['lat']
                longitude = r[0]['lon']
            except Exception:
                latitude = ''
                longitude = ''

        # attach the geocoded values
        unique_addresses.loc[i, 'Latitude'] = latitude
        unique_addresses.loc[i, 'Longitude'] = longitude
        time.sleep(1)

    # drop the address query column
    if property:
        result_df = pd.merge(data, unique_addresses, on=[street, property], how='left')
        result_df = result_df.drop([street, property], axis=1)
    else:
        result_df = pd.merge(data, unique_addresses, on=[street], how='left')
        result_df = result_df.drop([street], axis=1)
    return result_df

# visualization of housing transaction data
def data_visualization(housing, boundary, show_column):
    city_housing = gpd.GeoDataFrame(housing,
                                    geometry=gpd.points_from_xy(housing.Longitude, housing.Latitude))

    # ny_housing.apply(lambda row: folium.CircleMarker(location=[row["Latitude"], row["Longitude"]],
    #                                                  color='red',
    #                                                  fill=True,
    #                                                  fill_opacity=0.6,
    #                                                  radius=1).add_to(m),
    #                                                  axis=1)

    # m.save('ny.html')
    ax = boundary.plot(figsize=(10, 8), color='gray', alpha=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cx.add_basemap(ax, source=cx.providers.Esri.WorldTopoMap, crs="EPSG:4326")
    city_housing.plot(ax=ax, markersize=2, legend=True, column=show_column,
                      legend_kwds={"label": show_column, 'orientation': 'horizontal'})
    plt.show()


if __name__ == "__main__":
    # Singapore
    data = pd.read_csv('./paper_test/sg_test/singapore.csv')

    pta_columns = ['Area (SQFT)', 'Number of Units', 'Property Type', 'Tenure', 'Type of Sale', 'Type of Area',
                   'Floor Level']
    address_columns = ['Project Name', 'Street Name', 'Postal District', 'Market Segment']  # estate-street-city
    price_column = ['Transacted Price ($)']

    df = DP(housing_columns=pta_columns,
            address_columns=address_columns,
            price_column=price_column,
            property_data=data,
            ordinal_columns=['Floor Level'],
            ordinal_rankings=[['B1 to B5', '01 to 05', '06 to 10',
                              '11 to 15', '16 to 20', '21 to 25',
                              '26 to 30', '31 to 35', '36 to 40',
                              '41 to 45', '46 to 50', '51 to 55',
                              '56 to 60', '61 to 65', '66 to 70']],
            reorganize_columns=['Tenure'],
            reorganize_codes=[['data[column] = data[column].apply(lambda x: "freehold" if "freehold" in x.lower() else "leasehold")']]).clean_data()

    # New York
    # pta_columns = ['RESIDENTIAL UNITS',
    #                'COMMERCIAL UNITS',
    #                'GROSS SQUARE FEET',
    #                'YEAR BUILT',
    #                'TAX CLASS AT TIME OF SALE',
    #                 'BUILDING CLASS AT TIME OF SALE']
    #
    # address_columns = ['BOROUGH', 'BLOCK', 'LOT', 'ADDRESS', 'ZIP CODE']  # estate-street-city
    # price_column = ['SALE PRICE']


    # London
    # pta_columns = ['Property Type', 'Old/New', 'Duration', 'PPD Category Type']
    # address_columns = ['PAON', 'SAON', 'Street']  # estate-street-city
    # price_column = ['Price']

    # a = address_transformation(data=data, property='Project Name', street='Street Name', city='Singapore')




