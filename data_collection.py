import json
import requests as rq
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import zipfile
from ipyleaflet import Map, GeoJSON

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


# collect OSM features
def collect_osm_feature():
    # open the json file of OSM map
    with open('index-v1.json', 'r', encoding='utf-8') as osm_file:
        osm_data = json.load(osm_file)
    # print(len(osm_data['features']))  # 476 record
    # type, Feature
    # properties: 'id', 'parent', 'iso3166-1:alpha2', 'name', urls: ['pbf', 'bz2', 'shp', 'pbf-internal', 'history', 'taginfo', 'updates']
    # geometry: type, coordinates
    features = osm_data['features']

    column = ['type', 'id', 'parent', 'iso3166-2', 'iso3166-1:alpha2', 'name', 'pbf', 'bz2', 'shp', 'pbf-internal',
              'history', 'taginfo', 'updates', 'geometry']

    all_values = []
    for i in range(len(features)):
        values = []
        type = features[i]['type']
        values.append(type)
        for j in ['id', 'parent', 'iso3166-2', 'iso3166-1:alpha2', 'name']:
            values.append(features[i]['properties'][j] if j in features[i]['properties'] else '')
        for k in ['pbf', 'bz2', 'shp', 'pbf-internal', 'history', 'taginfo', 'updates']:
            values.append(features[i]['properties']['urls'][k] if k in features[i]['properties']['urls'] else '')
        geometry = features[i]['geometry']['coordinates']
        values.append(geometry)
        all_values.append(values)

    all_values = pd.DataFrame(all_values, columns=column)
    all_values.to_csv('osm_feature_index.csv', index=False, encoding='utf-8-sig')


def osm_list():
    osm_features = pd.read_csv('osm_feature_index.csv')
    continents = osm_features[osm_features['parent'].isnull()]['id'].values
    # continent = continent_list[0]
    # country_list = osm_features[osm_features['parent'] == continent]['name'].values
    continents_dict = {}
    for continent in continents:
        country_dict = {}
        for country in osm_features[osm_features['parent'] == continent]['name'].values:
            country_detail = osm_features[(osm_features['parent'] == continent) &
                                          (osm_features['name'] == country)]
            country_dict[country_detail.reset_index().loc[0, 'shp']] = country_detail.reset_index().loc[0, 'name']
        continents_dict[continent] = country_dict
    return continents_dict


def osm_feature_download(continent=None, country=None):
    osm_features = pd.read_csv('osm_feature_index.csv')
    # continent_list = osm_features[osm_features['parent'].isnull()]['id'].values
    # country_list = osm_features[osm_features['parent'] == continent]['name'].values

    selected_continent = osm_features[osm_features['parent'] == continent]
    selected_country = selected_continent[selected_continent['name'] == country]
    shapefile_url = selected_country.reset_index().loc[0, 'shp']

    # download the shapefile
    headers = {'User-agent': 'Mozilla/5.0'}
    down_res = rq.get(shapefile_url, headers=headers)
    with open('{}.zip'.format(country), 'wb') as file:
        file.write(down_res.content)

    # os.makedirs(country)
    with zipfile.ZipFile('{}.zip'.format(country), 'r') as zip_ref:
        zip_ref.extractall(country)


# # collect OSM boundary
# def collect_osm_boundary(search_query):
#     headers = {'User-agent': 'Mozilla/5.0'}
#
#     # osm id
#     id_url = "https://nominatim.openstreetmap.org/search?q={}&format=json".format(search_query)
#     r = rq.get(id_url).text
#     osmId = json.loads(r)[0]['osm_id']
#
#     # boundary file
#     url = 'https://osm-boundaries.com/Download/Submit?apiKey=6a162f090cd46e501debec8ce92db0bc&db=osm20240205&osmIds=-{}&format=GeoJSON&srid=4326&includeAllTags'.format( osmId)
#     down_res = rq.get(url, headers=headers)
#     with open('{}.gz'.format('boundary'), 'wb') as file:
#         file.write(down_res.content)
#     g_file = gzip.GzipFile('{}.gz'.format('boundary'))
#     open('{}.geojson'.format('boundary'), "wb+").write(g_file.read())
#     g_file.close()


def osm_feature_json():

    with open('index-v1.json', 'r') as f:
        data = json.load(f)

    m = Map(center=(50.6252978589571, 0.34580993652344), zoom=3, scroll_wheel_zoom=True)
    geo_json = GeoJSON(
        data=data,
        style={
            'opacity': 1, 'dashArray': '9', 'fillOpacity': 0.1, 'weight': 1
        },
        hover_style={
            'color': 'white', 'dashArray': '0', 'fillOpacity': 0.5
        },)
    m.add(geo_json)
    m.save('test.html')


def geojson_visualization():
    boundary = gpd.read_file('boundary.geojson')
    boundary.explore()
    plt.show()


if __name__ == "__main__":
    # collect_osm_boundary(osmId=175905,
    #                      osmName='New York')
    # geojson_visualization()
    # osm_feature_download(continent='africa', country='Angola')
    # collect_osm_boundary(search_query='Singapore')
    # geojson_visualization()
    osm_feature_json()