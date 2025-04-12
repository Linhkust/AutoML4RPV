# -*- coding: utf-8 -*-
import geopandas as gpd
import osmnx as ox
import shapely
import pandas as pd
import warnings
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import Point, Polygon
from geopy import distance
from pyogrio import read_dataframe
import taxicab as tc
import networkx as nx
import multiprocessing
from tqdm import tqdm
import os
from tqdm import trange
import zipfile

warnings.filterwarnings("ignore")
proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

def geodesic_point_buffer(lat, lon, buffer):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(buffer)  # distance in metres
    return transform(project, buf).exterior.coords[:]

# Unzip the OSM .zip file into a folder
def unzip(file_name):
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(file_name))

    # new_path = os.path.join(os.path.dirname(file_name), file_name[file_name.rfind('/')+1:file_name.rfind('.')] + '1')
    # for file in f.namelist():
    #     f.extract(file, str(new_path))  # 解压位置
    # f.close()
    return os.path.join(os.path.dirname(file_name), file_name[file_name.rfind('/')+1:file_name.rfind('.')])

def boundary_import(boundary_file):
    return read_dataframe(boundary_file, encoding='utf-8',errors='ignore')

def osm_point_features(osm_path, boundary):
    # point features and combine
    poi = gpd.clip(read_dataframe(os.path.join(osm_path, 'gis_osm_pois_free_1.shp')).to_crs("EPSG:4326"), boundary)
    # pofw = gpd.clip(read_dataframe('{}/gis_osm_pofw_free_1.shp'.format(osm_path)).to_crs("EPSG:4326"), boundary)
    natural = gpd.clip(read_dataframe(os.path.join(osm_path, 'gis_osm_natural_free_1.shp')).to_crs("EPSG:4326"), boundary)
    traffic = gpd.clip(read_dataframe(os.path.join(osm_path, 'gis_osm_traffic_free_1.shp')).to_crs("EPSG:4326"), boundary)
    transport = gpd.clip(read_dataframe(os.path.join(osm_path, 'gis_osm_transport_free_1.shp')).to_crs("EPSG:4326"),
                         boundary)

    point_features = gpd.GeoDataFrame(pd.concat([poi, natural, traffic, transport]))
    return point_features


def osm_line_features(osm_path, boundary):
    # line features and combine

    roads = gpd.clip(read_dataframe('{}/gis_osm_roads_free_1.shp'.format(osm_path)).to_crs("EPSG:4326"), boundary)
    railways = gpd.clip(read_dataframe('{}/gis_osm_railways_free_1.shp'.format(osm_path)).to_crs("EPSG:4326"),
                        boundary)
    # waterways
    # waterways = gpd.clip(read_dataframe('{}/gis_osm_waterways_free_1.shp'.format(osm_path)).to_crs("EPSG:4326"),
    #                      boundary)
    # line_features = gpd.GeoDataFrame(pd.concat([roads, railways, waterways]))

    line_features = gpd.GeoDataFrame(pd.concat([roads, railways]))
    return line_features


def osm_polygon_features(osm_path, boundary):
    # area features and combine
    buildings = gpd.clip(read_dataframe('{}/gis_osm_buildings_a_free_1.shp'.format(osm_path)).to_crs("EPSG:4326"),
                         boundary)
    pois = gpd.clip(read_dataframe('{}/gis_osm_pois_a_free_1.shp'.format(osm_path)).to_crs("EPSG:4326"), boundary)
    water = gpd.clip(read_dataframe('{}/gis_osm_water_a_free_1.shp'.format(osm_path)).to_crs("EPSG:4326"), boundary)

    area_features = gpd.GeoDataFrame(pd.concat([pois, water]))
    return buildings, area_features

def feature_save(zip_file, boundary, saved_path):
    osm_path = unzip(zip_file)

    point_features = osm_point_features(osm_path=osm_path, boundary=boundary)
    line_features = osm_line_features(osm_path=osm_path, boundary=boundary)
    polygon_features = osm_polygon_features(osm_path=osm_path, boundary=boundary)

    points = point_features[point_features.geometry.type == 'Point'].copy().reset_index(drop=True)
    lines = line_features[line_features.geometry.type == 'LineString'].copy().reset_index(drop=True)
    building = polygon_features[0][polygon_features[0].geometry.type == 'Polygon'].copy().reset_index(drop=True)
    area = polygon_features[1][polygon_features[1].geometry.type == 'Polygon'].copy().reset_index(drop=True)

    # saved to local disk
    folder = os.path.join(os.path.dirname(zip_file), 'features')
    os.makedirs(folder)

    points.to_file(saved_path + '/features' + '/points.shp')
    lines.to_file(saved_path + '/features' + '/lines.shp')
    building.to_file(saved_path + '/features' + '/building.shp')
    area.to_file(saved_path + '/features' + '/area.shp')
    # return point_features, line_features, polygon_features

# For Point Features
def point_features_(housing,
                    row,
                    pois,
                    point_features,
                    dist=400,
                    dist_type='straight'):
    """Parameters"""
    # housing: housing transaction data
    # dist: int, distance threshold
    # dist_type: straight for Euclidean distance or network for Network distance
    # density_pois: pois used to calculate density
    # diversity_pois: pois used to calculate density
    # accessibility_pois: pois used to calculate density

    global point_density, new_walking_network
    global point_diversity
    global point_accessibility

    # using Network distance
    if dist_type == 'network':
        walking_network = ox.graph_from_point((housing.loc[row, 'Latitude'], housing.loc[row, 'Longitude']),
                                              dist_type='network',
                                              dist=dist,
                                              network_type='walk',
                                              simplify=True)
        walking_network = ox.project_graph(walking_network, to_crs="EPSG:4326")
        gdf_nodes = ox.graph_to_gdfs(walking_network, edges=False, nodes=True)

        # measurement area
        gdf_nodes = gdf_nodes.unary_union.convex_hull
        housing_point_features = point_features.clip(gdf_nodes)

        generated_features = {}

        # percentage of specific types of POI
        point_density = {}
        for poi in pois:
            selected_poi = housing_point_features[housing_point_features['fclass'] == poi]
            point_density['{}_num'.format(poi)] = len(selected_poi)
        point_density['poi_num'] = len(housing_point_features)
        generated_features = {**generated_features, **point_density}

        # number of point feature types
        # if diversity:
        point_diversity = {'type_num': len(housing_point_features['fclass'].unique())}
        generated_features = {**generated_features, **point_diversity}

        # nearest network distance
        point_accessibility = {}
        for poi in pois:
            # increase search range
            init_dist = dist
            selected_poi = []
            while len(selected_poi) == 0:
                new_walking_network = ox.graph_from_point(
                    (housing.loc[row, 'Latitude'], housing.loc[row, 'Longitude']),
                    dist_type='network',
                    dist=init_dist,
                    network_type='walk',
                    simplify=True)
                new_gdf_nodes = ox.graph_to_gdfs(new_walking_network, edges=False, nodes=True)
                new_gdf_nodes = new_gdf_nodes.unary_union.convex_hull
                new_housing_point_features = point_features.clip(new_gdf_nodes)
                selected_poi = new_housing_point_features[new_housing_point_features['fclass'] == poi]
                init_dist += 500

            selected_poi = selected_poi.reset_index().drop('index', axis=1)
            try:
                selected_poi['distance'] = selected_poi.apply(lambda x: tc.distance.shortest_path(
                    new_walking_network,
                    (housing.loc[row, 'Latitude'],
                     housing.loc[row, 'Longitude']),
                    (x['geometry'].y, x['geometry'].x))[0], axis=1)

            except IndexError:
                housing_point_node = ox.nearest_nodes(new_walking_network,
                                                      housing.loc[row, 'Longitude'],
                                                      housing.loc[row, 'Latitude'])

                selected_poi['nodes'] = selected_poi.apply(lambda x: ox.nearest_nodes(new_walking_network,
                                                                                      x['geometry'].y,
                                                                                      x['geometry'].x),
                                                           axis=1)

                selected_poi['distance'] = selected_poi.apply(lambda x: nx.shortest_path_length(new_walking_network,
                                                                                                housing_point_node,
                                                                                                x['nodes'],
                                                                                                weight='length'),
                                                              axis=1)
            point_accessibility['{}_Network_dist'.format(poi, dist_type)] = "%.1f" % selected_poi['distance'].min()
            generated_features = {**generated_features, **point_accessibility}

        return generated_features

    # using Euclidean distance
    else:
        # measurement area
        housing_buffer = geodesic_point_buffer(housing.loc[row, 'Latitude'],
                                               housing.loc[row, 'Longitude'],
                                               buffer=dist)
        housing_buffer = Polygon(tuple(housing_buffer))
        # ax = gpd.GeoSeries(housing_buffer).plot(figsize=(10, 10), color='blue', alpha=0.5)

        housing_point_features = point_features.clip(housing_buffer)
        # cx.add_basemap(ax, source=cx.providers.Esri.WorldTopoMap, crs="EPSG:4326")
        # plt.show()

        generated_features = {}
        # percentage of specific types of POI
        point_density = {}
        for poi in pois:
            selected_poi = housing_point_features[housing_point_features['fclass'] == poi]
            point_density['{}_num'.format(poi)] = len(selected_poi)
        point_density['poi_num'] = len(housing_point_features)
        generated_features = {**generated_features, **point_density}

        # number of point feature types
        point_diversity = {'type_num': len(housing_point_features['fclass'].unique())}
        generated_features = {**generated_features, **point_diversity}

        # nearest network distance
        # increase buffer area to narrow down the search area
        point_accessibility = {}
        for poi in pois:
            # increase search range
            init_dist = dist
            selected_poi = []
            while len(selected_poi) == 0:
                new_housing_buffer = geodesic_point_buffer(housing.loc[row, 'Latitude'],
                                                           housing.loc[row, 'Longitude'],
                                                           buffer=init_dist)
                new_housing_buffer = Polygon(tuple(new_housing_buffer))
                new_housing_point_features = point_features.clip(new_housing_buffer)
                selected_poi = new_housing_point_features[new_housing_point_features['fclass'] == poi]
                init_dist += dist

            selected_poi = selected_poi.reset_index().drop('index', axis=1)
            # taxicab to calculate the distance
            selected_poi['distance'] = selected_poi.apply(lambda x: distance.distance(
                (housing.loc[row, 'Latitude'],
                 housing.loc[row, 'Longitude']),
                (x['geometry'].y, x['geometry'].x)).m, axis=1)

            point_accessibility['{}_Euclidean_dist'.format(poi)] = "%.1f" % selected_poi['distance'].min()
            generated_features = {**generated_features, **point_accessibility}
        return generated_features


# For line features
def line_features_(housing,
                   row,
                   lines,
                   line_features):
    global line_dist

    line_dist = {}
    for line in lines:
        # increase search range
        init_dist = 500
        selected_line = []
        while len(selected_line) == 0:
            new_housing_buffer = geodesic_point_buffer(housing.loc[row, 'Latitude'],
                                                       housing.loc[row, 'Longitude'],
                                                       buffer=init_dist)
            new_housing_buffer = Polygon(tuple(new_housing_buffer))
            new_housing_line_features = line_features.clip(new_housing_buffer)
            selected_line = new_housing_line_features[new_housing_line_features['fclass'] == line]
            init_dist += 500

        selected_line = selected_line.to_crs(3857)
        housing_point = gpd.GeoDataFrame(geometry=[Point(housing.loc[row, 'Longitude'],
                                                         housing.loc[row, 'Latitude'])], crs=4326).to_crs(3857)

        selected_line = selected_line.reset_index()
        selected_line['distance'] = selected_line.apply(lambda x: shapely.distance(x['geometry'],
                                                                                   housing_point.loc[0, 'geometry']),
                                                        axis=1)

        line_dist['{}_dist'.format(line)] = "%.2f" % selected_line['distance'].min()
    return line_dist


# For polygon features
def polygon_features_(housing,
                      row,
                      polygons,
                      polygon_features,
                      dist=1200,
                      dist_type='straight'):

    global new_walking_network
    geod = pyproj.Geod(ellps='WGS84')

    # Euclidean distance
    if dist_type == 'straight':
        generated_features = {}

        # calculating building area ratio
        housing_buffer = geodesic_point_buffer(housing.loc[row, 'Latitude'],
                                               housing.loc[row, 'Longitude'],
                                               buffer=dist)

        if dist > 0:
            housing_buffer = Polygon(tuple(housing_buffer))
            selected_buildings = polygon_features[0].clip(housing_buffer)
            selected_buildings['area'] = selected_buildings.apply(lambda x:
                                                                  abs(geod.geometry_area_perimeter(x['geometry'])[0]),
                                                                  axis=1)
            housing_buffer_area = geod.geometry_area_perimeter(housing_buffer)[0]
            building_density = {'building_density': selected_buildings['area'].sum() / abs(housing_buffer_area)}
            generated_features = {**generated_features, **building_density}
        else:
            pass

        polygon_accessibility = {}
        for polygon in polygons:
            selected_polygons = []
            init_dist = dist
            while len(selected_polygons) == 0:
                new_housing_buffer = geodesic_point_buffer(housing.loc[row, 'Latitude'],
                                                           housing.loc[row, 'Longitude'],
                                                           buffer=init_dist)
                new_housing_buffer = Polygon(tuple(new_housing_buffer))
                new_housing_polygon_features = polygon_features[1].clip(new_housing_buffer)

                selected_polygons = new_housing_polygon_features[new_housing_polygon_features['fclass'] == polygon]
                init_dist += dist

            selected_polygons = selected_polygons.to_crs(3857)
            housing_point = gpd.GeoDataFrame(geometry=[Point(housing.loc[row, 'Longitude'],
                                                             housing.loc[row, 'Latitude'])], crs=4326).to_crs(3857)

            selected_polygons = selected_polygons.reset_index().drop('index', axis=1)
            selected_polygons['distance'] = selected_polygons.apply(lambda x: shapely.distance(x['geometry'],
                                                                                               housing_point.loc[
                                                                                                   0, 'geometry']),
                                                                    axis=1)
            polygon_accessibility['{}_Euclidean_dist'.format(polygon)] = "%.2f" % selected_polygons[
                'distance'].min()

        generated_features = {**generated_features, **polygon_accessibility}
        return generated_features

    # Network distance
    else:
        walking_network = ox.graph_from_point((housing.loc[row, 'Latitude'], housing.loc[row, 'Longitude']),
                                              dist_type='network',
                                              dist=dist,
                                              network_type='walk',
                                              simplify=True)
        walking_network = ox.project_graph(walking_network, to_crs="EPSG:4326")
        gdf_nodes = ox.graph_to_gdfs(walking_network, edges=False, nodes=True)
        gdf_nodes = gdf_nodes.unary_union.convex_hull

        generated_features = {}

        selected_buildings = polygon_features[0].clip(gdf_nodes)
        selected_buildings['area'] = selected_buildings.apply(lambda x:
                                                              abs(geod.geometry_area_perimeter(x['geometry'])[0]),
                                                              axis=1)
        housing_buffer_area = geod.geometry_area_perimeter(gdf_nodes)[0]
        building_density = {'building_density': selected_buildings['area'].sum() / abs(housing_buffer_area)}
        generated_features = {**generated_features, **building_density}

        polygon_accessibility = {}
        for polygon in polygons:
            # increase search range
            init_dist = dist
            selected_polygons = []
            while len(selected_polygons) == 0:
                new_walking_network = ox.graph_from_point(
                    (housing.loc[row, 'Latitude'], housing.loc[row, 'Longitude']),
                    dist_type='network',
                    dist=init_dist,
                    network_type='walk',
                    simplify=True)

                new_gdf_nodes = ox.graph_to_gdfs(new_walking_network, edges=False, nodes=True)
                new_gdf_nodes = new_gdf_nodes.unary_union.convex_hull

                new_housing_polygon_features = polygon_features[1].clip(new_gdf_nodes)
                selected_polygons = new_housing_polygon_features[new_housing_polygon_features['fclass'] == polygon]
                init_dist += dist

            selected_polygons = selected_polygons.reset_index().drop('index', axis=1)
            selected_polygons['nearest_node'] = selected_polygons.apply(lambda x:
                                                                        ox.distance.nearest_nodes(
                                                                            new_walking_network,
                                                                            x['geometry'].exterior.coords.xy[0],
                                                                            x['geometry'].exterior.coords.xy[1],
                                                                            return_dist=True),
                                                                        axis=1)

            min_dist = []

            # taxicab
            try:
                for i in range(len(selected_polygons)):
                    for j, node_id in enumerate(selected_polygons.loc[i, 'nearest_node'][0]):
                        lon = new_walking_network.nodes[node_id]['x']  # lon
                        lat = new_walking_network.nodes[node_id]['y']  # lat
                        walking_distance = tc.distance.shortest_path(new_walking_network,
                                                                     (housing.loc[row, 'Latitude'],
                                                                      housing.loc[row, 'Longitude']),
                                                                     (lat, lon))[0]
                        min_dist.append(walking_distance + selected_polygons.loc[i, 'nearest_node'][1][j])

            # taxicab not working, using networkx
            except Exception:
                housing_point_node = ox.nearest_nodes(new_walking_network,
                                                      housing.loc[row, 'Longitude'],
                                                      housing.loc[row, 'Latitude'])
                for i in range(len(selected_polygons)):
                    for j, node_id in enumerate(selected_polygons.loc[i, 'nearest_node'][0]):
                        walking_distance = nx.shortest_path_length(new_walking_network,
                                                                   housing_point_node,
                                                                   node_id,
                                                                   weight='length')
                        min_dist.append(walking_distance)

            polygon_accessibility['{}_Network_dist'.format(polygon)] = "%.2f" % min(min_dist)
            generated_features = {**generated_features, **polygon_accessibility}

        return generated_features

def feature_type(osm_path, boundary):
    point_features = osm_point_features(osm_path=osm_path, boundary=boundary)
    line_features = osm_line_features(osm_path=osm_path, boundary=boundary)
    polygon_features = osm_polygon_features(osm_path=osm_path, boundary=boundary)
    return point_features, line_features, polygon_features

def osm_feature_generation(data, row,
                           point_features,
                           line_features,
                           polygon_features,
                           point_dist_type,
                           point_dist,
                           polygon_dist_type,
                           polygon_dist,
                           pois,
                           lines=None,
                           polygons=None):

    point_attributes = point_features_(housing=data,
                                   row=row,
                                   pois=pois,
                                   dist_type=point_dist_type,
                                   dist=point_dist,
                                   point_features=point_features)
    if lines is not None:

        line_attributes = line_features_(housing=data,
                                       row=row,
                                       lines=lines,
                                       line_features=line_features)

    if polygons is not None:
        polygon_attributes = polygon_features_(housing=data,
                                             row=row,
                                             polygons=polygons,
                                             polygon_features=polygon_features,
                                             dist_type=polygon_dist_type,
                                             dist=polygon_dist)

    return {**point_attributes, **line_attributes, **polygon_attributes}


def parallel_feature_generation(data, folder, boundary, pois, lines, polygons,
                                point_dist_type,
                                point_dist,
                                polygon_dist_type,
                                polygon_dist,
                                n_jobs):
    # extract the unique latitude and longitude
    data = data[['Latitude','Longitude']].drop_duplicates().reset_index(drop=True)

    point_features = feature_type(folder, boundary)[0]
    line_features = feature_type(folder, boundary)[1]
    polygon_features = feature_type(folder, boundary)[2]

    cpu_num = range(1, multiprocessing.cpu_count() + 1)
    processor_num = cpu_num[n_jobs - 1] if n_jobs > 0 else cpu_num[n_jobs]

    # Testing
    data_count = len(data)
    # print(data_count)
    # data_count = 10
    pbar = tqdm(total=data_count)
    pbar.set_description('OSM Feature Generating:')
    update = lambda *args: pbar.update()

    # VERY IMPORTANT: check how many cores in your PC
    pool = multiprocessing.Pool(processes=processor_num)

    results = []

    # Parallel computation
    for num in range(data_count):
        # 将任务提交给进程池
        result = pool.apply_async(osm_feature_generation,
                                  args=(data,
                                        num,
                                        point_features,
                                        line_features,
                                        polygon_features,
                                       point_dist_type,
                                       point_dist,
                                       polygon_dist_type,
                                       polygon_dist,
                                       pois,
                                       lines,
                                       polygons,),
                                  callback=update)
        results.append(result)

    pool.close()
    pool.join()

    pred_results = []

    for result in results:
        pred_results.append(result.get())

    pred_results = pd.DataFrame(pred_results)
    saved_path = os.path.dirname(data) + '\\features.csv'
    pred_results.to_csv(saved_path, index=False)
    return pred_results


def stream_feature_generation(data,
                              osm_path,
                              boundary,
                              pois,
                              lines,
                              polygons,
                              point_dist_type,
                              point_dist,
                              polygon_dist_type,
                              polygon_dist,
                              saved_path):

    point_features = feature_type(osm_path, boundary)[0]
    line_features = feature_type(osm_path, boundary)[1]
    polygon_features = feature_type(osm_path, boundary)[2]

    results = []

    # Serial computation
    for num in trange(len(data)):
        result = osm_feature_generation(data,
                                        num,
                                        point_features,
                                        line_features,
                                        polygon_features,
                                        point_dist_type,
                                        point_dist,
                                        polygon_dist_type,
                                        polygon_dist,
                                        pois,
                                        lines,
                                        polygons)
        results.append(result)
    results = pd.DataFrame(results)
    results.to_csv(saved_path, index=False)
    return results

def statistics(data):
    data = data.describe().T
    data=data.round(2)
    data['variable'] = data.index
    col = data.pop('variable')
    data.insert(loc=0, column='variable', value=col)
    return data

def combine(preparation, features, saved_path):
    data = pd.concat([features, preparation], axis=1)
    data.to_csv(saved_path, index=False)
    return data

if __name__ == "__main__":
    # print(statistics(data=pd.read_csv('./paper_test/sg_test/preparation_test.csv')))
    # feature_generation(data=pd.read_csv('./test/singapore_clean_encoding_coords_corrected.csv'),
    #                    boundary=boundary_import(boundary_file='./test/singapore.geojson'),
    #                    folder='./test/singapore',
    #                    pois=['bus_stop', 'railway_station', 'supermarket', 'mall', 'hospital'],
    #                    dist_type='straight',
    #                    dist=800,
    #                    lines=['motorway', 'residential', 'service'],
    #                    polygons=['park', 'school'])
    # result1 = point_features_(
    #                 housing=pd.read_csv('./test/singapore_clean_encoding_coords_corrected.csv'),
    #                 row=2,
    #                 pois=['bus_stop', 'railway_station', 'supermarket', 'mall', 'hospital'],
    #                 point_features=file[0],
    #                 dist=400,
    #                 dist_type='straight')

    # result2 = line_features_(housing=pd.read_csv('./test/singapore_clean_encoding_coords_corrected.csv'),
    #                          row=0,
    #                          lines=['motorway', 'residential'],
    #                          line_features=file[1])

    # result3 = polygon_features_(housing=pd.read_csv('./test/singapore_clean_encoding_coords_corrected.csv'),
    #                   row=1,
    #                   polygons=['park', 'school'],
    #                   polygon_features=file[2],
    #                   dist=400,
    #                   dist_type='straight')

    # data = pd.read_csv('./paper_test/lo_test/preparation.csv')
    # unique_addresses = data[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)

    # feature_type(folder='./paper_test/lo_test/london_osm',
    #              boundary=boundary_import(boundary_file='./paper_test/lo_test/london.geojson'),
    #                                       saved_path='./paper_test/lo_test')

    # un_zip(file_name='./paper_test/sg_test/singapore_osm.zip')
    osm_zip = './paper_test/sg_test/singapore_osm.zip'
    osm_path = unzip(osm_zip)

    # poi = gpd.clip(read_dataframe(os.path.join(osm_path, 'gis_osm_pois_free_1.shp')).to_crs("EPSG:4326"), boundary)

    # result = parallel_feature_generation(data=unique_addresses,
    #                                      boundary=boundary_import(boundary_file='./paper_test/lo_test/london.geojson'),
    #                                      folder='./paper_test/lo_test/london_osm',
    #                                      pois=['bus_stop', 'railway_station', 'supermarket', 'mall', 'hospital', 'park'],
    #                                      point_dist_type='straight',
    #                                      point_dist=1000,
    #                                      polygon_dist=1000,
    #                                      polygon_dist_type='straight',
    #                                      lines=['primary', 'rail'],
    #                                      polygons=['water'],
    #                                      n_jobs=6)


    # result = stream_feature_generation(data=pd.read_csv('./test/singapore_clean_encoding_coords_corrected.csv'),
    #                                      boundary=boundary_import(boundary_file='./test/singapore.geojson'),
    #                                      folder='./test/singapore',
    #                                      pois=['bus_stop', 'railway_station', 'supermarket', 'mall', 'hospital'],
    #                                      point_dist_type='straight',
    #                                      point_dist=800,
    #                                      polygon_dist=1000,
    #                                      polygon_dist_type='straight',
    #                                      lines=['motorway', 'residential', 'service'],
    #                                      polygons=['park', 'school'],
    #                                    saved_path='./paper_test/')