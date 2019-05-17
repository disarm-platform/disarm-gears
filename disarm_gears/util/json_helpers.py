import json
import pandas as pd


def geojson_encoder_1(dataframe, layer_names=[], lng='lng', lat='lat'):
    #TODO this will be replaced by encoder_3
    '''
    Put dataframe into geojson format

    :param dataframe: Data to put into geojson format.
                      pandas DataFrame.
    :param layer_names:
                        List of strings.
    :param lng: Name of longitude variable within dataframe.
                String, defaults to 'lng'.
    :param lat: Name of latitude variable within dataframe.
                String, defaults to 'lat'.
    '''
    #TODO add tests
    geojson_dict = {'type': 'FeatureCollection', 'features': []}
    for _, row in dataframe.iterrows():
        _feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Point', 'coordinates': []}}
        _feature['geometry']['coordinates'] = [row[lng], row[lat]]
        geojson_dict['features'].append(_feature)

    new_json = {'layer_names': layer_names, 'points': geojson_dict}

    return json.dumps(new_json)


def geojson_encoder_2(dataframe, fields=[], lng='lng', lat='lat', dumps=False):
    '''
    Put dataframe into geojson format

    :param dataframe: Data to put into geojson format.
                      pandas DataFrame.
    :param fields:
                        List of strings.
    :param lng: Name of longitude variable within dataframe.
                String, defaults to 'lng'.
    :param lat: Name of latitude variable within dataframe.
                String, defaults to 'lat'.
    '''
    #TODO add tests
    geojson_dict = {'type': 'FeatureCollection', 'features': []}
    for _, row in dataframe.iterrows():
        _feature = {'type': 'Feature', 'properties': row[fields].to_dict(), 'geometry': {'type': 'Point', 'coordinates': []}}
        _feature['geometry']['coordinates'] = [row[lng], row[lat]]
        geojson_dict['features'].append(_feature)

    result = {'point_data': geojson_dict}

    if dumps:
        result = json.dumps(result)

    return result


def geojson_encoder_3(dataframe, fields=[], layer_names=[], lng='lng', lat='lat', dumps=False):
    '''
    Put dataframe into geojson format

    :param dataframe: Data to put into geojson format.
                      pandas DataFrame.
    :param layer_names:
                        List of strings.
    :param lng: Name of longitude variable within dataframe.
                String, defaults to 'lng'.
    :param lat: Name of latitude variable within dataframe.
                String, defaults to 'lat'.
    '''
    #TODO add tests
    geojson_dict = {'type': 'FeatureCollection', 'features': []}
    for _, row in dataframe.iterrows():
        _feature = {'type': 'Feature', 'properties': row[fields].to_dict(), 'geometry': {'type': 'Point', 'coordinates': []}}
        _feature['geometry']['coordinates'] = [row[lng], row[lat]]
        geojson_dict['features'].append(_feature)

    result = {'layer_names': layer_names, 'points': geojson_dict}

    if dumps:
        result = json.dumps(result)

    return result


def geojson_decoder_1(json_dict):
    '''Takes a json dictionary and puts it into a pandas dataframe'''
    dframe = pd.DataFrame([pi['properties'] for pi in json_dict['features']])
    dframe['lng'] = [pi['geometry']['coordinates'][0] for pi in json_dict['features']]
    dframe['lat'] = [pi['geometry']['coordinates'][1] for pi in json_dict['features']]
    return dframe


