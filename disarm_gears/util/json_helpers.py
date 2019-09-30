#import json
#import pandas as pd
#
##TODO unify terms: 'points', 'point_data', 'region_definition', etc.
#
#def geojson_encoder_2(dataframe, fields=[], lng='lng', lat='lat', dumps=False):
#    '''
#    Serialize a dataframe as GEOJSON format
#
#    :param dataframe: Data source to serialize.
#                      Pandas DataFrame.
#    :param fields: Fields from dataframe to encode.
#                   List of strings.
#    :param lng: Name of longitude variable within dataframe.
#                String, defaults to 'lng'.
#    :param lat: Name of latitude variable within dataframe.
#                String, defaults to 'lat'.
#    :param dumps: Whether to return a GEOJSON formatted string (True) or a GEOJSON formatted stream.
#                  Boolean (default False).
#
#    :return: Dictionary (if dumps = False) or string if (dumps = True)
#    '''
#    #TODO add tests
#    geojson_dict = {'type': 'FeatureCollection', 'features': []}
#    for _, row in dataframe.iterrows():
#        _feature = {'type': 'Feature', 'properties': row[fields].to_dict(), 'geometry': {'type': 'Point', 'coordinates': []}}
#        _feature['geometry']['coordinates'] = [row[lng], row[lat]]
#        geojson_dict['features'].append(_feature)
#
#    result = {'point_data': geojson_dict}
#
#    if dumps:
#        result = json.dumps(result)
#
#    return result
#
#
#def geojson_encoder_3(dataframe, fields=[], layer_names=[], lng='lng', lat='lat', dumps=False):
#    '''
#    Serialize a dataframe as GEOJSON format
#
#    :param dataframe: Data source to serialize.
#                      Pandas DataFrame.
#    :param fields: Fields from dataframe to encode.
#                   List of strings.
#    :param layer_names: Additional fields to include into the GEOJSON.
#                        List of strings.
#    :param lng: Name of longitude variable within dataframe.
#                String, defaults to 'lng'.
#    :param lat: Name of latitude variable within dataframe.
#                String, defaults to 'lat'.
#    :param dumps: Whether to return a JSON formatted string (True) or a JSON formatted stream.
#                  Boolean (default False).
#
#    :return: Dictionary (if dumps = False) or string if (dumps = True)
#    '''
#
#    #TODO add tests
#    geojson_dict = {'type': 'FeatureCollection', 'features': []}
#    for _, row in dataframe.iterrows():
#        _feature = {'type': 'Feature', 'properties': row[fields].to_dict(), 'geometry': {'type': 'Point', 'coordinates': []}}
#        _feature['geometry']['coordinates'] = [row[lng], row[lat]]
#        geojson_dict['features'].append(_feature)
#
#    result = {'layer_names': layer_names, 'points': geojson_dict}
#
#    if dumps:
#        result = json.dumps(result)
#
#    return result


def geojson_decoder_1(json_dict):
    '''
    Deserialize an instance containing a GEOJSON

    :param json_dict: Instance containg the GEOJSON data.
                      Dictionary.

    :return: Pandas DataFrame.
    '''
    dframe = pd.DataFrame([pi['properties'] for pi in json_dict['features']])
    dframe['lng'] = [pi['geometry']['coordinates'][0] for pi in json_dict['features']]
    dframe['lat'] = [pi['geometry']['coordinates'][1] for pi in json_dict['features']]
    return dframe


