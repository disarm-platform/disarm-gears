import json


def geojson_encoder_1(dataframe, layer_names=[], lng='lng', lat='lat'):
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

