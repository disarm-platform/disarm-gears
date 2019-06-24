#import json
#import pandas as pd
#from disarm_gears.util import json_helpers
#
#
#def test_geojson_encoder_3():
#
#    df = pd.DataFrame({'lng': [1.1, 1.2, 1.3], 'lat': [-2.3, -2.0, -3.3]})
#
#    # dumps = True
#    js = json_helpers.geojson_encoder_3(df, layer_names=['layer_1', 'layer_2'], lng='lng', lat='lat', dumps=True)
#    js_loads = json.loads(js)
#
#    assert isinstance(js, str)
#    assert isinstance(js_loads, dict)
#
#    assert 'layer_names' in js_loads.keys()
#    assert isinstance(js_loads['layer_names'], list)
#    assert len(js_loads['layer_names']) == 2
#    assert 'layer_1' in js_loads['layer_names']
#    assert 'layer_2' in js_loads['layer_names']
#
#    assert 'points' in js_loads.keys()
#    assert isinstance(js_loads['points'], dict)
#    assert len(js_loads['points']) == 2
#    assert 'type' in js_loads['points']
#    assert js_loads['points']['type'] == 'FeatureCollection'
#    assert 'features' in js_loads['points']
#    assert isinstance(js_loads['points']['features'], list)
#    assert isinstance(js_loads['points']['features'][0], dict)
#    assert len(js_loads['points']['features']) == df.shape[0]
#    assert 'type' in js_loads['points']['features'][0]
#    assert 'geometry' in js_loads['points']['features'][0]
#    assert 'properties' in js_loads['points']['features'][0]
#    assert js_loads['points']['features'][0]['type'] == 'Feature'
#    assert 'coordinates' in js_loads['points']['features'][0]['geometry'].keys()
#    assert 'type' in js_loads['points']['features'][0]['geometry'].keys()
#
#    # dumps = False
#    js_loads = json_helpers.geojson_encoder_3(df, layer_names=['layer_1', 'layer_2'], lng='lng', lat='lat', dumps=False)
#
#    assert isinstance(js_loads, dict)
#
#    assert 'layer_names' in js_loads.keys()
#    assert isinstance(js_loads['layer_names'], list)
#    assert len(js_loads['layer_names']) == 2
#    assert 'layer_1' in js_loads['layer_names']
#    assert 'layer_2' in js_loads['layer_names']
#
#    assert 'points' in js_loads.keys()
#    assert isinstance(js_loads['points'], dict)
#    assert len(js_loads['points']) == 2
#    assert 'type' in js_loads['points']
#    assert js_loads['points']['type'] == 'FeatureCollection'
#    assert 'features' in js_loads['points']
#    assert isinstance(js_loads['points']['features'], list)
#    assert isinstance(js_loads['points']['features'][0], dict)
#    assert len(js_loads['points']['features']) == df.shape[0]
#    assert 'type' in js_loads['points']['features'][0]
#    assert 'geometry' in js_loads['points']['features'][0]
#    assert 'properties' in js_loads['points']['features'][0]
#    assert js_loads['points']['features'][0]['type'] == 'Feature'
#    assert 'coordinates' in js_loads['points']['features'][0]['geometry'].keys()
#    assert 'type' in js_loads['points']['features'][0]['geometry'].keys()


