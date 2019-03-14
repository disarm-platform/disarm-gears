import urllib.parse

faas_endpoint = 'https://faas.srv.disarm.io'
faas_covariate_extractor_endpoint = urllib.parse.urljoin(faas_endpoint, '/function/fn-covariate-extractor')