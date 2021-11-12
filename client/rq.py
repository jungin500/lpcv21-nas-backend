import requests
from urllib import parse

class MeasurementRequest:
    def __init__(self, server_url):
        self.server_url = server_url
        self.api = {
            'ready': '/meter/ready',
            'start': '/meter/start',
            'end': '/meter/end',
            'hello': '/hello'
        }

    def hello(self):
        return requests.get(self.server_url + self.api['hello'])

    def ready(self, model_name):
        url_base = self.server_url + self.api['ready']
        url_query = '&'.join([
            'modelname=%s' % model_name
        ])
        query = parse.parse_qs(url_query)
        url_query = parse.urlencode(query, doseq=True)
        response = requests.get(url_base + '?' + url_query)
        data = response.json()
        return data

    def start(self):
        response = requests.get(self.server_url + self.api['start'])
        data = response.json()
        return data

    def end(self, model_name, elapsed_time_sec, total_frames):
        url_base = self.server_url + self.api['end']
        url_query = '&'.join([
            'modelname=%s' % model_name,
            'elapsedtime=%d' % elapsed_time_sec,
            'totalframes=%d' % total_frames
        ])
        query = parse.parse_qs(url_query)
        url_query = parse.urlencode(query, doseq=True)
        response = requests.get(url_base + '?' + url_query)
        data = response.json()
        return data['energy_mwh']