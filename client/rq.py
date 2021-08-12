import requests
from urllib import parse

class MeasurementRequest:
    def __init__(self):
        self.server_url = 'http://210.115.46.198:48090'
        self.api = {
            'start': '/meter/start',
            'end': '/meter/end',
            'hello': '/hello'
        }

    def hello(self):
        return requests.get(self.server_url + self.api['hello'])

    def start(self, model_name):
        return requests.get(self.server_url + self.api['start'])

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