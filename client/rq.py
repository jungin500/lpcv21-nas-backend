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
        response = requests.get(self.server_url + self.api['end'] + '?' + '&'.join([
            'model_name=%s' % parse.urlencode(model_name, doseq=True),
            'elapsed_time_sec=%d' % elapsed_time_sec,
            'total_frames=%d' % total_frames
        ]))
        data = response.json()
        if 'energy_mwh' not in data:
            print("Error: ", data)
            return data
            
        return data['energy_mwh']