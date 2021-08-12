import sys
if sys.platform == 'linux':
    print("FATAL: Server must be ran on Windows environment\n"
          "with Power-Z KM001 current analyzer connected via HID.")
    sys.exit(0)

'''Server

server will provide power measurement information to client.
'''

from analyzer import PiCurrentAnalyzer
from flask import Flask
from flask_restful import Resource, reqparse, Api

analyzer = PiCurrentAnalyzer(executable_path='./bin/Power-Z.exe')

class PmReadyApi(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('modelname', type=str)
            args = parser.parse_args()

            analyzer.open()
            analyzer.set_title(title=args['modelname'])
            return {'result': True}
        except Exception as e:
            return {'error': str(e)}

class PmBeginApi(Resource):
    def get(self):
        try:
            analyzer.begin()
            return {'result': True}
        except Exception as e:
            return {'error': str(e)}

class PmEndApi(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('modelname', type=str)
        parser.add_argument('elapsedtime', type=str)
        parser.add_argument('totalframes', type=str)
        args = parser.parse_args()

        model_name = args['modelname']
        elapsed_time_sec = int(args['elapsedtime'])
        total_frames = int(args['totalframes'])

        analyzer.end()
        total_energy_consumption_mwh = analyzer.postprocess(model_name, elapsed_time_sec, total_frames)
        print("Done")

        return {'energy_mwh': total_energy_consumption_mwh}

class HeartbeatApi(Resource):
    def get(self):
        try:
            return 'World!'
        except Exception as e:
            return "{'error': str(e)}"

app = Flask(__name__)
api = Api(app)

api.add_resource(PmReadyApi, '/meter/ready')
api.add_resource(PmBeginApi, '/meter/start')
api.add_resource(PmEndApi, '/meter/end')
api.add_resource(HeartbeatApi, '/hello')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=48090)