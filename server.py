from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask_restful import reqparse
from sample import sample


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('start_text', type=str, help='Starting text')
parser.add_argument('n', type=int, help='Number of suggestions per request')

class Home(Resource):
    def get(self):
        return {'message': 'Have Fun!'}

class Generate(Resource):
    def get(self):
        args = parser.parse_args()
        sample_args = {'save_dir': 'save', 'n': 10, 'prime': args.start_text, 'sample': 1}
        sampled_text = sample(sample_args)[len(args.start_text)::]
        result = {'completions': [sampled_text], 'start_text': args.start_text, 'time': 15}
        return result

api.add_resource(Home, '/')
api.add_resource(Generate, '/generate')

if __name__ == '__main__':
     app.run(port = 8080, debug=True)
