import os
import requests

from flask import Flask, json, make_response, request
from flask_cors import CORS
from gevent import pywsgi


app = Flask(__name__)
CORS(app)
token_str = os.environ.get('gpt_key')
cloud_url = 'https://xxx00.win/v1/chat/completions'


@app.route('/ask_gpt', methods=['GET', 'POST'])
def ask_gpt():
    data = request.get_data()
    json_re = json.loads(data)
    text = json_re.get('text')
    headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + token_str}
    body_json = {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'messages': [{'role': 'user',
                      'content': '{}'.format(text)
                      }]
    }
    res_dict = requests.post(cloud_url, json=body_json, headers=headers)
    res_dict = res_dict.json()
    try:
        choices_list = res_dict['choices']
        if len(choices_list) == 1:
            response_dict = choices_list[0]
            message_dict = response_dict['message']
            content: str = message_dict['content']
            return {'content': content, 'code': 200}
        else:
            return {'content': res_dict, 'code': 201}

    except Exception as e:
        return {'content': e, 'code': 500}


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 47862), app)
    server.serve_forever()
