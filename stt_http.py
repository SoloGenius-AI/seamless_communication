import os

import numpy as np
import requests
import sounddevice as sd
from flask import Flask, json, make_response, request
from flask_cors import CORS
from gevent import pywsgi

from seamless_communication.inference.generator import SequenceGeneratorOptions
from seamless_communication.inference import Translator
import torch


app = Flask(__name__)
CORS(app)

token_str = os.environ.get('gpt_key')
cloud_url = 'https://xxx00.win/v1/chat/completions'
url_1 = 'http://localhost:57861/gen_audio'

translator = Translator("seamlessM4T_large",
                        "vocoder_36langs",
                        torch.device('cpu'))

m4t_text_generation_opts = SequenceGeneratorOptions(
    beam_size=5,
    unk_penalty=torch.inf,
    soft_max_seq_len=(1, 200)
)

API_TOKEN = os.environ.get('API_TOKEN')
KK_TOKEN = os.environ.get('KK_API_TOKEN')
DEBUG = os.environ.get('DEBUG')


def verify_token(token_str: str):
    try:
        if DEBUG or token_str == 'Bearer ' + API_TOKEN or 'Bearer ' + KK_TOKEN:
            return True
        return False
    except Exception as e:
        return False


@app.route('/stt', methods=['GET', 'POST'])
def speaker_to_text():
    try:
        # token_ = request.headers.get('Authorization')
        # if not verify_token(token_):
        #     raise Exception('verify token failed.')

        data = request.get_data()
        json_re = json.loads(data)
        wav_file_path = json_re.get('wav_file_path')
        tgt_lang = json_re.get('tgt_lang')

        text_output, _ = translator.predict(
            input=wav_file_path,
            task_str="S2TT",
            # cmn, jpn
            tgt_lang=tgt_lang,
            text_generation_opts=m4t_text_generation_opts,
            unit_generation_opts=None
        )
        res_str_list = list(map(lambda x_: f'{x_.__str__()}', text_output))
        res_str = ''.join(res_str_list)
        print(res_str)

        response = ask_gpt(res_str)
        print(f'ask_gpt: {response=}')
        audio_concat = []
        sampling_rate = 44100

        response_text = response['content']
        response_text_list = response_text.split('。') if len(response) > 64 else [response_text]
        for tem_1 in response_text_list:
            headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + 'kk'}
            r_1 = requests.post(url_1, json=tem_1, headers=headers).json()
            audio_concat.extend(r_1['audio_concat'])
            sampling_rate = r_1['sampling_rate']

        res_dict = {'audio_concat': audio_concat, 'code': 200, 'sampling_rate': sampling_rate}
    except Exception as e_:
        print(e_)
        res_dict = {'text': '', 'code': 500, 'err_info': e_}
    return res_dict


def ask_gpt(text):
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
    server = pywsgi.WSGIServer(('0.0.0.0', 47861), app)
    server.serve_forever()
