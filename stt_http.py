import os

from flask import Flask, json, make_response, request
from flask_cors import CORS
from gevent import pywsgi

from seamless_communication.inference.generator import SequenceGeneratorOptions
from seamless_communication.inference import Translator
import torch


app = Flask(__name__)
CORS(app)

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
        token_ = request.headers.get('Authorization')
        if not verify_token(token_):
            raise Exception('verify token failed.')

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

        res_dict = {'text': res_str, 'code': 200}
    except Exception as e_:
        print(e_)
        res_dict = {'text': None, 'code': 500, 'err_info': e_}
    return res_dict


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 47861), app)
    server.serve_forever()
