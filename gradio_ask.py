import os

import gradio as gr
import numpy as np
import requests

from seamless_communication.inference.generator import SequenceGeneratorOptions
from seamless_communication.inference import Translator
import torch

import scipy.io.wavfile as wav
from scipy.signal import resample

token_str = os.environ.get('gpt_key')
cloud_url = 'https://xxx00.win/v1/chat/completions'
url_1 = 'http://localhost:57861/gen_audio'
ask_model = 'gpt-3.5-turbo'

translator = Translator("seamlessM4T_large",
                        "vocoder_36langs",
                        torch.device('cpu'))

m4t_text_generation_opts = SequenceGeneratorOptions(
    beam_size=5,
    unk_penalty=torch.inf,
    soft_max_seq_len=(1, 200)
)


def speaker_to_text(wav_file_path, tgt_lang='cmn'):
    try:
        sample_rate, data = wav.read(wav_file_path)
        new_rate = 16000
        resampled_data = resample(data, int(len(data) * new_rate / sample_rate))
        resampled_data = resampled_data.astype('int16')
        wav.write(wav_file_path, new_rate, resampled_data)
        print(f'转换为16k: {wav_file_path}')
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

        ask_count = 0
        ask_done = False
        while not ask_done and ask_count < 3:
            try:
                response = ask_gpt(res_str)
                ask_done = True
            except Exception as e_:
                ask_count += 1
                print(f'ask try {ask_count}: {e_}')
        if not ask_done:
            raise Exception(f'ask fail..')

        print(f'gpt_res: {response=}')
        audio_concat = []
        sampling_rate = 44100

        response_text = response['content']
        response_text_list = response_text.split('。') if len(response) > 64 else [response_text]
        for tem_1 in response_text_list:
            headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + 'kk'}
            r_1 = requests.post(url_1, json={'text': tem_1}, headers=headers).json()
            audio_concat.extend(list(map(int, r_1['audio_concat'].split(','))))
            sampling_rate = r_1['sampling_rate']

        res_dict = {'audio_concat': audio_concat, 'code': 200, 'ask_text': res_str,
                    'sampling_rate': sampling_rate, 'text': response_text}
    except Exception as e_:
        print(e_)
        res_dict = {'text': '', 'code': 500, 'err_info': e_}
    return res_dict


def ask_gpt(text):
    headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + token_str}
    body_json = {
        'model': ask_model,
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


def rec_audio(audio):
    print(audio)
    res = speaker_to_text(audio)
    try:
        audio_concat = res['audio_concat']
        code = res['code']
        sampling_rate = res['sampling_rate']
        response_text = res['text']
        ask_text = res['ask_text']
        data = gr.processing_utils.convert_to_16_bit_wav(1.0 * np.array(audio_concat))
        new_rate = 16000
        resampled_data = resample(data, int(len(data) * new_rate / int(sampling_rate)))
        resampled_data = resampled_data.astype('int16')
        return ask_text, response_text, (new_rate, resampled_data)
    except Exception as e_:
        print(f'gen_audio_err: {e_}')


demo = gr.Interface(
    rec_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type='filepath', show_label=True, editable=True),
    outputs=[gr.Text(label='ask'), gr.Text(label='response'), gr.Audio(type='numpy', autoplay=True)],
)

if __name__ == "__main__":
    demo.launch(server_port=57895)
