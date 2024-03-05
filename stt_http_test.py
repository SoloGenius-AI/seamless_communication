import requests

if __name__ == '__main__':
    tem_1 = {'wav_file_path': './LJ_eng.wav', 'tgt_lang': 'cmn'}
    url_1 = 'http://127.0.0.1:47861/stt'
    headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + 'kk'}
    r_1 = requests.post(url_1, json=tem_1, headers=headers)
    print('{}'.format(r_1.json()))
