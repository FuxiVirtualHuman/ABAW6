#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from io import BytesIO
import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json
import os
from tqdm import tqdm
import time
import hashlib
import hmac
import base64
import requests
import cv2
from pydub import AudioSegment
from timeit import default_timer as timer
import pandas as pd
import whisper

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(
        quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(
            shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(
            desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def metadata_json_output(metadata):
    json_result = dict()
    json_result["transcripts"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return json.dumps(json_result, indent=2)


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        print('DeepSpeech ', version())
        exit(0)


class Speech2Text():
    def __init__(self, args) -> None:
        ds = Model(args.model)

        if args.beam_width:
            ds.setBeamWidth(args.beam_width)

        self.desired_sample_rate = ds.sampleRate()
        self.candidate_transcripts = args.candidate_transcripts
        if args.scorer:
            print('Loading scorer from files {}'.format(
                args.scorer), file=sys.stderr)
            scorer_load_start = timer()
            ds.enableExternalScorer(args.scorer)
            scorer_load_end = timer() - scorer_load_start
            print('Loaded scorer in {:.3}s.'.format(
                scorer_load_end), file=sys.stderr)

            if args.lm_alpha and args.lm_beta:
                ds.setScorerAlphaBeta(args.lm_alpha, args.lm_beta)

        if args.hot_words:
            print('Adding hot-words', file=sys.stderr)
            for word_boost in args.hot_words.split(','):
                word, boost = word_boost.split(':')
                ds.addHotWord(word, float(boost))

        self.ds = ds

    def extract_text(self, audio_file):
        fin = wave.open(audio_file, 'rb')
        fs_orig = fin.getframerate()
        # audio_length = fin.getnframes() * (1/fs_orig)
        if fs_orig != self.desired_sample_rate:
            print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
                fs_orig, self.desired_sample_rate), file=sys.stderr)
            fs_new, audio = convert_samplerate(
                audio_file, self.desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        fin.close()

        # print('Running inference.', file=sys.stderr)
        # inference_start = timer()
        # sphinx-doc: python_ref_inference_start
        # if args.extended:
        #     print(metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0]))
        # elif args.json:
        #     print(metadata_json_output(ds.sttWithMetadata(audio, args.candidate_transcripts)))
        # else:
        #     print(ds.stt(audio))
        # sphinx-doc: python_ref_inference_stop
        # inference_end = timer() - inference_start
        # print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
        res_json = json.loads(metadata_json_output(
            self.ds.sttWithMetadata(audio, self.candidate_transcripts)))
        res_json = res_json['transcripts'][0]
        # text = metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0])
        text = ' '.join([w['word'] for w in res_json['words']])
        res_json['text'] = text
        return res_json, text


class HangyanRec():
    def __init__(self, max_duration=50):
        self.appkey = 'phr-fuxi'
        self.appsecret = '74c72dee-bf9a-4de2-8c1f-96be1a1ecabd'
        self.url = 'http://api-test.vop.netease.com/phone_rec'
        self.max_duration = max_duration

    def run(self, wav_path, lang='en', type="wav",max_duration=50):
        # lang: zh, en
        # split_path = '/project/zhangwei/ABAW5/challenge1_3/videos/new_audios_seg'
        words = []
        
        audio = AudioSegment.from_file(wav_path, format='mp3')
        duration_ms = len(audio)
        if duration_ms < 500:
            return ''
        chunk_length_ms = max_duration * 1000  # 60s
        text = ''
        for i in range(0, duration_ms,chunk_length_ms):
            new_audio = audio[i:min(i+chunk_length_ms,duration_ms)]
            byte_io = BytesIO()
            new_audio.export(byte_io, format="wav")
            speech = byte_io.getvalue()
        # if type == "wav":
        #     speech = self.read_wave(wav_path)
        # else:
        #     raise Exception("audio type error:", wav_path)
        

        
            curtime = str(int(time.time()))
            hl = hashlib.md5()
            hl.update((self.appkey + curtime).encode(encoding='utf-8'))
            sign = hmac.new(self.appsecret.encode('utf-8'),
                            hl.hexdigest().encode('utf-8'), hashlib.sha1).digest()
            checksum = base64.b64encode(sign)
            params = {'appkey': self.appkey, 'lan': lang}
            headers = {
                'curtime': curtime,
                'checksum': checksum,
                'content-type': 'audio/wav',
                'cuid': 'fuxi-avatarlib'
            }
            response = requests.post(self.url,
                                        params=params,
                                        headers=headers,
                                        data=speech)
            r = response.json()
            if r['ret_code'] != 1:
                error = r['ret_msg']
                # print(wav_path)
                raise RuntimeError('Hangyan Rec Error: ' + error+f'[{wav_path}]')
            aligned_text = r['result']
            text += ' '.join([w['word'] for w in aligned_text if w['word'] != 'sil'])
        return text

    def split_wave(self, wav_path, save_root, max_duration=50):
        wav_name = wav_path.split('/')[-1].replace('.wav', '')
        fin = wave.open(wav_path, 'rb')
        fs_orig = fin.getframerate()
        audio_length = fin.getnframes() * (1/fs_orig)
        fin.close()

        audio = AudioSegment.from_wav(wav_path)
        n = int(audio_length//max_duration + 1)
        for i in range(n):
            new_audio = audio[i*max_duration*1000:(i+1)*max_duration*1000]
            new_audio.export(os.path.join(
                save_root, wav_name+f'_{i}.wav'), format='wav')
        return n

    def read_wave(self, wav_path, max_duration=60):
        with open(wav_path, 'rb') as f:
            wav_data = f.read()
        return wav_data


class Speech2TextWhisper():
    def __init__(self, modelname='large', max_duration=50):
        self.model = whisper.load_model(modelname)
        

    def run(self, wav_path, lang='en', type="wav",max_duration=500):
        # lang: zh, en
        # split_path = '/project/zhangwei/ABAW5/challenge1_3/videos/new_audios_seg'
        res = self.model.transcribe(wav_path)
        return res["text"]
    
def main(engine='deepspeech'):
    parser = argparse.ArgumentParser(
        description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=False, default='/data/Workspace/ABAW/checkpoints/deepspeech-0.9.3-models.pbmm',
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=False, default='/data/Workspace/ABAW/checkpoints/deepspeech-0.9.3-models.scorer',
                        help='Path to the external scorer file')
    # parser.add_argument('--audio', required=True,
    #                     help='Path to the audio file to run (WAV format)')
    parser.add_argument('--beam_width', type=int,
                        help='Beam width for the CTC decoder')
    parser.add_argument('--lm_alpha', type=float,
                        help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
    parser.add_argument('--lm_beta', type=float,
                        help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
    parser.add_argument('--version', action=VersionAction,
                        help='Print version and exits')
    parser.add_argument('--extended', required=False, action='store_true',
                        help='Output string from extended metadata')
    parser.add_argument('--json', required=False, action='store_true',
                        help='Output json from metadata with timestamp of each word')
    parser.add_argument('--candidate_transcripts', type=int, default=1,
                        help='Number of candidate transcripts to include in JSON output')
    parser.add_argument('--hot_words', type=str,
                        help='Hot-words and their boosts.')
    args = parser.parse_args()

    print('Loading model from file {}'.format(args.model), file=sys.stderr)
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    assert engine in ['deepspeech', 'fuxi', 'whisper']
    if engine == 'deepspeech':
        from deepspeech import Model, version
        
        speech2text_deepspeech = Speech2Text(args)
    elif engine == 'fuxi':
        speech2text_fuxi = HangyanRec()
    elif engine == 'whisper':
        speech2text_whipser = Speech2TextWhisper()
        
    root = '/project/ABAW6/data/challenge5/test'

    audio_root = os.path.join(root, 'audio')
    audio_list = os.listdir(audio_root)[::]
    save_root = os.path.join(root, f'text_{engine}_large')


    text_list = []
    os.makedirs(save_root, exist_ok=True)
    res_dict = {}
    for audio_name in tqdm(audio_list, total=len(audio_list)):
        audio_file = os.path.join(audio_root, audio_name)
        save_path = os.path.join(
            save_root, audio_name.split('.')[0]+ '.json')
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                text = json.load(f)['text']
                text_list.append(audio_name + ';' + text+'\n')
            continue
        if engine == 'deepspeech':
            res_json, text = speech2text_deepspeech.extract_text(
                audio_file)
        elif engine == 'fuxi':
            text = speech2text_fuxi.run(audio_file)
        elif engine == 'whisper':
            text = speech2text_whipser.run(audio_file)
        # text_list.append(audio_name + ';' + text+'\n')
        res_dict[audio_name] = text.strip()
        # with open(save_path, 'w') as f:
        #     json.dump(res_json, f)

    with open(os.path.join(root, save_root+'.json'), 'w') as f:
        # f.writelines(text_list)
        json.dump(res_dict, f, indent=4)


def align_text_by_frames():
    text_time_root = '/project/zhangwei/ABAW5/challenge1_3/text_newvid_fuxi'
    save_root = '/project/zhangwei/ABAW5/challenge1_3/wordlist'
    # images_root = '/project/zhangwei/ABAW5/challenge1_3/ffmpeg_crop'
    video_root = '/project/zhangwei/ABAW5/challenge1_3/videos/new_vids'
    jsons = os.listdir(text_time_root)
    punct_file = '/project/zhangwei/ABAW5/challenge1_3/sentences_finish.xlsx'
    pd_punct = pd.read_excel(punct_file,engine='openpyxl',)
    sent_with_punct= {}
    n = len(pd_punct['原'])
    for i in range(n):
        vidid = pd_punct['Unnamed: 1'][i].replace('.txt', '')
        sent_with_punct[vidid] = pd_punct['改'][i]
        
    for json_file in tqdm(jsons, total=len(jsons)):
        vid_name = json_file.replace('.json', '')
        save_path = os.path.join(save_root, vid_name+'.txt')

        # get frame rate
        vid_path = os.path.join(video_root, vid_name+'.mp4')
        if not os.path.exists(vid_path):
            vid_path = vid_path.replace('.mp4', '.avi')
        assert os.path.exists(vid_path)
        cam = cv2.VideoCapture(os.path.join(vid_path))
        frame_rate = int(cam.get(cv2.CAP_PROP_FPS))
        frame_n = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        wordlist = ['-1' for _ in range(frame_n)]

        with open(os.path.join(text_time_root, json_file), 'r') as f:
            data = json.load(f)
        for word in data['words']:
            text, phones = word['word'], word['phones']
            start = phones[0]['bg']
            end = phones[-1]['ed']
            # 10ms is an unit of time for audio
            frame_start = int(start * 10 / 1000 * frame_rate)
            frame_end = int(end * 10 / 1000 * frame_rate)
            if text != 'sil':
                wordlist[frame_start: frame_end + 1] = [text] * \
                    (frame_end-frame_start+1)

        with open(save_path, 'w') as f:
            f.writelines([l + '\n' for l in wordlist])

    return


def auto_puntuation(text=''):
    from fastpunct import FastPunct
    fastpunct = FastPunct()
    fastpunct.punct([
                    "sil THE sil I AM FRIDAY sil START NOW sil sil THE sil GOOD THING ANY A GRIN OF SMALL CHANNELS sil SOME STILL GOING STRONG sil \
                        DAMN sil HE PUT A CRATER sil IN THE DAM POOL sil THE sil TOILET HUMOR ALWAYS WORKED sil AS WHAT WAS sil SHE LIKE AIRPLANE sil MAKERS sil sil\
                            ANYTHING FOR MY BOY sil STANDS sil AND YOU KNOW sil HOW IS IT ALLOWS AN EASY OKAY sil HERE'S YOUR COOKIE sil I WANTED TO KNOW sil OK sil\
                                SO sil ONE HUNDRED AND FIFTY TWO BILLION PEOPLE HAVE BEEN ASKING ME sil TO sil DO SORT OF A CHALLENGE sil IF YOU HOLD A MY FAN PAGE sil ON\
                                    sil FACEBOOK sil OR ANY COMMENTS FROM THE LAST sil VIDEO sil THAT THERE ISN'T A GESTURE sil AND I USE AN OLD ROOMMATE A PEER PRESSURE sil BECAUSE sil MOST sil CHALLENGING ABOUT THIS ENEMY sil BRAIN sil SO sil EVERYONE HAS THE GOVERNOR'S EYE DOESN'T SAY sil HOW DO YOU KNOW sil THIS WAS A BIG sil YOU KNOW BECAUSE THEN IT REALLY IS GOOD FOR YOU sil YOU SHALL SET THEM IN ALL NIGHT sil sil YOU KNOW sil I'VE HEARD ABOUT THIS sil I MEAN sil HER SPECIFIC CITY I'VE SEEN OTHER CINNAMON sil CHALLENGES sil I GUESS sil SO I WAS BORN IN A sil VENICE sil I WOULD\
                                        WATCH BEAUVILLE BEFORE AT THIS POINT sil BUT IT'S WORTH SAYING SO sil YOU KNEW sil YOU JUST TAKE IT sil I GUESS YOU COUNT THE TEAM\
                                            sil OR sil YOU SAY sil CRAZY sil MANUEL HERE'S A SETTLEMENT sil I'M NOT LAUGHING sil sil EVER BROUGHT A BRILLIANT sil sil AGAIN sil\
                                                I BURNED EVERYTHING sil OVER NOW sil sil A MARRIED sil MAN SAYS sil I DID NOT sil".replace('sil', '')
                    ])

    # ["John Smith's dog is creating a ruccus.",
    # 'Ys Jagan is the chief minister of Andhra Pradesh.',
    # 'We visted New York last year in May.']

    # punctuation correction with optional spell correction (experimental)

    fastpunct.punct([
                    'johns son peter is marring estella in jun',
                    'kamal hassan is a gud actr'], correct=True)

    # ["John's son Peter is marrying Estella in June.",
    # 'Kamal Hassan is a good actor.']


if __name__ == '__main__':
    main(engine='whisper')
    # align_text_by_frames()
    
    # from deepsegment import DeepSegment
    # from deepcorrect import DeepCorrect
    # # text = "sil THE sil I AM FRIDAY sil START NOW sil sil THE sil GOOD THING ANY A GRIN OF SMALL CHANNELS sil SOME STILL GOING STRONG sil DAMN sil HE PUT A CRATER sil IN THE DAM POOL sil THE sil TOILET HUMOR ALWAYS WORKED sil AS WHAT WAS sil SHE LIKE AIRPLANE sil MAKERS sil sil ANYTHING FOR MY BOY sil STANDS sil AND YOU KNOW sil HOW IS IT ALLOWS AN EASY OKAY sil HERE'S YOUR COOKIE sil I WANTED TO KNOW sil OK sil SO sil ONE HUNDRED AND FIFTY TWO BILLION PEOPLE HAVE BEEN ASKING ME sil TO sil DO SORT OF A CHALLENGE sil IF YOU HOLD A MY FAN PAGE sil ON sil FACEBOOK sil OR ANY COMMENTS FROM THE LAST sil VIDEO sil THAT THERE ISN'T A GESTURE sil AND I USE AN OLD ROOMMATE A PEER PRESSURE sil BECAUSE sil MOST sil CHALLENGING ABOUT THIS ENEMY sil BRAIN sil SO sil EVERYONE HAS THE GOVERNOR'S EYE DOESN'T SAY sil HOW DO YOU KNOW sil THIS WAS A BIG sil YOU KNOW BECAUSE THEN IT REALLY IS GOOD FOR YOU sil YOU SHALL SET THEM IN ALL NIGHT sil sil YOU KNOW sil I'VE HEARD ABOUT THIS sil I MEAN sil HER SPECIFIC CITY I'VE SEEN OTHER CINNAMON sil CHALLENGES sil I GUESS sil SO I WAS BORN IN A sil VENICE sil I WOULD WATCH BEAUVILLE BEFORE AT THIS POINT sil BUT IT'S WORTH SAYING SO sil YOU KNEW sil YOU JUST TAKE IT sil I GUESS YOU COUNT THE TEAM sil OR sil YOU SAY sil CRAZY sil MANUEL HERE'S A SETTLEMENT sil I'M NOT LAUGHING sil sil EVER BROUGHT A BRILLIANT sil sil AGAIN sil I BURNED EVERYTHING sil OVER NOW sil sil A MARRIED sil MAN SAYS sil I DID NOT sil".replace('sil ', '').lower()
    # text = 'hello word today is friday i dont know'
    # segmenter = DeepSegment('en')
    # # corrector = DeepCorrect('/data/Workspace/ABAW/checkpoints/deeppunct_params_en','/data/Workspace/ABAW/checkpoints/deeppunct_checkpoint_google_news')
    # seg_text = segmenter.segment(text)
    # for sentence in seg_text:
    #     # sentence_c = corrector(sentence)
    #     print(sentence)
    #     # print(sentence_c)
    # auto_puntuation()