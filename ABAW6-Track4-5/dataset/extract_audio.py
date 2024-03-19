# s3prl
# import s3prl.hub as hub
import torch
from scipy.io import wavfile
import os
from tqdm import tqdm
# import pickle
# from transformers import Wav2Vec2Processor 
# import soundfile as sf
# import torchaudio
import numpy as np
# from torch import nn
# import arff
import librosa

def extract_audio_feature():
    # root = '/project/zhangwei/ABAW5/challenge1_3/videos/audio'
    root ='/project/ABAW6/data/challenge4'
    # mode_name = 'hubertbase'
    
    mode_name = 'wav2vec2'
    if mode_name == 'wav2vec2':
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model_4 = bundle.get_model().cuda()
    elif mode_name == 'hubertbase':
        bundle = torchaudio.pipelines.HUBERT_BASE
        model_4 = bundle.get_model().cuda()
    
    save_root = f'/project/ABAW6/data/challenge4/audio_{mode_name}'
    os.makedirs(save_root, exist_ok=True)
    # model_4 = getattr(hub, mode_name)()
    # device = 'cuda'  # or cpu
    # model_4 = model_4.to(device)
    # audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    
    print('loading model')

    root_wav = os.path.join(root, 'audios')
    wav_files = os.listdir(root_wav)[::-1]
    print(f'processing {root_wav}')
    save_folder = save_root
    os.makedirs(save_folder, exist_ok=True)
    for wav_f in tqdm(wav_files, total=len(wav_files)):
        save_file = os.path.join(save_folder, wav_f.split('.')[0]+'.npy')
        if os.path.exists(save_file):
            continue
        # wav, rate = sf.read(os.path.join(root_wav, wav_f))
        # wav = audio_processor(wav[:], sampling_rate=rate, return_tensors='pt').input_values
        wav, rate = torchaudio.load(os.path.join(root_wav, wav_f))
        if rate!= bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, rate, bundle.sample_rate)
        reps = []
        channel, length = wav.shape
        max_length = 2500000
        # for wav_c in wav:
        
        # rate, wav = wavfile.read(os.path.join(root_wav, wav_f))
        # wavs = [torch.tensor(wav, dtype=torch.float).to(device)]
        with torch.no_grad():
            # reps = model_4(wav)['last_hidden_state']
            for i in range(length//max_length+1):
                reps.append(model_4.extract_features(wav.cuda()[:,i*max_length:(i+1)*max_length])[0][-1])
            # reps_c = model_4(wav_c.cuda().unsqueeze(dim=0))[0][-1]
            # reps.append(reps_c)
        # with open(save_file, 'wb') as f:
        #     pickle.dump(reps,f)
        reps = torch.concatenate(reps,dim=1)
        if channel !=1:
            reps = torch.mean(reps, dim=0).unsqueeze(dim=0)
        
        np.save(save_file,reps.cpu().numpy().squeeze(0))    

def extract_text_from_wav():
    root = '/project/zhangwei/ABAW5/challenge4'
    mode_name = 'hubert_large_ll60k'
    model_4 = getattr(hub, mode_name)()
    device = 'cuda'  # or cpu
    model_4 = model_4.to(device)
    
    
    for split in ['train', 'val']:
        root_wav = os.path.join(root, split, 'wav')
        wav_files = os.listdir(root_wav)
        print(f'processing {root_wav}')
        save_root = os.path.join(root, mode_name, split)
        os.makedirs(save_root, exist_ok=True)
        for wav_f in tqdm(wav_files, total=len(wav_files)):
            rate, wav = wavfile.read(os.path.join(root_wav, wav_f))
            wavs = [torch.tensor(wav, dtype=torch.float).to(device)]
            with torch.no_grad():
                reps = model_4(wavs)['last_hidden_state']
            save_file = os.path.join(save_root, wav_f.split('.')[0]+'.pkl')
            with open(save_file, 'wb') as f:
                f.write(reps.cpu().numpy())

def extract_audio_feature_vggish():
    root = '/project/zhangwei/ABAW5/challenge4'
    mode_name = 'vggish'
    save_root = '/data/data/ABAW5/challenge4'

    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval().cuda()
    
    for split in ['train']:
        root_wav = os.path.join(root, split, 'wav')
        wav_files = os.listdir(root_wav)[::]
        print(f'processing {root_wav}')
        save_folder = os.path.join(save_root, f'{split}_{mode_name}_npy')
        os.makedirs(save_folder, exist_ok=True)
        for wav_f in tqdm(wav_files, total=len(wav_files)):
            save_file = os.path.join(save_folder, wav_f.split('.')[0]+'.npy')
            if os.path.exists(save_file):
                continue

            with torch.no_grad():
                reps = model.forward(os.path.join(root_wav, wav_f))
            np.save(save_file,reps.cpu().numpy()/255.) 


def extract_audio_feature_deepspectrum():
    root ='/project/ABAW6/data/challenge5'
    mode_name = 'deepspectrum'
    save_root = '/project/ABAW6/data/challenge5/audio_deepspectrum'
    print('loading model')

    root_wav = os.path.join(root, 'audio')
    wav_files = os.listdir(root_wav)[::-1]
    print(f'processing {root_wav}')
    save_folder = save_root
    os.makedirs(save_folder, exist_ok=True)
    for wav_f in tqdm(wav_files, total=len(wav_files)):
        save_file = os.path.join(save_folder, wav_f.split('.')[0]+'.arff')
        if os.path.exists(save_file):
            continue
        # wav, rate = sf.read(os.path.join(root_wav, wav_f))
        # wav = audio_processor(wav[:], sampling_rate=rate, return_tensors='pt').input_values
        wav_file = os.path.join(root_wav, wav_f)
        cmd = f'deepspectrum features {wav_file} -t 1 0.1 -nl -en densenet121 -fl fc2 -m mel -o {save_file}'
        os.system(cmd)
            # with open(save_file, 'wb') as f:
            #     pickle.dump(reps,f)
            # reps = torch.concatenate(reps,dim=1)
            # if channel !=1:
            #     reps = torch.mean(reps, dim=0).unsqueeze(dim=0)
            
            # np.save(save_file,reps.cpu().numpy())    
      
def extract_audio_feature_MFCC():
    root ='/project/zhangwei/ABAW5/challenge4'
    mode_name = 'MFCC'
    save_root = '/data/data/ABAW5/challenge4'
    print('loading model')
    for split in ['test', 'val']:
        root_wav = os.path.join(root, split, 'wav')
        wav_files = os.listdir(root_wav)[::-1]
        print(f'processing {root_wav}')
        save_folder = os.path.join(save_root, f'{split}_{mode_name}_npy')
        os.makedirs(save_folder, exist_ok=True)
        for wav_f in tqdm(wav_files, total=len(wav_files)):
            save_file = os.path.join(save_folder, wav_f.split('.')[0]+'.npy')
            if os.path.exists(save_file):
                continue
            # wav, rate = sf.read(os.path.join(root_wav, wav_f))
            # wav = audio_processor(wav[:], sampling_rate=rate, return_tensors='pt').input_values
            wav_file = os.path.join(root_wav, wav_f)
            y, sr = librosa.load(wav_file)
            mel_128=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
            
            np.save(save_file,mel_128)         

from moviepy.editor import VideoFileClip

def extract_audio_from_videos(video_folder='/project/ABAW6/data/challenge5/test/raw', output_folder='/project/ABAW6/data/challenge5/test/audio'):
    """
    提取给定文件夹中所有视频文件的音频，并将其存储为MP3格式。

    参数:
    video_folder -- 包含视频文件的文件夹路径
    output_folder -- 存储输出MP3文件的文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历视频文件夹中的所有文件
    for file_name in tqdm(os.listdir(video_folder), total=len(os.listdir(video_folder))):
        # 创建完整的文件路径
        video_path = os.path.join(video_folder, file_name)
        # 构建输出音频文件的路径（更改文件扩展名为.mp3）
        audio_file_name = os.path.splitext(file_name)[0] + '.mp3'
        audio_file_path = os.path.join(output_folder, audio_file_name)
        if os.path.exists(audio_file_path):
            continue
        # 检查文件是否是视频文件（这里以文件扩展名进行简单判断）
        if os.path.isfile(video_path) and file_name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            try:
                # 使用MoviePy加载视频文件
                video_clip = VideoFileClip(video_path)
                # 提取视频的音频部分
                audio_clip = video_clip.audio

                # 将音频保存为MP3格式
                audio_clip.write_audiofile(audio_file_path)
                # 释放视频和音频片段（释放资源）
                audio_clip.close()
                video_clip.close()
                print(f"音频已成功提取并保存为：{audio_file_path}")
            except Exception as e:
                print(f"提取音频时发生错误：{e}")
                continue
    
if __name__ == '__main__':
    extract_audio_from_videos()
    # extract_audio_feature()
    # extract_audio_feature_deepspectrum()
    # extract_audio_feature_MFCC()
    # extract_audio_feature_vggish()