#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import librosa
import soundfile
import wave
from tqdm import tqdm

fail_list = ['Abishini A.H..', 'Craig Emmington','Guilherme Vieira de Araújo', 'TTSmale', 'Anbu Sankar', 'Craig Emmington','TTSfemale', 'Matt D M Burrell' ]

######################################################## speed ########################################################

CHANNELS = 1
swidth = 2
Change_RATE_1 = 3 #2배속
Change_RATE_2 = 1.5 #0.5배속

for dir_num in range(1, 21):
    path_dir = '/home/lab/Documents/Human/Elevator_Sound_Classification/Record/' + str(dir_num) + '/'
    #path_dir = './thirteenthFloor/'
    file_list = os.listdir(path_dir)
    for file in tqdm(file_list):
        isFail = False
        
        for fail in fail_list:
            if fail in file:
                isFail = True
                break
        
        if file == '.ipynb_checkpoints':
            continue
            
        if isFail == True:
            continue

        # print(path_dir, file)
        try:    
            spf = wave.open(path_dir + file, 'rb')
            
            RATE=spf.getframerate()
            signal = spf.readframes(-1)
            
            output_file_name1 = file.replace('.wav', '_x2.wav')
            wf = wave.open(path_dir + output_file_name1, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(swidth)
            wf.setframerate(RATE*Change_RATE_1)
            wf.writeframes(signal)
            wf.close()
            
            output_file_name2 = file.replace('.wav', '_x0.5.wav')
            wf = wave.open(path_dir + output_file_name2, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(swidth)
            wf.setframerate(RATE*Change_RATE_2)
            wf.writeframes(signal)
            wf.close()
            
        except:
            print(path_dir, file)