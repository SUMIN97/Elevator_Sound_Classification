#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import librosa
import soundfile

fail_list = ['Abishini A.H..', 'Craig Emmington','Guilherme Vieira de Araújo', 'TTSmale', 'Anbu Sankar', 'Craig Emmington','TTSfemale', 'Matt D M Burrell' ]

######################################################## pitch ########################################################
for dir_num in range(1, 21):
    path_dir = '/home/lab/Documents/Human/Elevator_Sound_Classification/Record/' + dir_num + '/'
    #path_dir = './thirteenthFloor/'
    file_list = os.listdir(path_dir)
    for file in file_list:
        isFail = False
        
        for fail in fail_list:
            if fail in file:
                isFail = True
                break
        
        if file == '.ipynb_checkpoints':
            continue
            
        if isFail == True:
            continue
        data, sr = librosa.load(path_dir + file)
        
        for i in range(-5, 6):
            output_file_name = file.replace('.wav', '_' + str(i) + '.wav')
            pitch_data = librosa.effects.pitch_shift(data, sr, i)
            soundfile.write(path_dir + output_file_name, pitch_data, sr)
