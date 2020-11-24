## Train code

### [train_data & test data 경로 변경]
wavs_path = glob(os.path.join('/home/lab/Documents/Human/ESC-50-master/Record', '*', '*')) =>
wavs_path = glob(os.path.join('/home/lab/Documents/Human/Elevator_Sound_Classification/Record', '*', '*'))

### [data augmentation 코드 변경]
기존 pitch 조절 코드 이름 변경: data_augmentation.py -> data_augmentation_pitch.py \n
speed 조절 코드 : data_augmentation_speed.py
