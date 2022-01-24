import h5py

core_dir = 'D:/Work/Challenges/ChaLearn-SLR/yvan/data/'

h5file = 'key_frames_16.h5'

h5_keyframes = h5py.File(core_dir+h5file, 'r')

# a = keyframes['signer0_sample1002_color'].value


with open('data/train.txt', "r") as txt_file:
    text_lines = txt_file.readlines()

with open('data/keyframes_train.txt',"w") as txt_file:
    for line in text_lines:
        sample = text_lines[0].split(';')[0].split('/')[-1]
        signer = text_lines[0].split(';')[0].split('/')[-2]
        h5_key = signer+'_'+sample+'_color'
        keyframes = (h5_keyframes[h5_key].value)
        txt_file.write(h5_key+';'+str(list(keyframes))[1:-1]+"\n")

