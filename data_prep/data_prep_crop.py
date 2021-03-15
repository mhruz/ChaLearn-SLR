import os
import sys
import random
import h5py

import numpy
import cv2
import matplotlib.pyplot as plt



############################
# original nesahat
#!!!!!!!!!!!!!!!!!!!
def process_image(im, dpt, X, X_first, range_x=2.0, range_y=2.0, dsize=(256,256), shift_x=0.0, shift_y=0.3):
    
    #nl je vzdalenost ramen v pixelech
    
    x = X[0::3]
    y = X[1::3]
    w = X[2::3]
    
    x_first = X_first[0::3]
    y_first = X_first[1::3]
    w_first = X_first[2::3]
    
    #x_center = x[1]
    #y_center = y[1]
    x_center = x_first[1]
    y_center = y_first[1]
    #print("center", x_center, y_center)
    
    
    nl = numpy.linalg.norm(numpy.array([x_first[5]-x_first[2], y_first[5]-y_first[2]])) 
    
    if nl < 30.0:
        print('low nl', nl, 'conf shoulder', w_first[5], w_first[2] )
        nl = 80.0
    
    #print(nl)
   
    
    
    #relative shift in neck
    x_center += shift_x * nl
    y_center += shift_y * nl
    
    row_min = int(y_center - range_y * nl)
    row_max = int(y_center + range_y * nl)
    colum_min = int(x_center - range_x * nl)
    colum_max = int(x_center + range_x * nl)
    #print('crop', row_min, row_max, colum_min, colum_max)
    
    #pridej 100px okraj na vsechny strany
    border_size = 150
    im_border = numpy.zeros([im.shape[0]+2*border_size,im.shape[1]+2*border_size,3], dtype=im.dtype)
    im_border[border_size:im.shape[0]+border_size,border_size:im.shape[1]+border_size,:] = im
    row_min += border_size
    row_max += border_size
    colum_min += border_size
    colum_max += border_size
    #print('crop border', row_min, row_max, colum_min, colum_max)
     
    #plt.imshow(im_border)
    #plt.show()
    
    #plt.imshow(im)
    #plt.show()
    
    im = im_border[row_min:row_max,colum_min:colum_max]
    #dpt = dpt[row_min:row_max,colum_min:colum_max]
    
    im = cv2.resize(im, dsize, interpolation=cv2.INTER_NEAREST)
    #dpt = cv2.resize(dpt, dsize, interpolation=cv2.INTER_NEAREST)
    
    #plt.imshow(im[:,:,::-1])
    #plt.show()
    ##mask
    #mask = dpt != 0.0
    #mask = numpy.expand_dims(mask,axis=2)
    
    #return im*mask, dpt
    return im, dpt
    
    

##########################
# original funkce kterou vznikly data train_crop a val_crop
# !!!!!!!!!!!nesahat
def SaveData(data_dir, istest=False, isval=False):
            
        if istest:
            label_file_name =  data_dir+"test_labels.csv"
            h5_file_name = data_dir+"test_json_keypoints-raw.h5"
            save_dir = data_dir+"test_crop/"
            video_dir_name = data_dir+"test/"
                
        else:
            if not isval:
                label_file_name =  data_dir+"train_labels.csv"
                #label_file_name =  "./train_labels.csv"
                h5_file_name = data_dir+"train_json_keypoints-raw.h5"
                save_dir = data_dir+"train_crop/"
                video_dir_name = data_dir+"train/"
            
            else:
                #source
                label_file_name =  data_dir+"ground_truth.csv"
                h5_file_name = data_dir+"val_json_keypoints-raw.h5"
                save_dir = data_dir+"val_crop/"
                video_dir_name = data_dir+"val/"
                
        
        
        with open(label_file_name) as f:
           content = f.readlines()
        
        data_f = h5py.File(h5_file_name, 'r')
        #print(data_f.keys())
        #exit()
        N = len(content)
        print('N',N)
        
        count = 0
        #import random #kurva to pojebe validaci na codalabu!!!!
        #random.shuffle(content)
        for i, line in enumerate(content):
            
            #if i > 1000:
              #break
            if (i % 100) == 0:
                print("... processing '%s'" % i)
            
            
            key_token, classID = line.split(',')
            classID = int(classID)
            #print(key_token+'_color')
            #exit()
            X = numpy.array(data_f.get(key_token+'_color'))
            
            #print(X.shape)
            #exit()
            
            ##############
            #prepare dirs
            key = key_token.split('_')
            speaker_dir = save_dir+key[0].strip()
            sample_dir = speaker_dir+'/'+key[1].strip()
            #print(speaker_dir, sample_dir)
            if not os.path.isdir(speaker_dir):
                os.mkdir(speaker_dir)
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
    
            
            #newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in key)
            #speaker = newstr.split()
            #speakerID = int(speaker[0])
        
            N_frames = X.shape[0]
            
            im_file_name = video_dir_name+key_token
            #print(im_file_name)
            vidcap_image = cv2.VideoCapture(im_file_name+'_color.mp4')
            #vidcap_depth = cv2.VideoCapture(im_file_name+'_depth.mp4')
                            
            
            for j in range(0, N_frames):
                
                success, image = vidcap_image.read()
                #success, depth = vidcap_depth.read()
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
                #depth = depth[:,:,0]
                ##if not success:
                    ##break
                #print(depth[220:225,220,:])
                #exit()
                im, _ = process_image(image, None, X[j], X[0])
                
                cv2.imwrite("{}/frame_{:03d}.jpg".format(sample_dir, j), im)
                
                #plt.imshow(d)
                #plt.imshow(im[:,:,::-1])
                
                #plt.plot(x[j, 0::3], x[j, 1::3],'b')
                
                #plt.show()
            ###################





##############################
def process_image_mask(im, dpt, X, X_first, range_x=2.0, range_y=2.0, dsize=(256,256), shift_x=0.0, shift_y=0.3):
    
        
        idx_a = [ 0, 
                  8,9, 10,11, 8, 13,14,15, 8, 17,18,19, 8, 21,22,23, 8, 25,26,27,  #leva
                  29, 30, 31, 32, 29, 34, 35, 36, 29, 38, 39, 40, 29, 42, 43, 44, 29, 46, 47, 48] #prava
        idx_b = [ 1,
                  9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 25,26,27,28,  #leva
                  30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49] #prava
        rep = [ 20,
                4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2,
                4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2]
        
        
        x = X[0::3]
        y = X[1::3]
        w = X[2::3]
        
        x_first = X_first[0::3]
        y_first = X_first[1::3]
        w_first = X_first[2::3]
        
        #x_center = x[1]
        #y_center = y[1]
        x_center = x_first[1]
        y_center = y_first[1]
        #print("center", x_center, y_center)
        
        
        #nl je vzdalenost ramen v pixelech
        nl = numpy.linalg.norm(numpy.array([x_first[5]-x_first[2], y_first[5]-y_first[2]])) 
        #print(nl)
        if nl < 30.0:
            print('low nl', nl, 'conf shoulder', w_first[5], w_first[2] )
            nl = 80.0
        
        
        ####################
        #maskovani skeletonu
        vx = x[idx_b] - x[idx_a]
        vy = y[idx_b] - y[idx_a]
        kernel = numpy.ones((3,3),numpy.uint8)
        mask_full = numpy.zeros((im.shape[0], im.shape[1]), dtype=numpy.uint8)
        for w in range(75):
            W = w / 75.0
            row = y[idx_a] + W * vy
            colum = x[idx_a] + W * vx
            for r,c,n in zip(row,colum,rep):
                if (int(r) >= 0) and (int(r) < im.shape[0]) and (int(c) >= 0) and (int(c) < im.shape[1]):
                    mask_full[int(r),int(c)] = n
        
        ###########
        mask = numpy.zeros_like(mask_full)
        for n in [2,4,20]:
            mask_n = cv2.dilate(numpy.array(mask_full==n, dtype=numpy.uint8), kernel,iterations=n)
            #print(mask_n.shape)
            mask = numpy.logical_or(mask, mask_n)
        
        mask = numpy.expand_dims(mask, 2)
        im *= mask
        ###########
        
        #print(mask.dtype)
        #plt.imshow(im * numpy.expand_dims(mask_full,2))
        #plt.imshow(im)
        #plt.plot(x, y)
        #plt.show()
        ############
        
        
        #############
        # resize
        #relative shift in neck
        x_center += shift_x * nl
        y_center += shift_y * nl
        
        row_min = int(y_center - range_y * nl)
        row_max = int(y_center + range_y * nl)
        colum_min = int(x_center - range_x * nl)
        colum_max = int(x_center + range_x * nl)
        #print('crop', row_min, row_max, colum_min, colum_max)
        
        #pridej 150px okraj na vsechny strany
        border_size = 150
        row_min += border_size
        row_max += border_size
        colum_min += border_size
        colum_max += border_size
        #print('crop border', row_min, row_max, colum_min, colum_max)
        
        im_border = numpy.zeros([im.shape[0]+2*border_size,im.shape[1]+2*border_size,3], dtype=im.dtype)
        im_border[border_size:im.shape[0]+border_size,border_size:im.shape[1]+border_size,:] = im
        #mask_border = numpy.zeros([im.shape[0]+2*border_size,im.shape[1]+2*border_size], dtype=im.dtype)
        #mask_border[border_size:im.shape[0]+border_size,border_size:im.shape[1]+border_size] = mask_full
        
        im = im_border[row_min:row_max,colum_min:colum_max]
        #dpt = dpt[row_min:row_max,colum_min:colum_max]
        #mask_full = mask_border[row_min:row_max,colum_min:colum_max]
        
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_CUBIC)
        #mask_full = cv2.resize(mask_full, dsize, interpolation=cv2.INTER_CUBIC)
         
        #plt.imshow(im[:,:,::-1])
        #plt.show()
       
        
        return im, dpt
    

##########################
def SaveDataMask(data_dir, istest=False, isval=False):
            
        
        
        if istest:
            label_file_name =  data_dir+"test_labels.csv"
            h5_file_name = data_dir+"test_json_keypoints-raw.h5"
            save_dir = data_dir+"test_crop_mask/"
            video_dir_name = data_dir+"test/"
                
        else:
            if not isval:
                label_file_name =  data_dir+"train_labels.csv"
                #label_file_name =  "./train_labels.csv"
                h5_file_name = data_dir+"train_json_keypoints-raw.h5"
                save_dir = data_dir+"train_crop_mask/"
                video_dir_name = data_dir+"train/"
            
            else:
                #source
                label_file_name =  data_dir+"ground_truth.csv"
                h5_file_name = data_dir+"val_json_keypoints-raw.h5"
                save_dir = data_dir+"val_crop_mask/"
                video_dir_name = data_dir+"val/"
                
        #if not isforcodalab:
            #label_file_name =  data_dir+"train_labels.csv"
            ##label_file_name =  "./train_labels.csv"
            #h5_file_name = data_dir+"train_json_keypoints-raw.h5"
            #save_dir = data_dir+"train_crop_mask/"
            #video_dir_name = data_dir+"train/"
        
        #else:
            ##source
            #label_file_name =  data_dir+"predictions.csv"
            #h5_file_name = data_dir+"val_json_keypoints-raw.h5"
            #save_dir = data_dir+"val_crop_mask/"
            #video_dir_name = data_dir+"val/"
            
        
        
        with open(label_file_name) as f:
           content = f.readlines()
        
        data_f = h5py.File(h5_file_name, 'r')
        
        N = len(content)
        print('N',N)
        
        count = 0
        #import random #kurva to pojebe validaci na codalabu!!!!
        #random.shuffle(content)
        for i, line in enumerate(content):
            
            #if i > 1000:
              #break
            if (i % 100) == 0:
                print("... processing '%s'" % i)
            
            
            key_token, classID = line.split(',')
            classID = int(classID)
            
            X = numpy.array(data_f.get(key_token+'_color'))
            
            #print(X.shape)
            
            ##############
            #prepare dirs
            key = key_token.split('_')
            speaker_dir = save_dir+key[0].strip()
            sample_dir = speaker_dir+'/'+key[1].strip()
            #print(speaker_dir, sample_dir)
            if not os.path.isdir(speaker_dir):
                os.mkdir(speaker_dir)
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
       
            N_frames = X.shape[0]
            
            im_file_name = video_dir_name+key_token
            #print(im_file_name)
            vidcap_image = cv2.VideoCapture(im_file_name+'_color.mp4')
            #vidcap_depth = cv2.VideoCapture(im_file_name+'_depth.mp4')
                            
            
            for j in range(0, N_frames):
                
                success, image = vidcap_image.read()
                #success, depth = vidcap_depth.read()
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
                #depth = depth[:,:,0]
                ##if not success:
                    ##break
                #print(depth[220:225,220,:])
                #exit()
                im, _ = process_image_mask(image, None, X[j], X[0])
                
                cv2.imwrite("{}/frame_{:03d}.jpg".format(sample_dir, j), im)
                
                #plt.imshow(d)
                #plt.imshow(im[:,:,::-1])
                
                #plt.plot(x[j, 0::3], x[j, 1::3],'b')
                
                #plt.show()
            ################### 
            
        

############################################
if __name__ == '__main__':
    
    data_dir = sys.argv[1]
    
    if not os.path.isdir(data_dir+"train_crop"): 
       os.mkdir(data_dir+"train_crop")
    if not os.path.isdir(data_dir+"val_crop"): 
       os.mkdir(data_dir+"val_crop")
    if not os.path.isdir(data_dir+"test_crop"): 
       os.mkdir(data_dir+"test_crop")
    
    SaveData(data_dir, istest=False, isval=True)
    SaveData(data_dir, istest=False, isval=False)
    SaveData(data_dir, istest=True, isval=False)

    if not os.path.isdir(data_dir+"train_crop_mask"): 
       os.mkdir(data_dir+"train_crop_mask")
    if not os.path.isdir(data_dir+"val_crop_mask"): 
       os.mkdir(data_dir+"val_crop_mask")
    if not os.path.isdir(data_dir+"test_crop_mask"): 
       os.mkdir(data_dir+"test_crop_mask")
    
    SaveDataMask(data_dir, istest=False, isval=True)
    SaveDataMask(data_dir, istest=False, isval=False)
    SaveDataMask(data_dir, istest=True, isval=False)
