import os
import copy
import numpy
import io
import h5py
import random
from chainer.dataset import DatasetMixin

from pose import pose2D
from vector.semantic_vector_utils import get_semantic_vector_location_hand_crop_keyframes, normalize_hand_image, location_vector_v1_to_v2, get_semantic_vector_location_vle_keyframes, get_semantic_vector_location_vle_keyframes_v2


class HpoesDataset(DatasetMixin):

    def process_x_HZE(self, x):
        dtype = "float32"
        
        Xx = x[0:x.shape[0], 0:(x.shape[1]):3]
        Xy = x[0:x.shape[0], 1:(x.shape[1]):3]
        Xw = x[0:x.shape[0], 2:(x.shape[1]):3]
        
        # Normalization of the picture (x and y axis has the same scale)
        Xx, Xy, q, kx, ky = pose2D.normalization2(Xx, Xy)
        
        # Delete all skeletal models which have a lot of missing parts.
        Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)
        #Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.8, dtype) #toto mi poslalo trenovani do stavu, val data acuuracy 0 :-{(((
        
        # Preliminary filtering: weighted linear interpolation of missing points.
        Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, dtype)
        
        return [Xx, Xy, Xw]
    
    
    ############################
    def process_x_ZKR(self, x):
        
        Xx = x[0:x.shape[0], 0:(x.shape[1]):3]
        Xy = x[0:x.shape[0], 1:(x.shape[1]):3]
        Xw = x[0:x.shape[0], 2:(x.shape[1]):3]
        
        
        #########################
        #dej nulu do krku [1]
        Xx -= Xx[:,1:2]
        Xy -= Xy[:,1:2]
        
        ######################
        #normalizace meritka
        #delka nedominantni paze z prvniho sminku, varianta je take sirka ramen
        #nl = numpy.linalg.norm(numpy.array([Xx[0,2]-Xx[0,3], Xy[0,2]-Xy[0,3]]))
        #norma je sirka ramen z prvniho snimku
        nl = numpy.linalg.norm(numpy.array([Xx[0,5]-Xx[0,2], Xy[0,5]-Xy[0,2]])) 
        #print(nl)
        if nl < 30.0:
            nl = 80.0
        Xx /= nl
        Xy /= nl
        #####################

        ###############
        # Preliminary filtering: weighted linear interpolation of missing points.
        #Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, "float32")
        ################
        
        ############
        #Hand pose
        #nulu dej do wristu
        #Lokace, uz je znormalizovany cely sleketon, takze staci odecist prislusou hodnotu
        LLx = Xx[:,7:8]
        LLy = Xy[:,7:8]
        LRx = Xx[:,4:5]
        LRy = Xy[:,4:5]
        
        HLx = Xx[:,8:29] - LLx
        HLy = Xy[:,8:29] - LLy
        HLw = numpy.mean(Xw[:,8:29], axis=1)
        HRx = Xx[:,29:50] - LRx
        HRy = Xy[:,29:50] - LRy
        HRw = numpy.mean(Xw[:,29:50], axis=1)
        
        ##konfidence
        #conf_t = 0.2
        #HLw = numpy.mean(Xw[:,8:29], axis=1)
        #HLw = numpy.expand_dims(HLw, axis=1)
        #mask = numpy.array(HLw > conf_t, numpy.float32)
        #HLx *= mask
        #HLy *= mask
        #HRw = numpy.mean(Xw[:,29:50], axis=1)
        #HRw = numpy.expand_dims(HRw, axis=1)
        #mask = numpy.array(HRw > conf_t, numpy.float32)
        #HRx *= mask
        #HRy *= mask
        ##########################
        
        ##################
        # filtrace
        last_Hx = HLx[0]
        last_Hy = HLy[0]
        #pro levou
        for i in range(HLx.shape[0]):
          if HLw[i] > 0.3:
            last_Hx = HLx[i]
            last_Hy = HLy[i]
          else:
            HLx[i] = last_Hx
            HLy[i] = last_Hy
          
        last_Hx = HRx[0]
        last_Hy = HRy[0]
        #pro pravou
        for i in range(HRx.shape[0]):
          if HRw[i] > 0.3:
            last_Hx = HRx[i]
            last_Hy = HRy[i]
          else:
            HRx[i] = last_Hx
            HRy[i] = last_Hy
         
          
        
        return [HLx, HLy, HLw, HRx, HRy, HRw, LLx, LLy, LRx, LRy]



    def __init__(self, data_dir="./", input_size=100, max_len= 120, isval=False, isforcodalab=False):
        super().__init__()
       
        self.isval = isval
        self.max_len = max_len
        self.input_size = input_size
        
        if not isforcodalab:
            
            
            if self.isval:
                #source
                #label_file_name =  data_dir+"train_labels_val.csv"
                label_file_name =  data_dir+"ground_truth.csv"
                
                #parametrizace
                #label_file_name_vle =  data_dir+"vle_hand_crops_val.h5"
                label_file_name_vle =  data_dir+"vle_hand_crops_val_v2.h5" #3GB
                label_file_name_loc =  data_dir+"location_vectors_val.h5"  #70MB
                label_file_name_handcrops = data_dir+"val_hand_images.h5"  #5GB
                
                #openpose
                h5_file_name = data_dir+"val_json_keypoints-raw.h5"
                #keyframes
                h5_keyframes_file_name = data_dir+'val_key_frames_16.h5'

                
            else:
                #source
                #label_file_name =  data_dir+"train_labels_train.csv"
                label_file_name =  data_dir+"train_labels.csv"
                
                #parametrizace
                #label_file_name_vle =  data_dir+"vle_hand_crops_train.h5"
                label_file_name_vle =  data_dir+"vle_hand_crops_train_v2.h5" #20GB
                label_file_name_loc =  data_dir+"location_vectors.h5"        #400MB
                label_file_name_handcrops = data_dir+"train_hand_images.h5"  #33 GB
                
                #open pose
                h5_file_name = data_dir+"train_json_keypoints-raw.h5"
                #key_frames
                h5_keyframes_file_name = data_dir+'key_frames_16.h5'
                
               
            
                
        else: #test
            #source
            label_file_name =  data_dir+"test_labels.csv"
            
            #parametrizace
            label_file_name_vle =  data_dir+"vle_hand_crops_test_v2.h5" #2.5GB
            label_file_name_loc =  data_dir+"location_vectors_test.h5"
            label_file_name_handcrops = data_dir+"test_hand_images.h5"  #4,3 GB
            
            #open pose
            h5_file_name = data_dir+"test_json_keypoints-raw.h5"
            #key_frames
            h5_keyframes_file_name = data_dir+'test_key_frames_16.h5' #1.8GB
         
        loc_vectors = h5py.File(label_file_name_loc, "r")
        vle_h5 = h5py.File(label_file_name_vle, "r")
        keyframes = h5py.File(h5_keyframes_file_name, "r")
        #hand_crops = h5py.File(label_file_name_handcrops, "r")

        
        ################
        #nacteni do pameti
        read_to_mem = True
        #self.width = 24 #32   raw maji 28..64
        if read_to_mem:
            self.loc_vectors_data = {}
            self.vle_data = {}
            self.keyframes_data = {}
            self.hand_crops_data = {}

            for i, sample in enumerate(loc_vectors.keys()):
                #print(sample)
                if (i % 500) == 0:
                #if (i > 500):
                    print('reading to memory', i)
                    #break
                # v1
                # loc_vectors_data[sample] = loc_vectors[sample][:]
                # v2
                self.loc_vectors_data[sample] = {}
                #self.loc_vectors_data[sample]["frame_number"] = loc_vectors[sample]["frame_number"][:]
                #self.loc_vectors_data[sample]["vector"] = loc_vectors[sample]["vector"][:]
                #ikurva f isinstance(loc_vectors[sample], numpy.ndarray):
                if not self.isval:
                    self.loc_vectors_data[sample] = location_vector_v1_to_v2(loc_vectors[sample])
                #elif isinstance(loc_vectors[sample], h5py.Dataset):
                else:
                    self.loc_vectors_data[sample]['frame_number'] = loc_vectors[sample]['frame_number']
                    self.loc_vectors_data[sample]['vector'] = loc_vectors[sample]['vector']
                    
                
                #if self.isval:
                    #self.loc_vectors_data[sample]["frame_number"] = loc_vectors[sample]["frame_number"][:]
                    #self.loc_vectors_data[sample]["vector"] = loc_vectors[sample]["vector"][:]
                
                #else:
                    #self.loc_vectors_data[sample] = loc_vectors[sample][:]

                self.vle_data[sample] = {}
                self.vle_data[sample]["left_hand"] = {}
                self.vle_data[sample]["left_hand"]["frames"] = vle_h5[sample]["left_hand"]["frames"][:]
                self.vle_data[sample]["left_hand"]["embeddings"] = vle_h5[sample]["left_hand"]["embeddings"][:]

                self.vle_data[sample]["right_hand"] = {}
                self.vle_data[sample]["right_hand"]["frames"] = vle_h5[sample]["right_hand"]["frames"][:]
                self.vle_data[sample]["right_hand"]["embeddings"] = vle_h5[sample]["right_hand"]["embeddings"][:]

                self.keyframes_data[sample] = keyframes[sample][:]

                #image hand patch
                #self.hand_crops_data[sample] = {}
                #self.hand_crops_data[sample]["left_hand"] = {}
                #self.hand_crops_data[sample]["left_hand"]["frames"] = hand_crops[sample]["left_hand"]["frames"][:]
                #self.hand_crops_data[sample]["left_hand"]["images"] = numpy.array(
                    #[normalize_hand_image(x, self.width) for x in hand_crops[sample]["left_hand"]["images"][:]])

                #self.hand_crops_data[sample]["right_hand"] = {}
                #self.hand_crops_data[sample]["right_hand"]["frames"] = hand_crops[sample]["right_hand"]["frames"][:]
                #self.hand_crops_data[sample]["right_hand"]["images"] = numpy.array(
                    #[normalize_hand_image(x, self.width) for x in hand_crops[sample]["right_hand"]["images"][:]])

        else:
            self.loc_vectors_data = loc_vectors
            self.vle_data = vle_h5
            self.keyframes_data = keyframes
            #self.hand_crops_data = hand_crops
        
        
        #exit()
        ##f self.isval:
        #key = 'signer11_sample104_color'
        ##    key = 'signer0_sample234_color'
        #print(self.loc_vectors_data[key])
        ##exit()
        #data = get_semantic_vector_location_hand_crop_keyframes(self.loc_vectors_data[key], self.hand_crops_data[key], self.keyframes_data[key], self.width)
        #print(data.shape)
        #exit()
        
        
        with open(label_file_name) as f:
            self.content = f.readlines()
            
            
        self.N = len(self.content)
        print('self.N', self.N)
            
            
        
    def __len__(self):
        
        #return 500
        return self.N
    
    
    

    #########################
    def get_example(self, i):
      
        #if self.isval:
            #print(self.loc_vectors_data.keys())
            #exit()
        #exit()
        #i = 
        #print('i',i)
        line = self.content[i]

        key, classID = line.split(',')

        classID = numpy.int32(classID)
        
        key = key+'_color'
        
        ##print(key)
         
        #if self.isval:
            #key = 'signer11_sample544_color'
        #else:
            #key = 'signer0_sample230_color'
        
        #x = numpy.array(self.data_f.get(key))
        ##print(x.shaped)
        #[HLx, HLy, HLw, HRx, HRy, HRw, LLx, LLy, LRx, LRy] = self.process_x_ZKR(x)
        #N_frames = HLx.shape[0]
        #print(N_frames)
        
        
        #data = self.get_semantic_vector_location_vle_keyframes(self.data_h5file_loc[key], self.data_h5file_vle[key], self.data_keyframes[key])
        #data = self.get_semantic_vector_location_vle(self.data_h5file_loc[key], self.data_h5file_vle[key])
        #data = self.data_h5file_loc[key]
        #data = get_semantic_vector_location_hand_crop_keyframes(self.loc_vectors_data[key], self.hand_crops_data[key], self.keyframes_data[key], self.width)
        data = get_semantic_vector_location_vle_keyframes_v2(self.loc_vectors_data[key], self.vle_data[key],  self.keyframes_data[key])
        
        #print(data.shape)
        #exit()
        
        data = numpy.array(data, dtype=numpy.float32)
        
        data = numpy.reshape(data,(data.shape[0], -1))
        #print(data.shape)
        #exit()
        
        
        ###############
        ##augmentace, kazdy druhy
        #ef = data.shape[0]
        #if ef > self.max_len:
            #ef = self.max_len
        
        #sf = random.randint(0,5)
        #if (ef - sf) > 35:
            #sf += 10
            #ef -= 15
        
        #data = data[sf:ef:2]
                
        
        ####################
        ##vizualizace
        #import cv2
        #import matplotlib.pyplot as plt
        #video_dir_name = data_dir+"/train/"
        #im_file_name = video_dir_name+key+'.mp4'
        #print(im_file_name)
        #vidcap = cv2.VideoCapture(im_file_name)
                        
              
        ##print(H.shape)
        #labels = self.labels[idx,1]
        #print(labels.shape)
        #exit()
        
        return {
            "data": data.copy(),
            "label": classID,
        }
      
      
     
