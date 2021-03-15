import os
import copy
import numpy
import io
import h5py
import random
from chainer.dataset import DatasetMixin

from pose import pose2D

import matplotlib.pyplot as plt


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
    
    
    ###############################
    # tato funkce se zda byt dobra
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
        # odecist pozici wristu
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
        # filtrace spatne hand pose
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
         #####################
          
        
        return [HLx, HLy, HLw, HRx, HRy, HRw, LLx, LLy, LRx, LRy]



    def __init__(self, data_dir="./", input_size=100, max_len= 120, isval=False, isforcodalab=False):
        super().__init__()
       
        self.isval = isval
        self.max_len = max_len
        self.input_size = input_size
        self.isforcodalab = isforcodalab
        
        ##################
        ## pro rozsirenou train o 4 z val .... je to v cache_ex
        #if not isforcodalab:
            #if self.isval:
                ##source
                #h5_file_name = "./cache_ex/val_json_keypoints-raw_ex.h5"
                #label_file_name =  "./cache_ex/ground_truth_ex.csv"
                ##cache
                #datah5filename = './cache_ex/data_val_ex.h5' #s tim vzniklo odevzdavaci vysledek
            #else:
                ##source
                #h5_file_name = "./cache_ex/train_json_keypoints-raw_ex.h5"
                #label_file_name =  "./cache_ex/train_labels_ex.csv"
                ##cache
                #datah5filename = './cache_ex/data_train_ex.h5' #s tim vzniklo odevzdavaci vysledek
           
        #else:
            ##source
            #h5_file_name = data_dir+"test_json_keypoints-raw.h5"
            #label_file_name =  data_dir+"test_labels.csv"
            ##cache
            #datah5filename = './cache/data_pred.h5'  #s tim vzniklo odevzdavaci vysledek
            
        ##############
        #odevzdavaci, pouziva ./cache
        if not isforcodalab:
            if self.isval:
                #source
                h5_file_name = data_dir+"val_json_keypoints-raw.h5"
                label_file_name =  data_dir+"ground_truth.csv"
                #cache
                datah5filename = './cache/data_val.h5' #s tim vzniklo odevzdavaci vysledek
            else:
                #source
                h5_file_name = data_dir+"train_json_keypoints-raw.h5"
                label_file_name =  data_dir+"train_labels.csv"
                #cache
                datah5filename = './cache/data_train.h5' #s tim vzniklo odevzdavaci vysledek
           
        else:
            #source
            h5_file_name = data_dir+"test_json_keypoints-raw.h5"
            label_file_name =  data_dir+"test_labels.csv"
            #cache
            datah5filename = './cache/data_pred.h5'  #s tim vzniklo odevzdavaci vysledek
        
        
        ################################
        if os.path.isfile(datah5filename):
            print('Reading cache to memory')
            data_h5file = h5py.File(datah5filename, 'r')
            
            self.data_handpose = data_h5file["data_handpose"][:]
            self.data_location = data_h5file["data_location"][:]
            self.labels = data_h5file["labels"][:]
            data_h5file.close()
            print('N', self.labels.shape[0])
            
        else: #data nejsou pripravena
            with open(label_file_name) as f:
                content = f.readlines()
            
            data_f = h5py.File(h5_file_name, 'r')
            
            N = len(content)
            print('self.N',N)
            
            ###########
            M = N*max_len
            input_size = 42
            print('M=', M)
            self.data_handpose = numpy.zeros((M, 2, input_size), numpy.float32)
            self.data_location = numpy.zeros((M, 2, 2), numpy.float32)
            self.labels = numpy.zeros((M, 2), numpy.int32)
            print('data alocated')
            
            count = 0
            #import random #kurva to pojebe validaci na codalabu!!!!
            #random.shuffle(content)
            for i, line in enumerate(content):
                
                key, classID = line.split(',')
                if (i % 100) == 0:
                    print("... processing '%s'" % i)
                key = key+'_color'
                #self.keys.append(key)
                
                x = numpy.array(data_f.get(key))
                #print(x.shaped)
                
                [HLx, HLy, HLw, HRx, HRy, HRw, LLx, LLy, LRx, LRy] = self.process_x_ZKR(x)
                
                N_frames = HLx.shape[0]
                
                
                ####################
                ##vizualizace
                #import cv2
                #import matplotlib.pyplot as plt
                #video_dir_name = data_dir+"/train/"
                #im_file_name = video_dir_name+key+'.mp4'
                #print(im_file_name)
                #vidcap = cv2.VideoCapture(im_file_name)
                
                ###################
                ###prohod pravou/levou
                ###a = -b udela kopii
                ###a = b  udela odkaz
                ##x = -HLx
                ##y = HLy
                ##HLx = -HRx
                ##HLy = HRy
                ##HRx = x
                ##HRy = y
                
                #for j in range(0, N_frames):
                    
                    #success,image = vidcap.read()
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ##if not success:
                        ##break
                    #plt.imshow(image)
                    ##print(data_item)
                    ##exit()
                    #print(HLw[j], HRw[j])
                    ##plt.plot(Xx[j] * 200, Xy[j] * 200,'r')
                    #plt.plot(HLx[j] * 100, HLy[j] * 100,'r')
                    #plt.plot(HRx[j] * 100, HRy[j] * 100,'b')
                    ##print(Hw[j])
                    
                    ##print(Xx, Xy)
                    ##count += 1
                    
                    #plt.show()
                ###################
                    
                
                
                #print(N_frames)
                
                ##kopirovani cele sekvence
                self.data_handpose[count:count+N_frames, 0] = numpy.hstack([HLx,HLy])
                self.data_handpose[count:count+N_frames, 1] = numpy.hstack([HRx,HRy])
                
                self.data_location[count:count+N_frames, 0] = numpy.hstack([LLx,LLy])
                self.data_location[count:count+N_frames, 1] = numpy.hstack([LRx,LRy])
                
                self.labels[count:count+N_frames, 0] = i
                self.labels[count:count+N_frames, 1] = numpy.int32(classID)
                
                count += N_frames
            
            
            #jen ty nakopirovane
            self.data_handpose = self.data_handpose[:count]
            self.data_location = self.data_location[:count]
            self.labels = self.labels[:count]
            
            #################
            # ulozeni na priste
            data_h5file = h5py.File(datah5filename, 'w')
            data_h5file.create_dataset(name="data_handpose", shape=self.data_handpose.shape, dtype=self.data_handpose.dtype)
            data_h5file.create_dataset(name="data_location", shape=self.data_location.shape, dtype=self.data_location.dtype)
            data_h5file.create_dataset(name="labels", shape=self.labels.shape, dtype=self.labels.dtype)
            
            data_h5file["data_handpose"][:] = self.data_handpose[:]
            data_h5file["data_location"][:] = self.data_location[:]
            data_h5file["labels"][:] = self.labels[:]
            data_h5file.close()
            
        #pocet znaku
        self.N = numpy.max(self.labels[:,0]) + 1

    def __len__(self):
        
        return self.N
        

    def get_example(self, i):
        
        
        #print(Xx.shape, Xy.shape, self.Y[i])
        #exit()
        
        ##augmentace
        #shift = numpy.random.randint(-10, 10 , 1, dtype=numpy.int32)
        #data_aug = numpy.roll(self.data[i], shift, axis=0) #roll dela kopii
        
        
        #data = self.data[i] + noise_all
        #data[:50] += noise_root[0]
        #data[50:] += noise_root[1]
        
        
        
        #with numpy.printoptions(edgeitems=50):
        #print(self.data[i])
        #print(noise)
        #exit()
        
        #print(data.shape, data.dtype)
        #print(labels.shape, labels.dtype)
        #exit()
        
        idx = self.labels[:,0] == i
        
        ##vyber vse !!!!
        #idx[:] = 1
        
        H = self.data_handpose[idx]
        L = self.data_location[idx]
        
        
        ###################
        #orez na max_len ... mozna udelal chytreji, zacatek konec, sude liche apod 
        ##############

        #augmentace, kazdy druhy

        ef = H.shape[0]

        if ef > self.max_len:
            ef = self.max_len
        
        #augmentace vyberu frejmu
        sf = 0
        #if (not self.isforcodalab) and (not self.isval):
        if False:
            sf = random.randint(0,5)
        
        if (ef - sf) > 35:
            sf += 10
            ef -= 15

        H = H[sf:ef:2]
        L = L[sf:ef:2]
        ##############################
        
        #vyber jen liche nebo sude
        #if random.rand()
        
        #print(H.shape[0])
        #####################
        #dropout na indexu sekvence, 15% 25% vyhodit z celkove delky 240 vyhodit, kratke bude zachovavat
        #if not self.isval: je to o malinko horsi nez kdyz i pro val se to udela jako pro train
        if False:
            idxidx = numpy.random.rand(self.max_len * 2) > 0.25
            nl = H.shape[0]
            H = H[idxidx[:nl]]
            L = L[idxidx[:nl]]
        ####################
        
        #print(H.shape[0])
        #exit()
        
        ##########################
        #augmentace leva/prava znakuje
        #zde je hnadpose jeste nulova do writu
        #if not self.isval: je to o malinko horsi nez kdyz i pro val se to udela jako pro train
        if (not self.isforcodalab) and (not self.isval):  #no nedelejme zavery, toto pro vysokem natrenovani dava lepsi
        #if True:  #ano ano ano, i na val to davej
        #if False: #finetunig ?????
            if random.random() < 0.5:
                H[:, [0,1]] = H[:, [1,0]] # prohod ruku
                H[:, :, :21] *= -1.0 #otoc u obou znamenko x souradnice, x-flip celeho skeletonu
                L[:,[0,1]] = L[:,[1,0]] # a i u lokace
                L[:,:,0] *= -1.0
        ####################
        
        ##########################
        #augmentace posun wristu std 0.1 je cca +-20% sirky  ramen ... validace neroste :((((((((( kurva, keruva kurva nebrat
        if (not self.isforcodalab) and (not self.isval):
        #if True:
        #if False:
            #L *= numpy.random.normal(1.0, 0.05, (1,2)) # 
            #L += numpy.random.normal(0.0, 0.05, (1,2))
            L *= numpy.random.normal(1.0, 0.05, (1,)) # 0.05 dava az cca +-10%
            L += numpy.random.normal(0.0, 0.05, (1, 1, 2)) #snad to dore broadcastuje, x a y jina nahoda, ale pro obe ruce spolecny ... dotyky
            #H *= numpy.random.normal(1.0, 0.05, (1,)) #uniforme vsechny stejnym cislem
            H[:, :, :21] *= numpy.random.normal(1.0, 0.05, (1,)) #scale x jinak
            H[:, :, 21:] *= numpy.random.normal(1.0, 0.05, (1,)) #scale y jinak
            
        #print(H.shape) #(30, 2, 42)
        #exit()
        
        ######################
        ##vyber z hanspose jen neco
        #H = H[:,:,[4,8,9,12,20]]
        #print(H.shape)
        #exit()
        nH = H.shape[2] // 2
        #lokaci pricist k hand pose pozici
        H[:, 0, :nH] += L[:,0,0:1] #x
        H[:, 0, nH:] += L[:,0,1:2] #y
        H[:, 1, :nH] += L[:,1,0:1] #x
        H[:, 1, nH:] += L[:,1,1:2] #y
        #######################
        
        ##kresleni
        #for h in H:
            #plt.plot(h[0, :nH], -h[0, nH:])
            #plt.plot(h[1, :nH], -h[1, nH:],'r')
        #plt.show()
        
        ######################
        #data od obou rukou poskladej do vektoru
        data = numpy.reshape(H, (H.shape[0], -1))
        
        ##################
        #normalizace
        #mean, std = numpy.mean(data,axis=0), numpy.std(data,axis=0) 
        #print('mean', mean,'std', std)
        #mean = [ 0.21308188,  0.28273147,  0.4301371,   0.3440964,   0.47026107, -0.01294663, -0.05886922, -0.24841164, -0.1652384,  -0.3612619 ]
        #std = [0.11685108, 0.16854945, 0.12182809, 0.16714302, 0.19735824, 0.04335316, 0.06185774, 0.05963062, 0.06458344, 0.07533549]
        #data -= mean
        #data /= std
        #print('mean', numpy.mean(data,axis=0),'std', numpy.std(data,axis=0) )
        
        #print(data.shape)
        #exit()
    
        #############
        ##kombinuj
        #data = numpy.concatenate([H, L], axis=2)
        #data = numpy.reshape(data, (data.shape[0], -1))
        #mean =  [-0.2597802,  -0.24768949, -0.10408389, -0.20723517, -0.12236418 , 0.5095484,  0.5955113,   0.26567262,  0.24567293,  0.10092048,  0.19360399,  0.07872532, -0.4478477,  0.9278501 ]
        #std = [0.15678214, 0.18309686, 0.12376345, 0.18535215, 0.19786257, 0.26969045, 0.60690176, 0.1383375,  0.16206872, 0.11601944, 0.17762652, 0.1986605, 0.2420555,  0.54219073]
        #data -= mean
        #data /= std
        ############
        
        ##############
        ### pridej delta a delta-delta parametry
        #velocity = numpy.zeros(data.shape, dtype=data.dtype)
        #acceleration = numpy.zeros(data.shape, dtype=data.dtype)
        #velocity[1:] =  data[1:] - data[:-1] 
        #acceleration[1:] = velocity[1:] - velocity[:-1] 
        ##print(data.shape)
        ##print(velocity.shape)
        #data = numpy.hstack([data, velocity, acceleration])
        ##print(data.shape)
        ##exit()
        ###################
        
        #print('mean', numpy.mean(data, axis=0))
        #print('std', numpy.std(data, axis=0))
        #exit()
        
        #data += numpy.random.normal(0, 0.001, data.shape)
        
        #print(L.shape, H.shape)
        #print(data.shape)
        #exit()
        
        #data = numpy.vstack([H,L])
        
        #print(H.shape)
        labels = self.labels[idx, 1] #.. cela sekvence ale to je OK, beru stejne jen prvni cislo
        #print(labels.shape)
        #exit()
        
        return {
            "data": data.copy(),
            "label": labels[0],
        }
      
      
     
