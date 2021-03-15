python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/crop/model_best.pth.tar --logdir /home/data/test_csv/crop --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/keyframe/model_best.pth.tar --logdir /home/data/test_csv/keyframe --gpu cuda:0 -b 10 --save_softmax True --workers 0 --keyframes data/keyframes/test_key_frames_16.h5

python test.py --datadir data/test_mask/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/mask/model_best.pth.tar --logdir /home/data/test_csv/mask --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test_mask/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/keyframe_mask/model_best.pth.tar --logdir /home/data/test_csv/keyframe_mask --gpu cuda:0 -b 10 --save_softmax True --workers 0 --keyframes data/keyframes/test_key_frames_16.h5

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/crop_new/model_best.pth.tar --logdir /home/data/test_csv/crop_new --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/keyframe_new/model_best.pth.tar --logdir /home/data/test_csv/keyframe_new --gpu cuda:0 -b 10 --save_softmax True --workers 0 --keyframes data/keyframes/test_key_frames_16.h5

python test.py --datadir data/test_mask/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/mask_new/model_best.pth.tar --logdir /home/data/test_csv/mask_new --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test_mask/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/keyframe_mask_new/model_best.pth.tar --logdir /home/data/test_csv/keyframe_mask_new --gpu cuda:0 -b 10 --save_softmax True --workers 0 --keyframes data/keyframes/test_key_frames_16.h5

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/1/model_best.pth.tar --logdir /home/data/test_csv/1 --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/2/model_best.pth.tar --logdir /home/data/test_csv/2 --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/3/model_best.pth.tar --logdir /home/data/test_csv/3 --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/4/model_best.pth.tar --logdir /home/data/test_csv/4 --gpu cuda:0 -b 10 --save_softmax True --workers 0

python test.py --datadir data/test/ --threed_data --dataset chalearn --backbone_net i3d_resnet -d 50 --pretrained /home/data/models/5/model_best.pth.tar --logdir /home/data/test_csv/5 --gpu cuda:0 -b 10 --save_softmax True --workers 0
