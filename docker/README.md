

#data dir share data over all dockers, for example: 
DATADIR="/home/dnn-user/data"

#OpenPose and Crop data
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/openpose

#VLE data
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/pytorch2

#(train)/predict I3D
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/pytorch

#(train)/predict transformer
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/chainer

#ensambling
#see src git repo
