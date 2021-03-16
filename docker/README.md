# Docker file instructions

## Dockers

```
docker build --rm -t chalearn/openpose -f Dockerfile.data_prep .
docker build --rm -t chalearn/pytorch2 -f Dockerfile.pytorch_vle .
docker build --rm -t chalearn/pytorch -f Dockerfile.pytorch_i3d .
docker build --rm -t chalearn/chainer -f Dockerfile.transformer .
```

## Data preparation

Define data dir share data over all dockers, for example: 
```
DATADIR=/home/dnn-user/data
```

### OpenPose and Crop data
```
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/openpose
```
### VLE data
```
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/pytorch2
```

## Train and test data predict

### (train)/predict I3D
```
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/pytorch
```
### (train)/predict transformer
```
docker run -it --rm --gpus all \
		-v $DATADIR:/home/data \
		-e NVIDIA_VISIBLE_DEVICES=0 chalearn/chainer
```

# Ensembling results
```
python3 ./ensemble/evaluate_results.py
```
