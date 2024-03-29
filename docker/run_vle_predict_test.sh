#create VLE input
#./docker/run_create_data.sh

#predict test VLE Vectors
python /home/experiment/visual_language_embedding/predict.py /home/data/test_hand_images.h5 \
       /home/data/models/vle/vle_mobilenet_pretrained_normalized_resized_aug_ref_images_39.pthepoch_9.tar \
       mobilenet 39 /home/data/vle_hand_crops_test_v2.h5 --open_pose_h5 /home/data/train_json_keypoints-raw.h5 --resize 224 --min_conf 0.55
       
#create Locations Vectors
#./docker/run_locations.sh

