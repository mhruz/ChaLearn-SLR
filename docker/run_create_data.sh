# generate test hand crop images
python open_pose_hand_grabber.py /home/data/test /home/data/test_json_keypoints-raw.h5 --out_h5 /home/data/test_hand_images.h5
# generate test key-frames
python key_frame_extractor.py /home/data/test_json_keypoints-raw.h5 16 /home/data/test_key_frames_16.h5
