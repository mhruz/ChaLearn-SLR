#Location Vectors
python -m mbohacek.annotations_script --video_directory /home/data/test/ \
        --keypoints_datafile /home/data/test_json_keypoints-raw.h5 \
        --labels_info /home/data/test_labels.csv \
        --output_file /home/data/location_vectors_test.h5 \
        --logging normal
