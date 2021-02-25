
"""
Script for downloading random frames from the disc server, from which the face could have been analyzed with dlib.
"""

import random

from mbohacek.location_analysis import *
from mbohacek.fav_disc_space_management import *

chalearn_data_manager = ChaLearnDataManager("/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/train_json_keypoints-raw.h5",
                                            "/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/labels_jpg.csv", True,
                                            "img_transfered")
downloaded_frames = []

while len(downloaded_frames) < 25:
    try:
        img_location, img_pil, landmarks_fav_format = chalearn_data_manager.download_and_process_frame(random.choice(chalearn_data_manager.labels_info_df["vid_file"].tolist()), 20, True, True)
    except:
        print("Error occured. Skipping iteration")
        continue

    if img_location in downloaded_frames:
        continue

    img = cv2.imread(img_location)

    try:
        found_body, found_hands = chalearn_data_manager.fav_format_to_structured(landmarks_fav_format)
        found_face = analyze_face_landmarks(img)[0]
    except:
        os.remove(img_location)
        print("One frame discarded, as face was not identified.")
        continue

    if not found_face:
        os.remove(img_location)
        print("One frame discarded, as face was not identified.")
    else:
        downloaded_frames.append(img_location)
        print("One Frame added.", str(len(downloaded_frames)), "/25")
