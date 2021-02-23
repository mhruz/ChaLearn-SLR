
"""
Script for downloading random frames from the disc server, from which the face could have been analyzed with dlib.
"""

import random

from mbohacek.location_analysis import *
from mbohacek.fav_disc_space_management import *

downloaded_frames = []

while len(downloaded_frames) < 15:
    try:
        img_location, img_pil, landmarks_fav_format = download_and_process_frame(random.choice(df["vid_file"].tolist()), 20, True, True)
    except:
        print("Error occured. Skipping iteration")
        continue

    if img_location in downloaded_frames:
        continue

    img = cv2.imread(img_location)

    found_body, found_hands = fav_format_to_structured(landmarks_fav_format)
    found_face = analyze_face_landmarks(img)[0]

    if not found_face:
        os.remove(img_location)
        print("One frame discarded, as face was not identified.")
    else:
        downloaded_frames.append(img_location)
        print("One Frame added.", str(len(downloaded_frames)), "/25")
