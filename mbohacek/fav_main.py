
import random
import logging

from mbohacek.location_analysis import *
from mbohacek.fav_disc_space_management import *

# img_location, img_pil, landmarks_fav_format = download_and_process_frame(random.choice(df["vid_file"].tolist()), 10, True, True)
img_location, img_pil, landmarks_fav_format = download_and_process_frame("signer0_sample1103_color.mp4", 20, True, True)

img = cv2.imread(img_location)

found_body, found_hands = fav_format_to_structured(landmarks_fav_format)
found_face = analyze_face_landmarks(img)[0]

if not found_face:
    logging.error("Could not analyze the face from the image.")
    exit()

results = analyze_hands_areas(found_body, found_hands, found_face)
converted_tensor = area_dictionary_to_tensor(results[0])

print(results)
print(converted_tensor)

cv2.imshow("Visualization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
