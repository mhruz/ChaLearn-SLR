
from mbohacek.location_analysis import *
from mbohacek.fav_disc_space_management import *


chalearn_data_manager = ChaLearnDataManager("/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/train_json_keypoints-raw.h5",
                                            "/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/labels_jpg.csv", True,
                                            "img_transfered")

img_location, img_pil, landmarks_fav_format = chalearn_data_manager.download_and_process_frame("signer0_sample1103_color.mp4", 20, True, True)

img = cv2.imread(img_location)

found_body, found_hands = chalearn_data_manager.fav_format_to_structured(landmarks_fav_format)
found_face = analyze_face_landmarks(img)[0]

if found_face:
    results = analyze_hands_areas(found_body, found_hands, found_face)
    converted_tensor = area_dictionary_to_tensor(results[0])

    print(results)
    print(converted_tensor)

cv2.imshow("Visualization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
