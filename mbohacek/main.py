
from location_analysis import *

img = cv2.imread("img/x12.jpg")

found_hands = analyze_hand_landmarks(img)[0]
found_body = convert_mp_to_lim_op(analyze_body_landmarks(img)[0])
found_face = analyze_face_landmarks(img)[0]

results = analyze_hands_areas(found_body, found_hands, found_face)
converted_tensor = area_dictionary_to_tensor(results[0])

print(results)
print(converted_tensor)

cv2.imshow("Visualization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
