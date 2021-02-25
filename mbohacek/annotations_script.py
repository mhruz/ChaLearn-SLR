
import argparse

from mbohacek.fav_disc_space_management import *
from mbohacek.location_analysis import *


# Arguments
parser = argparse.ArgumentParser("Hand Location Analysis: Annotations Script", add_help=False)

parser.add_argument("--images_directory", type=str, default="img_transfered", help="Path to the directory with the "
                    "images of the individul frames")
parser.add_argument("--keypoints_datafile", type=str, default="/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/train_json_keypoints-raw.h5",
                    help="Path to the HDF5 file with the body pose landmarks from OpenPose for all the sign instances")
parser.add_argument("--labels_info_path", type=str, default="/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/labels_jpg.csv",
                    help="Path to the CSV table with the properties of all the sign instances and corresponding frames")
parser.add_argument("--output_file", type=str, default="out.h5", help="Path to the HDF5 file into which the arrays with"
                    " the soft vectors for each sign video sample will written.")
parser.add_argument("--download", type=bool, default=False, help="Determines whether to download the frame images from "
                    "the NTIS server.")
parser.add_argument("--logging", type=str, choices=["normal", "full"], default="full", help="Sets the extensiveness of "
                    "the logging. If set to full, additional information concerning the sign instances and their frames"
                    " is logged during the annotation.")

args = parser.parse_args()

# MARK: Properies
chalearn_data_manager = ChaLearnDataManager(args.keypoints_datafile, args.labels_info_path, args.download,
                                            args.images_directory)
output_datafile = h5py.File(args.output_file, 'w')
print("Successfully loaded all of the supporting files.")

# Group the instances by the individual videos
grouped_sign_vid_instances = chalearn_data_manager.labels_info_df.groupby(chalearn_data_manager.labels_info_df.vid_file)
print("Found", grouped_sign_vid_instances.ngroups, "individual sign instances.")

print("Starting the annotation process.")

for sign_sample_id, group_subdf in grouped_sign_vid_instances:

    locations_soft_vec_array = np.empty(shape=(0, 2, 15))

    if args.logging == "full":
        print("\tCurrently annotating `" + sign_sample_id + "`. Found " + str(group_subdf.shape[0]) + " frames.")

    # Reset the indexing of the group to discard the original full-dataframe order
    group_subdf = group_subdf.reset_index()

    for row_index, row in group_subdf.iterrows():
        file_name = row["img_file"]
        local_file_name = os.path.join(args.images_directory, file_name)

        # Download the frame image from server if relevant flag is set
        if args.download:
            chalearn_data_manager.download_image(file_name, file_name)

        img = cv2.imread(local_file_name)
        height, width, _ = img.shape

        # Fetch the landmarks and the confidence saved from the OpenPose
        landmarks_fav_format, average_confidence = chalearn_data_manager.get_saved_landmarks(sign_sample_id.replace(".mp4", ""), row_index, True, True, width, height)

        # Convert the landmarks to relevant format and analyze face landmarks
        found_body, found_hands = chalearn_data_manager.fav_format_to_structured(landmarks_fav_format)
        try:
            found_face = analyze_face_landmarks(img)[0]
        except:
            # TODO: This should be thoroughly investigated and reimplemented after the competition
            # If no face at all was found at all, we expect that the signer has covered the face with both of their
            # hands and thus distribute the weight into the relevant face locations
            vec_covered_face = np.array([[0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0]])
            locations_soft_vec_array = np.append(locations_soft_vec_array, [vec_covered_face], axis=0)
            print("-> Prevented")
            continue

        if found_face:
            # Perform the location analysis and structure the resulting soft vector into a np.array
            results = analyze_hands_areas(found_body, found_hands, found_face, landmarks_analysis_confidence=average_confidence)
            converted_tensors = [area_dictionary_to_tensor(result) for result in results]
            converted_array = np.array([tensor.detach().numpy() for tensor in converted_tensors])
            locations_soft_vec_array = np.append(locations_soft_vec_array, [converted_array], axis=0)
        else:
            if args.logging == "full":
                print("\tFor one frame, face could not have been analyzed and thus its analysis it will be invalid.")

            # In case that the face could not have been analyzed, add fallback vectors pointing to neutral space
            fallback_vec = np.zeros((2, 15))
            fallback_vec[0][0] = 1
            fallback_vec[0][1] = 1
            locations_soft_vec_array = np.append(locations_soft_vec_array, [fallback_vec], axis=0)

    output_datafile.create_dataset(sign_sample_id.replace(".mp4", ""), data=locations_soft_vec_array)

print("Annotation finished.")

output_datafile.close()

print("The data was successfully written into a HDF5 file.")
