
import paramiko
import h5py
import os
import PIL.Image

import pandas as pd


data_file = h5py.File("/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/train_json_keypoints-raw.h5", "r")

df = pd.read_csv("/Users/matyasbohacek/Documents/Academics/Materials/CVPR SLR ChaLearn/Data/labels_jpg.csv", encoding="utf-8", sep=" ", header=None)[30000:100000]
df.columns = ["img_file", "vid_file", "frame_index", "label", "signer"]

ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname="", username="mbohacek")


def download_image(remote_name: str, local_name: str):
    """
    Downloads the given image from the AUTSL dataset's training image folder on the disc server.

    :param remote_name: Name of the remote file (image)
    :param local_name: Name for local saving
    """

    ftp_client = ssh_client.open_sftp()
    ftp_client.get("/data-ntis/projects/korpusy_cv/AUTSL/train_jpg/" + remote_name, "img_transfered/" + local_name)

    ftp_client.close()


def get_saved_landmarks(sign_sample_id: str, frame: int, convert_to_coordinates: bool = False, normalize: bool = False,
                        width: int = 0, height: int = 0):
    """
    Process the OpenCV body landmarks saved in the .h5 file.

    :param sign_sample_id: Identifier of the specific sign video sample
    :param frame: Frame index
    :param convert_to_coordinates: Determines whether the data should be converted into coordinates (50, 2) or left in
           the original format along with the confidence (150)
    :param normalize: Determines whether to normalize the landmarks according to the image's shape (the data will be
           relative)
    :param width: Width of the image (necessary only if `normalize`==True)
    :param height: Height of the image (necessary only if `normalize`==True)
    :return: Returns the specified landmarks (either as np.array or list of tuples, according to
             `convert_to_coordinates`)
    """

    # Structure only the raw sample name, if given with video extension
    if sign_sample_id.endswith(".mp4"):
        sign_sample_id = sign_sample_id.replace(".mp4", "")

    result = data_file[sign_sample_id][frame]

    # Normalize the landmarks positions according to the image if specified
    if normalize:
        if width == 0 or height == 0:
            print("Could not normalize due to the parameters not being passed.")
        else:
            result[::3] /= width
            result[1::3] /= height

    # Convert array into a list of tuples if specified
    if convert_to_coordinates:
        result = list(result)

        x = result[::3]
        y = result[1::3]

        result = list(zip(x, y))

    return result


def download_and_process_frame(sign_sample_id: str, frame: int, convert_to_coordinates: bool = False, normalize: bool =
                               False) -> (str, PIL.Image, list):
    """
    Combines and orchestrates the necessary processes to process landmarks of the given frame from the desired video
    sample. If the frame is not present, it is downloaded.

    :param sign_sample_id: Identifier of the specific sign video sample
    :param frame: Frame index
    :param convert_to_coordinates: Determines whether the data should be converted into coordinates (50, 2) or left in
           the original format along with the confidence (150)
    :param normalize: Determines whether to normalize the landmarks according to the image's shape (the data will be
           relative)
    :return: Relative path of the image, PIL image, landmarks data
    """

    img_name = sign_sample_id.replace(".mp4", "") + "_" + str(frame) + ".jpg"

    if not os.path.isfile("img_transfered/" + img_name):
        download_image(df[(df["vid_file"] == sign_sample_id) & (df["frame_index"] == frame)]["img_file"].tolist()[0] ,img_name)

    img = PIL.Image.open("img_transfered/" + img_name)
    width, height = img.size

    landmarks_data = get_saved_landmarks(sign_sample_id, frame, convert_to_coordinates, normalize, width, height)

    return "img_transfered/" + img_name, img, landmarks_data


def fav_format_to_structured(landmarks) -> (list, list):
    """
    Converts the format in which the training data is saved into the separated standardized format for the pipeline.

    :param landmarks: List of the landmarks
    :return: Body landmarks (18, 2), hands landmarks (2, 21, 2)

    :warn: The landmarks data should be normalized and in the converted format with shape (50, 2), not (150)
    """

    body = [None] * 18

    for body_index in range(8):
        body[body_index] = landmarks[body_index]

    hands = [[None] * 21] * 2

    hands[0] = landmarks[8:29]
    hands[1] = landmarks[29:50]

    return body, hands


if __name__ == "__main__":
    pass
