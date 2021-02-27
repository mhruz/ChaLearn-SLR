
import paramiko
import h5py
import os
import PIL.Image
import logging
import statistics

import pandas as pd


class ChaLearnDataManager:

    def __init__(self, keypoints_data_file_path, labels_info_df_path, enable_ssh=False, img_path_store=""):
        """
        Initializes the ChaLearnDataManager for processing of the data associated with the ChaLearn competition.

        :param keypoints_data_file_path: Path to the HDF5 file with the body pose landmarks from OpenPose for all the
                                         sign instances
        :param labels_info_df_path: Path to the CSV table with the properties of all the sign instances and
                                    corresponding frames
        :param enable_ssh: Determines whether a connection should be made to the NTIS server for potential data
                           downloading
        :param img_path_store: Path into which the downloaded frames' images should be saved
        """

        self.keypoints_data_file = h5py.File(keypoints_data_file_path, "r")

        self.labels_info_df = pd.read_csv(labels_info_df_path, encoding="utf-8", sep=" ", header=None)
        self.labels_info_df.columns = ["img_file", "vid_file", "frame_index", "label", "signer"]

        self.ssh_enabled = enable_ssh
        self.img_path_store = img_path_store
        if enable_ssh:
            self.__setup_ssh_client()

    def __setup_ssh_client(self):
        """
        Creates a SSH connection to NTIS
        """

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname="", username="mbohacek")

    def download_image(self, remote_name: str, local_name: str):
        """
        Downloads the given image from the AUTSL dataset's training image folder on the disc server.

        :param remote_name: Name of the remote file (image)
        :param local_name: Name for local saving
        """

        if not self.ssh_enabled:
            logging.warning("The SSH connection was not enabled. Downloading aborted.")
            return

        ftp_client = self.ssh_client.open_sftp()
        ftp_client.get("/data-ntis/projects/korpusy_cv/AUTSL/train_jpg/" + remote_name,
                       os.path.join(self.img_path_store, local_name))
        ftp_client.close()

    def get_saved_landmarks(self, sign_sample_id: str, frame: int, convert_to_coordinates: bool = False, normalize: bool = False,
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

        # Check whether the desired frame exists
        if len(self.keypoints_data_file[sign_sample_id]) <= frame:
            return False, 0.0

        # Structure only the raw sample name, if given with video extension
        if sign_sample_id.endswith(".mp4"):
            sign_sample_id = sign_sample_id.replace(".mp4", "")

        result = self.keypoints_data_file[sign_sample_id][frame]

        # Normalize the landmarks positions according to the image if specified
        if normalize:
            if width == 0 or height == 0:
                print("Could not normalize due to the parameters not being passed.")
            else:
                result[::3] /= width
                result[1::3] /= height

        # Calculate the average hand confidence
        average_confidence = statistics.mean(result[2::3][8:50])

        # Convert array into a list of tuples if specified
        if convert_to_coordinates:
            result = list(result)

            x = result[::3]
            y = result[1::3]

            result = list(zip(x, y))

        return result, average_confidence

    def download_and_process_frame(self, sign_sample_id: str, frame: int, convert_to_coordinates: bool = False, normalize: bool =
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
            self.download_image(self.labels_info_df[(self.labels_info_df["vid_file"] == sign_sample_id) & (self.labels_info_df["frame_index"] == frame)]["img_file"].tolist()[0] ,img_name)

        img = PIL.Image.open("img_transfered/" + img_name)
        width, height = img.size

        landmarks_data, average_confidence = self.get_saved_landmarks(sign_sample_id, frame, convert_to_coordinates, normalize, width, height)

        return "img_transfered/" + img_name, img, landmarks_data

    def fav_format_to_structured(self, landmarks) -> (list, list):
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
