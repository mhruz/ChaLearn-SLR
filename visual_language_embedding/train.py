import argparse
import h5py

if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Train VLE on ChaLearn-SLR data.')
    parser.add_argument('hands_h5', type=str, help='path to dataset with hands')
    parser.add_argument('output', type=str, help='path to output network')
    args = parser.parse_args()

    f = h5py.File(args.hands_h5, "r")

    
