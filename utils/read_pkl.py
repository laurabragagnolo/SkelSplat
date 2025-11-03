import pickle
import numpy as np


def read_pkl(file_path):
    """
    Reads a pickle file and returns the data.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        data: The data contained in the pickle file.
    """
    with open(args.file_path, 'rb') as f:
        while True:
            try:
                obj = pickle.load(f)
                return obj
            except EOFError:
                break  # end of file reached


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read a pickle file.")
    parser.add_argument("--file_path", type=str, help="Path to the pickle file to read.")
    args = parser.parse_args()

    data = read_pkl(args.file_path)
    print(data["testing_data"])

    data_5 = np.load("pose_5.npy")
    data_6 = np.load("pose_6.npy")
