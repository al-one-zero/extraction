from .preprocessing import load_txt_file, preprocess_dataset
from .training import load_data

if __name__ == "__main__":
    df = preprocess_dataset(load_txt_file("data/train.txt"), "data/lid.176.bin")
    print(df)
    # training workflow
