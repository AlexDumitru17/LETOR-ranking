import tensorflow as tf
import tensorflow_datasets as tfds

def load_and_save_data():
    # 10k
    print("started")
    ds = tfds.load("mslr_web")


def main():
    load_and_save_data()
    print("Hello World!")


if __name__ == "__main__":
    main()
