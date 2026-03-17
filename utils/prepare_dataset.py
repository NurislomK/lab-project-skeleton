import os
import shutil
import urllib.request
import zipfile

def download_dataset():
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"

    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tiny-imagenet")

def prepare_val_dataset():
    annotations_path = 'tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt'
    val_images_dir = 'tiny-imagenet/tiny-imagenet-200/val/images'

    with open(annotations_path) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')

            src_path = os.path.join(val_images_dir, fn)
            dst_dir = os.path.join('tiny-imagenet/tiny-imagenet-200/val', cls)
            dst_path = os.path.join(dst_dir, fn)

            os.makedirs(dst_dir, exist_ok=True)

            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)

    if os.path.exists(val_images_dir):
        shutil.rmtree(val_images_dir)

if __name__ == "__main__":
    download_dataset()
    prepare_val_dataset()