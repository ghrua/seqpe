"""Prepare the ImageNet dataset"""
import os
import argparse
import tarfile
import pickle
import gzip
import subprocess

import requests
import errno
import shutil
import hashlib
from tqdm import tqdm
import torch
from functools import partial
from multiprocessing import Process
from collections import defaultdict
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


# if needed, please download the orginal fiels from:
# - http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# - http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
_TRAIN_TAR = 'ILSVRC2012_img_train.tar'
_TRAIN_TAR_SHA1 = '43eda4fe35c1705d6606a6a7a633bc965d194284'
_VAL_TAR = 'ILSVRC2012_img_val.tar'
_VAL_TAR_SHA1 = '5f3f73da3395154b60528b2b2a2caf2374f5f178'

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for data processing.")
    
    parser.add_argument("--load_data_dir", type=str, required=True, default="./datasets/imagenet",
                        help="Path to the directory containing input data.")
    parser.add_argument("--save_data_dir", type=str, required=True, 
                        help="Path to the directory where processed data will be saved.")
    parser.add_argument("--worker_num", type=int, default=1,
                        help="The number of workers for processing each file")
    parser.add_argument("--from_hf", action="store_true", 
                        help="whether loading data from huggingface")
    return parser.parse_args()


def check_file(filename, checksum, sha1):
    if not os.path.exists(filename):
        raise ValueError('File not found: '+filename)
    if checksum and not check_sha1(filename, sha1):
        raise ValueError('Corrupted file: '+filename)


def extract_train(tar_fname, target_dir):
    mkdir(target_dir)
    with tarfile.open(tar_fname) as tar:
        print("Extracting "+tar_fname+"...")
        # extract each class one-by-one
        pbar = tqdm(total=len(tar.getnames()))
        for class_tar in tar:
            pbar.set_description('Extract '+class_tar.name)
            tar.extract(class_tar, target_dir)
            class_fname = os.path.join(target_dir, class_tar.name)
            class_dir = os.path.splitext(class_fname)[0]
            os.mkdir(class_dir)
            with tarfile.open(class_fname) as f:
                f.extractall(class_dir)
            os.remove(class_fname)
            pbar.update(1)
        pbar.close()


def extract_images_hf(tar_file, output_dir, worker_id, worker_num):
    # Create the output directory if it doesn't exist.
    mkdir(output_dir)
    log_fname = os.path.basename(tar_file)

    n = 0
    logging.info(f"{log_fname} Proc-{worker_id} | Start...")
    with tarfile.open(tar_file, "r") as tar:
        for i, member in enumerate(tar.getmembers()):
            if (i%worker_num) != worker_id:
                continue
            filename = os.path.basename(member.name)
            if not filename.endswith("JPEG"):
                continue
            parts = filename[:-5].split('_')
            cls = parts[-1]
            class_dir = os.path.join(output_dir, cls)
            mkdir(class_dir)  # Faster than checking with os.path.exists
            # Set new extraction path
            member.name = os.path.join(cls, filename)  # Adjust path for extraction
            tar.extract(member, path=output_dir)  # Extract directly
            n += 1
            if (n%1000) == 0:
                logging.info(f"{log_fname} Proc-{worker_id} | Extracted {n} images")
        logging.info(f"{log_fname} Proc-{worker_id} | Extracted total {n} images")


def extract_val(tar_fname, target_dir):
    mkdir(target_dir)
    print('Extracting ' + tar_fname)
    with tarfile.open(tar_fname) as tar:
        tar.extractall(target_dir)
    # build rec file before images are moved into subfolders
    # move images to proper subfolders
    subprocess.call(["wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash"],
                    cwd=target_dir, shell=True)


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def mkdir(path):
    """make dir exists okay"""
    try:
        os.makedirs(path, exist_ok=True)  # Avoid race conditions
    except FileExistsError:
        pass  # If another process created it, just continue
    except Exception as e:
        print(f"Error creating directory {path}: {e}")


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname


def main(args):
    if args.from_hf:
        p_list = []
        data_args = []
        data_args += [(os.path.join(args.load_data_dir, f'train_images_{i}.tar.gz'), 'train') for i in range(5)]
        data_args += [(os.path.join(args.load_data_dir, "val_images.tar.gz"), 'val')]
        for tar_fname, split in data_args:
            for j in range(args.worker_num):
                p = Process(target=extract_images_hf, args=(tar_fname, os.path.join(args.save_data_dir, split), j, args.worker_num))
                p.start()
                p_list.append(p)
        for p in p_list:
            p.join()
    else:
        train_tar_fname = os.path.join(args.load_data_dir, _TRAIN_TAR)
        val_tar_fname = os.path.join(args.load_data_dir, _VAL_TAR)
        extract_train(train_tar_fname, os.path.join(args.save_data_dir, 'train'))
        extract_val(val_tar_fname, os.path.join(args.save_data_dir, 'val'))

if __name__ == '__main__':
    args = get_args()
    main(args)
