import os.path as osp
import subprocess
from glob import glob
from os import makedirs
from shutil import rmtree

import extract_vgg_features as vgg
from tqdm import tqdm

VID_ROOT = '/data/videos'
TMP_DIR = '/data/vgg_tmp'
OUT_DIR = '/data/vgg_features'
FPS = 4


def get_videos(root):
    return glob(osp.join(root, '*.mp4'))


def reset_tmp_dir(tmp_dir=TMP_DIR):
    if osp.isdir(tmp_dir):
        rmtree(tmp_dir)
    makedirs(tmp_dir)
    makedirs(osp.join(tmp_dir, 'frames'))


def vid_to_tmp(vid_path, tmp_path=TMP_DIR):

    out_pattern = osp.join(tmp_path, '%6d.png')
    cmd = ' '.join(['ffmpeg',
                    '-i', vid_path,
                    '-vf fps='+str(FPS),
                    out_pattern
                    ])

    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    if not osp.exists(OUT_DIR):
        makedirs(OUT_DIR)
    vids = get_videos(VID_ROOT)
    for v in tqdm(vids):
        vid_to_tmp(v)
        out_path = osp.join(OUT_DIR, osp.splitext(osp.basename(v))[0])
        vgg.pipeline(TMP_DIR, out_path)
        reset_tmp_dir()
