#!/bin/bash
sudo nvidia-docker build -t $USER/pytorch:vgg_extraction .
sudo nvidia-docker run --rm -ti --volume=$(pwd):/app:rw --volume=/mnt/data/alex/data:/data:rw --workdir=/app --ipc=host $USER/pytorch:vgg_extraction /bin/bash
