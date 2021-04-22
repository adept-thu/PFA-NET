#!/usr/bin/bash
docker run -it --gpus all --hostname astyx-pcdet \
	--name astyx-pcdet \
	-v /home/kangle/projects/astyx-pcdet:/src \
	-v /home/kangle/dataset/dataset_astyx_hires2019:/src/data/astyx/testing \
	-v /home/kangle/dataset/dataset_astyx_hires2019:/src/data/astyx/training \
	astyx-pcdet:local bash
