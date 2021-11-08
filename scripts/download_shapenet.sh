#!/bin/bash
mkdir -p data
cd data
echo "Start downloading preprocessed ShapeNet data (~220G)"
wget https://s3.eu-central-1.amazonaws.com/avg-projects/shape_as_points/data/shapenet_psr.zip
unzip shapenet_psr.zip
rm shapenet_psr.zip
echo "Done!"