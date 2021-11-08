#!/bin/bash
mkdir -p data
cd data
echo "Start downloading ..."
wget https://s3.eu-central-1.amazonaws.com/avg-projects/shape_as_points/data/demo.zip
unzip demo.zip
rm demo.zip
echo "Done!"