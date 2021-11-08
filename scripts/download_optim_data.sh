#!/bin/bash
mkdir -p data
cd data
echo "Start downloading data for optimization-based setting (~200 MB)"
wget https://s3.eu-central-1.amazonaws.com/avg-projects/shape_as_points/data/optim_data.zip
unzip optim_data.zip
rm optim_data.zip
echo "Done!"