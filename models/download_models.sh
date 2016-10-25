#!/usr/bin/env sh
#
# This script downloads and unpacks the model files, solver
# definitions, and learned network weights. 

echo "Downloading..."

wget http://amos.csr.uky.edu/modelzoo/deeplyfound/deeplyfound.zip

echo "Unzipping..."

unzip deeplyfound.zip

echo "Done."
