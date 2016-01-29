#!/usr/bin/env sh
#
# This script downloads the model binaries. 

echo "Downloading..."

wget http://amos.csr.uky.edu/modelzoo/deeplyfound/cvplaces.caffemodel -P ./cvplaces 

echo "Downloading..."

wget http://amos.csr.uky.edu/modelzoo/deeplyfound/mcvplaces.caffemodel -P ./mcvplaces 

echo "Done."
