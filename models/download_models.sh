#!/usr/bin/env sh
#
# This script downloads and unpacks the model files, solver
# definitions, and learned network weights. 

echo "Downloading..."

wget https://wustl.box.com/shared/static/s3wbmwqelqe6po8qy4dm1sevhyht9xb7.zip 

echo "Unzipping..."

unzip deeplyfound.zip

echo "Done."
