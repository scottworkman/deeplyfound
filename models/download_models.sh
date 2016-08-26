#!/usr/bin/env sh
#
# This script downloads and unpacks the model files, solver
# definitions, and learned network weights. 

echo "Downloading..."

wget 'https://doc-0g-5g-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/720qsepsptd6en0r524ks3685untmfg8/1472248800000/05018003348500916163/*/0BzEcTtT1A2ILUUdLaFRWcVQ0SUU?e=download' -O 'deeplyfound.zip'

echo "Unzipping..."

unzip deeplyfound.zip

echo "Done."
