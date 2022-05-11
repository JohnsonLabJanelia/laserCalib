# laserCalib 

## Install

Use conda to manage python virutal environment 

```
conda create -n lasercalib python=3.9
conda activate lasercalib
conda install numpy
conda install -c anaconda scipy 
conda install -c conda-forge matplotlib 
conda install scikit-image
conda install -c menpo opencv
python -m pip install -U prettytable
```
## TODO 

about 20 minutes to run the centroid extraction

## Reference 
The library is built on pySBA -- Python Bundle Adjustment library: https://github.com/jahdiel/pySBA

Camera visualizer -- https://github.com/demul/extrinsic2pyramid
