# Issue Documentation

## Matterport 3D data CUDA out of memory

### Issue introduction

ScanNet dataset splits flats to separate rooms, which makes every scene/scan in ScanNet much smaller than a scene/scan in Matterport3D.

- A scene in ScanNet has ~ points
- A scene in Matterport3D has ~ points

### Evaluation

The dataset is Matterport 3D while the model is pretrained on Scannet.
Label conversion between mpcat40 and nyu40 (used by Scannet) is not bijection. Note that there are many categories projected to dummy categories after conversion. For more details, please refer to metadata files in `./metadata`
