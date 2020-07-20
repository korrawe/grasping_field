# Grasping Field: Learning Implicit Representations for Human Grasps - demo

This repository contains a grasp generation demo of "Grasping Field: Learning Implicit Representations for Human Grasps"

The following code samples hand grasps conditioned on given object meshes.
![Teaser Image](https://github.com/korrawe/grasping_field_demo/blob/master/resources/sample_hands.png)

## Required library:
- pytorch (with GPU)
- trimesh
- plyfile
- scikit-image
- sklearn
- chumpy
- cv2

You can run the following command in the command line to install all the dependencies.

    pip3 install -r requirements.txt

The code is tested on Ubuntu 18.04 with python3

## Download MANO pickle data-structures

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format `mano_v*_*.zip`). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the `models` folder into the `./mano` folder
- Your folder structure should look like this:
```
  mano/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      ...
    webuser/
    ...
```

## Sample hands

    python3 reconstruct.py

This may take about 20 minutes.

The demo takes objects from `./input` as input, generates 5 samples of grasping hand under `./output`. `./output/meshes/` contains the raw sdf reconstruction and `./output/mano` contains the fitted mano. You can use meshlab to visualize them. Please load the object and hand together for visualization.

The model in `./pretrained_model` is trained only on the [ObMan](https://hassony2.github.io/obman) dataset.
We include two sample objects from the [YCB](https://rse-lab.cs.washington.edu/projects/posecnn/) dataset in `./input`. These sample meshes are for demo purpose only, please downlond the full dataset from the website. Note that these objects are also used in the [HO3D](https://github.com/shreyashampali/ho3d) dataset.

New objects can be given to the model by providing the path to the meshes (`./input`) and the list of object (`input.json`). The object needs to be reachable when a hand wrist is at the origin and should not be in `[-x,-z]` quadrant (see example meshes).


