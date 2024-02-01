<div align="center">
    <h1>UAVTR</h1>
    <h3>Localization of a GNSS denied UAV <br> relative to a Moving Local Reference Frame in an Offshore Environment</h3>
</div>

<p align="justify">
Project deals with the problem of localizing a GNSS denied Unmanned Aerial Vehicle (UAV) in an offshore environment, where the aim of the UAV is to track a moving surface vessel. Due to the homogeneity of an offshore environment, the UAV is unable to localize relative to static features. Therefore, a method for localizing the UAV relative to the surface vessel is developed. This poses challenges in that the surface vessel is moving both according to its own volition and due to external influences from the sea, which means that its local frame is non-inertial. The aim of this thesis is to localize relative to the surface vessel in spite of these challenges, in a manner that allows the drone to fly autonomously using the localization as feedback.
</p>

### Modules

- `src/detection`: Real-time visual vessel detection using combination of `YOLOv7` and `MOSSE` tracker. Could be used for arbitrary vessel tracking tasks.
- `src/estimation`: Sensor fusion module for estimating UAV pose relative to the vessel. Contains many things smushed together (because of time limitation did not have time to properly separate them out):
    - Kalman filter implementation
    - Camera spatial velocity measurement system based on Interaction matrix
    - ROS2 wrapper for Madwick orientation filter 

### Installing

Code depends on:
- ROS2 Humble
- Eigen3
- OpenCV
- A bunch of ROS2 packages (should inspect `CMakelists.txt` files for full list)
- Fusion (https://github.com/xioTechnologies/Fusion) (auto install on configure)
- ONNX Runtime (auto install on configure)

Configure & Build:
```bash
make configure
make build
```

### Data

If you'd like to get access to the data used for experiments, please create an issue. The data is not included in this repository due to its size. Additionally, model weights for maritime vessel detection are freely available to download on HuggingFace [here](https://huggingface.co/ernielov/yolov7-marine). Note that these are downloaded automatically when running configure step.

### Quality Statement

This is research quality code, not suitable for production or any real use case. The code is provided as-is with no guarantees or warranties.
