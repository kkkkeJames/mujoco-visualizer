This repository is part of an independent study conducted under the supervision of Dr. Wang, PhD.
This repository is a visualizer for the the open source physics engine [MuJoCo](https://github.com/deepmind/mujoco) (>= 3.2.4) from DeepMind. 

# Installing dependencies
All required Python packages can be installed via:

```bash
pip install -r requirements.txt
```

## Additional Notes on MuJoCo
This visualizer is based on the official python binding of Mujoco. Starting from the version 2.1.2, Mujoco officially supports native bindings of Python. 
You can install the Python bindings via:
```bash
pip install mujoco
```
Make sure your system supports OpenGL and GLFW, which are required for the visualizer. See the [Mujoco installing instrction](https://github.com/google-deepmind/mujoco/blob/main/README.md) if you run into any issues with installing Mujoco.

# Usage
To render in a window, refers to the example provided.
```
import mujoco
import mujoco_visualizer
viewer = mujoco_visualizer.Mujoco_Visualizer(mujoco.MjModel.from_xml_path('humanoid.xml'))
viewer.run()
```
To modify the xml model used, modify the path to the path of the input xml file. `humanoid.xml` is already provided in this repository.

## Update render settings
To update render settings, press button corresponding to the instruction of rendering window. 
If this visualizer has previously executed successfully, a file named `mjc_visualizer_settings.json` would be created in the same path as the renderer. It stores the render settings of the user, and modifying it also changes the settings of the visualizer.

## Screenshot
This visualizer includes a built-in screenshot and video recording system. Press "P" to capture a screenshot, and press "Q" to start/stop recording a video.
If this visualizer has previously executed successfully, two folders named `screenshots` and `videos` would be created in the same path as the renderer, which will store all screenshots and videos.

## GPU data output (under implementation)
Press "G" for enabling/disabling GPU data output. However, as this function currently only supports PC with Nvidia GPU and is still under implementation, it is not recommended to enable that feature.
