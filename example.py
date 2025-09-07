import mujoco
import mujoco_visualizer
# To modify the xml model used, modify the path to the path that stores the input xml file.
viewer = mujoco_visualizer.Mujoco_Visualizer(mujoco.MjModel.from_xml_path('humanoid.xml'))
viewer.run()