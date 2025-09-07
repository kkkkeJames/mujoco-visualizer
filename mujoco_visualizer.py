# This code is largely based on basic.cc, the basic structure for mujoco visualization, and simulate.cc, an advanced visualizer with several functions
import mujoco
import glfw
import numpy
import json
import os
import cv2
import moviepy
import datetime
import gpu_data_monitor
from threading import Lock
from OpenGL.GL import *
from PIL import Image

class Mujoco_Visualizer():
    # Initialize the visualizer with model as input and maxgeom as optional input (default value is 10000) from user
    def __init__(self, model, maxgeom : int = 10000):
        # 
        self.lock = Lock()
        self.timer = 0
        # Get model and get data from model, get maxgeom from input
        self.model = model
        self.data = mujoco.MjData(model)
        self.maxgeom = maxgeom
        # Initialize variables
        # Flags for monitoring if mouse buttons are pressed
        self.leftbuttonpressed = False
        self.rightbuttonpressed = False
        self.leftbuttondoubleclick = False
        # Record if left button had already been pressed once
        self.leftbuttonhadpressed = False
        # Time for monitoring if double click applied
        self.leftbuttonpresstime = 0.0
        # Last x, y position of mouse
        self.lastmousexpos = 0
        self.lastmouseypos = 0
        # Variables for key bindings, rendering speed, and other stuff
        self.paused = False # Whether the simulation of robot is paused
        # Flags for mjtRndFlags and other flags in a dictionary
        self.settings = {
            "render_shadow": False, # Whether shadows are rendered
            "render_wireframe": False, # Whether the model is rendered in wireframe mode
            "render_reflection": False, # Whether reflections are rendered
            "render_skybox": False, # Whether skybox is rendered
            "output_gpu_data": False, # Whether output GPU data
            "rendering_multiplier": 1.0, # The rendering multiplier (final speed is this * 60 fps)
            "show_contact_forces": False, # Whether show contact forces
            "show_joints": False, # Whether show joints
            "show_constraints": False, # Whether display constraints
            "show_inertia": False, # Whether display inertia boxes
            "show_com": False # Whether display center of mass
        }
        # Init GLFW or raise error (if failed to init)
        if not glfw.init():
            raise Exception("Failed to initiate GLFW")
        # Get the width and height of the screen, and create a window with them
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        self.window = glfw.create_window(width, height, "Mujoco Visualizer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        glfw.maximize_window(self.window)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        # Initialize MjvOption, MjvPerturb, Camera, viewpoint, and get scene and context from model
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene(self.model, maxgeom)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        # TODO: Install text for overlay
        # Initialize overlay texts
        self.control_overlay = ["", ""]
        self.data_overlay = ["", ""]
        # Set the cursor position callback, scroll callback to the window with function defined
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)
        # Create a folder for storing images
        # Get the direction of this file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Create a path of screenshot folder
        self.screenshot_dir = os.path.join(self.base_dir, 'screenshots')
        # If there is not a folder in that path, create it
        os.makedirs(self.screenshot_dir, exist_ok=True)
        # Create a path of screenshot videos folder
        self.videos_dir = os.path.join(self.base_dir, 'videos')
        # If there is not a folder in that path, create it
        os.makedirs(self.videos_dir, exist_ok=True)
        # If a video is being recorded
        self.recording = False
        # If a recording is stopped, used for producing the final video
        self.stoppedrecording = False

        # Create a pixel buffer object for accelerating screenshot capturing
        # Get the width and height of window
        width, height = glfw.get_framebuffer_size(self.window)
        # Get the data size of rgb array for each pixel
        data_size = width * height * 3
        # Allocate a single PBO
        self.pbo = glGenBuffers(1)
        # Bind buffer, allocate space
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_PACK_BUFFER, data_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        # Array for storing pixel for screenshot image
        self.screenshot_pixel_array = numpy.array([])
        # Array for storing pixels for images that are appended to screenshot videos
        self.tempshot_pixel_array = numpy.array([])
        # List for storing pixels for screenshot videos
        self.video_frames = []
        self.nvmlinit = True # Whether the nivida gpu data monitor initialized successfully

    def save_render_settings(self):
        # Dump the render settings in the json file in the same directory as this file, and create such json if it doesn't exists
        with open(self.base_dir+"/mjc_visualizer_settings.json", "w", encoding="utf-8") as file:
            json.dump(self.settings, file, indent=4)

    def load_render_settings(self):
        # Try to load the settings json file, else ignore
        try:
            with open(self.base_dir+"/mjc_visualizer_settings.json", "r", encoding="utf-8") as file:
                setting = json.load(file)
                # Update self.settings by settings in the setting json file
                self.settings.update(setting)
                # Then call update render settings to truly update the setting dictionary to the visualizer
                self.update_render_settings()
        except json.JSONDecodeError:
            # No file exist (mainly because it is the first time to open the visualizer), return default
            print("Error finding file: no previous settings exist")
            # Then create a visualizer settings file set in default by calling save render settings
            self.save_render_settings()
        except FileNotFoundError:
            # No file exist (mainly because it is removed), return default
            print("Error finding file: no previous settings exist")
            # Then create a visualizer settings file set in default by calling save render settings
            self.save_render_settings()

    # Update flags to render settings, only called after key presses that change them or loading settings from saved files
    def update_render_settings(self):
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.settings["render_shadow"]
        self.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = self.settings["render_wireframe"]
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.settings["render_reflection"]
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = self.settings["render_skybox"]
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.settings["show_contact_forces"]
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.settings["show_contact_forces"]
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self.settings["show_joints"]
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = self.settings["show_constraints"]
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = self.settings["show_inertia"]
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = self.settings["show_com"]

    # Update control key instructions on topleft screen
    def update_control_overlay (self):
        self.control_overlay[0] += "Restart simulation\n"
        self.control_overlay[1] += "Backspace\n"
        if (self.paused):
            self.control_overlay[0] += "Continue simulation\n"
        else:
            self.control_overlay[0] += "Pause simulation\n"
        self.control_overlay[1] += "SPACE\n"
        self.control_overlay[0] += "Step forward simulation once\n"
        self.control_overlay[1] += "RIGHT\n"
        self.control_overlay[0] += "Speed up simulation\n"
        self.control_overlay[1] += "UP\n"
        self.control_overlay[0] += "Slow down simulation\n"
        self.control_overlay[1] += "DOWN\n"
        if self.settings["render_shadow"]:
            self.control_overlay[0] += "Disable shadow rendering \n"
        else:
            self.control_overlay[0] += "Enable shadow rendering (high GPU usage)\n"
        self.control_overlay[1] += "O\n"
        if self.settings["render_wireframe"]:
            self.control_overlay[0] += "Disable wireframe mode\n"
        else:
            self.control_overlay[0] += "Enable wireframe mode\n"
        self.control_overlay[1] += "W\n"
        if self.settings["render_reflection"]:
            self.control_overlay[0] += "Disable reflection rendering \n"
        else:
            self.control_overlay[0] += "Enable reflection rendering (high GPU usage)\n"
        self.control_overlay[1] += "R\n"
        if self.settings["render_skybox"]:
            self.control_overlay[0] += "Disable skybox rendering \n"
        else:
            self.control_overlay[0] += "Enable skybox rendering (high GPU usage)\n"
        self.control_overlay[1] += "S\n"
        self.control_overlay[0] += "Display contact forces \n"
        self.control_overlay[1] += "F\n"
        self.control_overlay[0] += "Display joints \n"
        self.control_overlay[1] += "J\n"
        self.control_overlay[0] += "Display constraints \n"
        self.control_overlay[1] += "C\n"
        self.control_overlay[0] += "Display inertia boxes \n"
        self.control_overlay[1] += "I\n"
        self.control_overlay[0] += "Display center of mass \n"
        self.control_overlay[1] += "M\n"
        self.control_overlay[0] += "Print GPU data \n"
        self.control_overlay[1] += "G\n"
        #self.control_overlay[0] += "t \n"
        #self.control_overlay[1] += f"{self.timer}\n"

    # Update data overlay on topright screen
    def update_data_overlay (self, gpu_index="", gpu_name="", gpu_utilization="", memory_used="", memory_total="", temperature=""):   
        # Update graphics overlay
        if self.settings["output_gpu_data"]:
            self.data_overlay[0] += "GPU\n"
            self.data_overlay[1] += gpu_index + "(" + gpu_name + ")\n"
            self.data_overlay[0] += "GPU utilization\n"
            self.data_overlay[1] += gpu_utilization + "%\n"
            self.data_overlay[0] += "Memory used\n"
            self.data_overlay[1] += memory_used + " MB\n"
            self.data_overlay[0] += "Memory total\n"
            self.data_overlay[1] += memory_total + " MB\n"
            self.data_overlay[0] += "GPU temperature\n"
            self.data_overlay[1] += temperature + " C\n"
        # Update other data overlay
        self.data_overlay[0] += "Rendering speed\n"
        self.data_overlay[1] += f"{round(self.settings["rendering_multiplier"], 1)}\n"

    # Monitor the press of keyboard and save them as default settings, called by glfw functions, never used directly
    def keyboard (self, window, key, scancode, act, mods):
        #with self.lock:
        # Press Backspace for reset data and setting forward (which eventually means the home position of the robot)
        if (act == glfw.PRESS and key == glfw.KEY_BACKSPACE):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        # Press O for enabling/disabling shadow rendering
        if (act == glfw.PRESS and key == glfw.KEY_O):
            self.settings["render_shadow"] = not self.settings["render_shadow"]
            self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.settings["render_shadow"]
        # Press W for enabling/disabling the wireframe rendering mode
        if (act == glfw.PRESS and key == glfw.KEY_W):
            self.settings["render_wireframe"] = not self.settings["render_wireframe"]
            self.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = self.settings["render_wireframe"]
        # Press R for enabling/disabling reflection rendering
        if (act == glfw.PRESS and key == glfw.KEY_R):
            self.settings["render_reflection"] = not self.settings["render_reflection"]
            self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.settings["render_reflection"]
        # Press S for enabling/disabling skybox rendering
        if (act == glfw.PRESS and key == glfw.KEY_S):
            self.settings["render_skybox"] = not self.settings["render_skybox"]
            self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = self.settings["render_skybox"]
        # Press G for enabling/disabling GPU data output
        if (act == glfw.PRESS and key == glfw.KEY_G):
            self.settings["output_gpu_data"] = not self.settings["output_gpu_data"]
        # Press space to pause/restart
        if (act == glfw.PRESS and key == glfw.KEY_SPACE):
            self.paused = not self.paused
        # Press right arrow to step forward once
        if (act == glfw.PRESS and key == glfw.KEY_RIGHT and self.paused == True):
            mujoco.mj_step(self.model, self.data)
        # Press up/down arrow to speed up/slow down rendering speed by 0.1 times (0.1 to 4)
        if (act == glfw.PRESS and key == glfw.KEY_UP):
            self.settings["rendering_multiplier"] += 0.1
        if (act == glfw.PRESS and key == glfw.KEY_DOWN and self.settings["rendering_multiplier"] > 0.2):
            self.settings["rendering_multiplier"] -= 0.1
        # Display contact forces
        if (act == glfw.PRESS and key == glfw.KEY_F):
            self.settings["show_contact_forces"] = not self.settings["show_contact_forces"]
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.settings["show_contact_forces"]
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.settings["show_contact_forces"]
        # Display only joints
        if (act == glfw.PRESS and key == glfw.KEY_J):
            self.settings["show_joints"] = not self.settings["show_joints"]
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self.settings["show_joints"]
        # Display constraints
        if (act == glfw.PRESS and key == glfw.KEY_C):
            self.settings["show_constraints"] = not self.settings["show_constraints"]
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = self.settings["show_constraints"]
        # Display inertia boxes
        if (act == glfw.PRESS and key == glfw.KEY_I):
            self.settings["show_inertia"] = not self.settings["show_inertia"]
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = self.settings["show_inertia"]
        # Display center of mass
        if (act == glfw.PRESS and key == glfw.KEY_M):
            self.settings["show_com"] = not self.settings["show_com"]
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = self.settings["show_com"]
        # Capture screenshot (while not recording video, they are using same pixel buffer object)
        if (act == glfw.PRESS and key == glfw.KEY_P and not self.recording):
            self.capture_screenshot()
        # Start/stop recording
        if (act == glfw.PRESS and key == glfw.KEY_Q):
            self.recording = not self.recording
            if not self.recording:
                self.stoppedrecording = True
        # After key inputs, save render settings into the json file
        self.save_render_settings()
        # Then call update render settings to truly update the setting dictionary to the visualizer
        self.update_render_settings()

    # Monitor the press of mouse buttons, called by glfw functions, never used directly
    def mouse_button (self, window, button, act, mods):
        self.leftbuttonpressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self.rightbuttonpressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS
        tempx, tempy = glfw.get_cursor_pos(window)
        self.lastmousexpos = int(tempx)
        self.lastmouseypos = int(tempy)

        # TODO: handle double click to select one of the body so it is possible to get Jacobians
        # If left mouse button is pressed
        if self.leftbuttonpressed:
            # If left mouse button has not been presse d before
            if not self.leftbuttonhadpressed:
                self.leftbuttonhadpressed = True
            # If left button had pressed...
            else:
                # ...and the time before last press is less than 0.2 sec:
                if glfw.get_time() - self.leftbuttonpresstime < 0.2:
                    self.leftbuttondoubleclick = True

        # TODO: If left mouse button has been double-clicked, find corresponding body
        if self.leftbuttondoubleclick:
            selpnt = numpy.zeros((3, 1), numpy.float64)
            selgeom = numpy.zeros((1, 1), numpy.int32)
            selflex = numpy.zeros((1, 1), numpy.int32)
            selskin = numpy.zeros((1, 1), numpy.int32)
            selbody = mujoco.mjv_select(self.model, 
                    self.data, 
                    self.opt, 
                    self.viewport.width/self.viewport.height, 
                    (tempx - self.viewport.left)/self.viewport.width,
                    (tempy - self.viewport.bottom)/self.viewport.height,
                    self.scene,
                    selpnt,
                    selgeom,
                    selflex,
                    selskin)
            jacp = jacr = numpy.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, selbody)
        
        # After double click, set it to false
        self.leftbuttondoubleclick = False
        # Get left button press time
        self.leftbuttonpresstime = glfw.get_time()

    # Monitor the move of cursor for cursor pos callback, called by glfw functions, never used directly
    def mouse_move (self, window, mousexpos, mouseypos):
        # If there is not even a mouse button pressed, just return
        if (not self.leftbuttonpressed and not self.rightbuttonpressed):
            return
        # Get the mouse displacement in one run, and make the new lastmousepos the recent pos
        dx = mousexpos - self.lastmousexpos
        dy = mouseypos - self.lastmouseypos
        self.lastmousexpos = mousexpos
        self.lastmouseypos = mouseypos
        # Get window size
        tempwidth, tempheight = glfw.get_framebuffer_size(window)

        # Get shift key state
        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        # Determine action based on mouse button
        action = None
        if self.rightbuttonpressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.leftbuttonpressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        # move camera
        with self.lock:
            mujoco.mjv_moveCamera(self.model, action, dx/tempheight, dy/tempheight, self.scene, self.cam)
        
    # Monitor the mouse scroll and transfer it to the move of camera, called by glfw functions, never used directly
    def scroll(self, window, x_offset, y_offset):
        with self.lock:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y_offset, self.scene, self.cam)

    # Capture screenshot after key binded is presed, stablized but inefficient, return to this if there are bugs but you want to get screenshots
    def capture_screenshot_stable(self):
        # Get the width and height of window
        width, height = glfw.get_framebuffer_size(self.window)
        # Variable rgb creates an empty array that records the rgb of each pixel of self.image, each rgb value is an unsigned 8 bit num (0-255)
        rgb = numpy.zeros((height, width, 3), dtype=numpy.uint8)
        # Read the screen buffer pixels to this rgb
        mujoco.mjr_readPixels(rgb, None, self.viewport, self.context)
        rgb = numpy.flipud(rgb)
        image = Image.fromarray(rgb)
        # To avoid repetition in screenshot names, use screenshot_system date_system time_microsecond
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        image.save(self.screenshot_dir+'/screenshot_'+date+'.png')

    # Capture screenshot after key binded is presed, PBO implemented for improving efficiency
    def capture_screenshot(self):
        # Get the width and height of window
        width, height = glfw.get_framebuffer_size(self.window)
        # Read the screen buffer to PBO created
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo)
        # Read the screen buffer pixels to this rgb and map it to screen, getting its address
        glReadPixels(self.viewport.left, self.viewport.bottom, self.viewport.width, self.viewport.height, GL_RGB, GL_UNSIGNED_BYTE, 0)
        pixel_address = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        # After getting the address, find its array and unmap PBO, delete it to free memory
        if pixel_address:
            self.screenshot_pixel_array = numpy.frombuffer(ctypes.string_at(pixel_address, width * height * 3), dtype=numpy.uint8)
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        # Transform the pixel array to image and save it
        image = cv2.cvtColor(self.screenshot_pixel_array.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 0)
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        cv2.imwrite(self.screenshot_dir+'/screenshot_'+date+'.png', image)

    # Capture screenshots for moviepy, so they are not saved as files
    def capture_video_screenshot(self):
        # Get the width and height of window
        width, height = glfw.get_framebuffer_size(self.window)
        # Read the screen buffer to PBO created
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo)
        # Read the screen buffer pixels to this rgb and map it to screen, getting its address
        glReadPixels(self.viewport.left, self.viewport.bottom, self.viewport.width, self.viewport.height, GL_RGB, GL_UNSIGNED_BYTE, 0)
        pixel_address = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        # After getting the address, find its array and unmap PBO, delete it to free memory
        if pixel_address:
            self.tempshot_pixel_array = numpy.frombuffer(ctypes.string_at(pixel_address, width * height * 3), dtype=numpy.uint8)
            self.tempshot_pixel_array = self.tempshot_pixel_array.reshape(height, width, 3)
            self.tempshot_pixel_array = numpy.flipud(self.tempshot_pixel_array)
            self.video_frames.append(self.tempshot_pixel_array.copy())
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

    # Generate screenshot videos through frames list
    def generate_video(self):
        video = moviepy.ImageSequenceClip(self.video_frames, fps=30)
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        video.write_videofile(self.videos_dir+'/video_'+date+".mp4", codec="libx264")
        self.video_frames = []
        self.stoppedrecording = False

    # The main function that runs the loop of visualization
    def run(self):
        # Load render settings from json files
        self.load_render_settings()
        # Create a GPU data monitor
        monitor = gpu_data_monitor.NVMLClient()
        if not monitor.nvmlflag:
            print("Failed to output GPU data")
            self.nvmlinit = False
        # Run the main loop
        while not glfw.window_should_close(self.window):
            with self.lock:
                self.timer = glfw.get_time()
                # Update overlay
                self.update_control_overlay()
                # Get current time and set the target frame
                simstart = self.data.time
                targetframe = self.settings["rendering_multiplier"] * 1.0 / 60.0
                # If not paused, step forward in 60 fps to match real time
                if not self.paused:
                    while (self.data.time - simstart < targetframe):
                        mujoco.mj_step(self.model, self.data)
                # Get the frame buffer of the window to get appropriate viewpoint
                tempwidth, tempheight = glfw.get_framebuffer_size(self.window)
                self.viewport.width, self.viewport.height = tempwidth, tempheight
                # Update the model to the scene
                mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
                mujoco.mjr_render(self.viewport, self.scene, self.context)
                # Overlay control
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, self.viewport, self.control_overlay[0], self.control_overlay[1], self.context)
                # If output GPU data
                if self.settings["output_gpu_data"] and self.nvmlinit:
                    # Get GPU data
                    gpu_data = monitor.list_gpus()
                    # Print GPU data
                    if gpu_data:
                        for data in gpu_data:
                            self.update_data_overlay(f"{data['gpu_index']}", f"{data['name']}", f"{data['util_gpu']}", f"{data['mem_used']}", f"{data['mem_total']}", f"{data['temperature']}")
                            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT, self.viewport, self.data_overlay[0], self.data_overlay[1], self.context)
                else:
                    self.update_data_overlay()
                    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT, self.viewport, self.data_overlay[0], self.data_overlay[1], self.context)
                # Swap OpenGL buffers
                glfw.swap_buffers(self.window)
                # If recording, keep saving screenshots
                if self.recording:
                    self.capture_video_screenshot()
                if self.stoppedrecording:
                    self.generate_video()
            # Get pending GUI events
            glfw.poll_events()
            # Clear overlay after rendering them
            self.control_overlay = ["", ""]
            self.data_overlay = ["", ""]

        # After the main loop ends, close GLFW and free resources
        glDeleteBuffers(1, [self.pbo])
        glfw.terminate()