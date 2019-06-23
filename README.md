# A live webcam filter, that animates in sync with musical inputs, using OpenCV

![](images-and-videos/giphy.gif)

A program that creates a dynamic image filter based on Delaunay triangulation. The application takes as input video feed from the webcam and applies the filter to it, in real-time.
The filter distorts through "wave-like" effects, in synchronization with musical input. See the [video example](images-and-videos/Video.mp4).
The distortions can also be triggered manually via keys.

The image filter is based on : contour detection, Delaunay triangulation, background subtraction, point selection...
I implemented a rhythm detection algorithm (signal processing, Fourier transform...) to detect the beats in any track, on-the-run.

## Interacting with the program
Use the following keys to interact with the different visual effects :
- S : enable or disable "sound synchronization" mode. If the sound-sync is disabled, trigger the effects manually using the keys.
- Left/right/up/down arrows; ENTER; SHIFT : different distortion effects
- D : segment background and foreground
- space bar : quit the application

## Usage
Run the ```filiterViewer``` target, and the UI will show.
Put the path of the track you want to play in the ```path_track``` variable (MP3,WAV file).
### Libraries required : 
-[OpenCV](https://opencv.org/releases/)
-[Aquila + SFML]( https://aquila-dsp.org/download/) (signal processing)
Change the paths to the include and library paths in the CMakeLists.txt : LINK_DIRECTORIES and INCLUDE_DIRECTORIES.
