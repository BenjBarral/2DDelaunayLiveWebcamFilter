![Readme image](images-and-videos/DelaunayGitImage.png)

A program that creates an artistic image filter based on Delaunay triangulation. The application takes the webcam input and applies the filter, in real-time.
The filter distorts through "wave-like" effects, in synchronization with musical input.
The distortions can also be triggered manually via keys.

Implementation of a rhythm detection algorithm (signal processing, Fourier transform...) to detect the beats in any track, on-the-run.
The image filter is based on : contour detection, Delaunay triangulation, background subtraction, point selection...

Keys : 
- S : enable or disable "sound synchronization" mode. If the sound-sync is disabled, trigger the effects manually using the keys.
- Left/right/up/down arrows; ENTER; SHIFT : different distortion effects
- D : segment background and foreground triangulation
- space bar : quit the application

Put the path of the track you want to play in the path_track variable (MP3,WAV file).
Change the variables disp_height and disp_width for the size of the filter window.

Libraries required : 
-OpenCV : https://opencv.org/releases/
-Aquila + SFML (signal processing) : https://aquila-dsp.org/download/
Change the paths to the include and library paths in the CMakeLists.txt : LINL_DIRECTORIES and INCLUDE_DIRECTORIES.

