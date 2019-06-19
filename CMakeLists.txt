CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(OpenCV_DIR /usr/local/share/OpenCV)
FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(delaunayEffect)

# Inform the paths of the libraries depending on your installation
INCLUDE_DIRECTORIES(/usr/local/include)
LINK_DIRECTORIES(/usr/local/lib)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

ADD_EXECUTABLE(filterViewer main.cpp)
TARGET_LINK_LIBRARIES(filterViewer ${OpenCV_LIBS} sfml-audio sfml-system aquila)
