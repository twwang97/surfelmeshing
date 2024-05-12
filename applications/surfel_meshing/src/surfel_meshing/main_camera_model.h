#pragma once
#include <iostream>
#include <memory>
// #include <math.h>
#include <libvis/camera.h> // Camera

namespace vis {

class main_camera_model {
 public:
    main_camera_model(
                        const Camera& , 
                        int pyramid_level); 
    
    std::unique_ptr<Camera> scaled_camera = nullptr;
    PinholeCamera4f depth_camera; 
    float camera_4parameters[4] {0,0,0,0}; 
    int width, height; // for the depth camera
};

}