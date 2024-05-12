#pragma once

#include <iostream> // std::cout
// #include <libvis/eigen.h> // Vec3f
#include "struct_utils.h"
#include <fstream>  // ifstream
#include <memory>
#include <spline_library/splines/uniform_cr_spline.h>

namespace vis {

class playbackKeyframe {
 public:

    playbackKeyframe(std::string filePath);

    void read(); 
    float getKeyframeIndex(usize frame_index); 
    cameraOrbit convertSpline2KeyframeTrajectory(usize frame_index); 
    bool isValid(); 

    std::string filePath_;

    std::ifstream keyframes_read_file;

    SplineParameter<unique_ptr<UniformCRSpline<FloatForSpline>>> spline;
    std::vector<float> spline_frame_indices;

    std::vector<FloatForSpline> offset_x_spline_points;
    std::vector<FloatForSpline> offset_y_spline_points;
    std::vector<FloatForSpline> offset_z_spline_points;
    std::vector<FloatForSpline> radius_spline_points;
    std::vector<FloatForSpline> theta_spline_points;
    std::vector<FloatForSpline> phi_spline_points;
    std::vector<FloatForSpline> max_depth_spline_points;
};

}