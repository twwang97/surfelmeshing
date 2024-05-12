#include "playbackKeyframe.h"

namespace vis {

playbackKeyframe::playbackKeyframe(std::string filePath){
    filePath_ = filePath; 
    if (!filePath_.empty()) {
        keyframes_read_file.open(filePath_, std::ios::in);
        if (!keyframes_read_file) {
            // LOG(FATAL) << "Cannot open " << filePath_;
            std::cerr << "Cannot open " << filePath_ << std::endl; 
        }
    }
}

void playbackKeyframe::read(){

    if (!filePath_.empty()) {
        while (!keyframes_read_file.eof() && !keyframes_read_file.bad()) {
            std::string line;
            std::getline(keyframes_read_file, line);
            if (line.size() == 0 || line[0] == '#') {
                continue;
            }

            std::istringstream line_stream(line);
            std::string word;
            usize frame_index;
            cameraOrbit camera_free_orbit; 
            line_stream >> word >> frame_index >> camera_free_orbit.offset.x() >>
                            camera_free_orbit.offset.y() >>
                            camera_free_orbit.offset.z() >>
                            camera_free_orbit.radius >> camera_free_orbit.theta >>
                            camera_free_orbit.phi >> camera_free_orbit.max_depth;
            // CHECK_EQ(word, "keyframe");
            spline_frame_indices.push_back(frame_index);
            offset_x_spline_points.push_back(FloatForSpline(camera_free_orbit.offset.x()));
            offset_y_spline_points.push_back(FloatForSpline(camera_free_orbit.offset.y()));
            offset_z_spline_points.push_back(FloatForSpline(camera_free_orbit.offset.z()));

            radius_spline_points.push_back(FloatForSpline(camera_free_orbit.radius));
            theta_spline_points.push_back(FloatForSpline(camera_free_orbit.theta));
            phi_spline_points.push_back(FloatForSpline(camera_free_orbit.phi));
            max_depth_spline_points.push_back(FloatForSpline(camera_free_orbit.max_depth));
        }

        keyframes_read_file.close();

        spline.offset_x.reset(new UniformCRSpline<FloatForSpline>(offset_x_spline_points));
        spline.offset_y.reset(new UniformCRSpline<FloatForSpline>(offset_y_spline_points));
        spline.offset_z.reset(new UniformCRSpline<FloatForSpline>(offset_z_spline_points));
        spline.radius.reset(new UniformCRSpline<FloatForSpline>(radius_spline_points));
        spline.theta.reset(new UniformCRSpline<FloatForSpline>(theta_spline_points));
        spline.phi.reset(new UniformCRSpline<FloatForSpline>(phi_spline_points));
        spline.max_depth.reset(new UniformCRSpline<FloatForSpline>(max_depth_spline_points));
    }
}

float playbackKeyframe::getKeyframeIndex(usize frame_index){
    // Determine camera pose from spline-based keyframe playback.
    usize first_keyframe_index = spline_frame_indices.size() - 1;
    for (usize i = 1; i < spline_frame_indices.size(); ++ i) {
        if (spline_frame_indices[i] >= frame_index) {
            first_keyframe_index = i - 1;
            break;
        }
    }
    usize prev_frame_index = spline_frame_indices[first_keyframe_index];
    usize next_frame_index = spline_frame_indices[first_keyframe_index + 1];
    float t = -1 + first_keyframe_index + (frame_index - prev_frame_index) * 1.0f / (next_frame_index - prev_frame_index);
    return t; 
}

cameraOrbit playbackKeyframe::convertSpline2KeyframeTrajectory(usize frame_index){
    cameraOrbit camera_free_orbit; 
    if (!isValid()) {
        camera_free_orbit.max_depth = -1; 
        return camera_free_orbit; 
    }
    float t = getKeyframeIndex(frame_index); 
    camera_free_orbit.offset.x() = spline.offset_x->getPosition(t);
    camera_free_orbit.offset.y() = spline.offset_y->getPosition(t);
    camera_free_orbit.offset.z() = spline.offset_z->getPosition(t);
    camera_free_orbit.radius = spline.radius->getPosition(t);
    camera_free_orbit.theta = spline.theta->getPosition(t);
    camera_free_orbit.phi = spline.phi->getPosition(t);
    camera_free_orbit.max_depth = spline.max_depth->getPosition(t);
    return camera_free_orbit; 
}

bool playbackKeyframe::isValid(){
    if (filePath_.empty()) {
        return false; 
    } else {
        return true; 
    }
}


}