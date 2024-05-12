// modified from
// #include <libvis/rgbd_video_io_tum_dataset.h>

#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include <libvis/logging.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>

#include <opencv2/core/persistence.hpp>

namespace vis {

template <typename PoseScalar>
bool InterpolatePose(
    double timestamp,
    const vector<double>& pose_timestamps,
    const vector<Sophus::SE3<PoseScalar>>& poses,
    Sophus::SE3<PoseScalar>* pose,
    double max_interpolation_time_extent = numeric_limits<double>::infinity()) {
  CHECK_EQ(pose_timestamps.size(), poses.size());
  CHECK_GE(pose_timestamps.size(), 2u);
  
  if (timestamp <= pose_timestamps[0]) {
    *pose = poses[0];
    return true;
  } else if (timestamp >= pose_timestamps.back()) {
    *pose = poses.back();
    return true;
  }
  
  // TODO: Binary search should be faster (or with given starting point if having monotonically increasing query points as is the case below).
  for (usize i = 0; i < pose_timestamps.size() - 1; ++ i) {
    if (timestamp >= pose_timestamps[i] && timestamp <= pose_timestamps[i + 1]) {
      if ((timestamp - pose_timestamps[i]) > max_interpolation_time_extent ||
          (pose_timestamps[i + 1] - timestamp) > max_interpolation_time_extent) {
        return false;
      }
      
      double factor = (timestamp - pose_timestamps[i]) / (pose_timestamps[i + 1] - pose_timestamps[i]);
      
      const Sophus::SE3<PoseScalar>& pose_a = poses[i];
      const Sophus::SE3<PoseScalar>& pose_b = poses[i + 1];
      
      *pose = Sophus::SE3<PoseScalar>(
          pose_a.unit_quaternion().slerp(factor, pose_b.unit_quaternion()),
          pose_a.translation() + factor * (pose_b.translation() - pose_a.translation()));
      return true;
    }
  }
  
  return false;
}

template <typename T>
bool ReadTUMRGBDTrajectory(
    const char* path,
    vector<double>* pose_timestamps,
    vector<Sophus::SE3<T>>* poses_global_T_frame) {
  ifstream trajectory_file(path);
  if (!trajectory_file) {
    LOG(ERROR) << "Could not open trajectory file: " << path;
    return false;
  }
  string line;
  getline(trajectory_file, line);
  while (! line.empty()) {
    char time_string[128];
    Vector3d cam_translation;
    Quaterniond cam_rotation;
    
    if (line[0] == '#') {
      getline(trajectory_file, line);
      continue;
    }
    if (sscanf(line.c_str(), "%s %lf %lf %lf %lf %lf %lf %lf",
        time_string,
        &cam_translation[0],
        &cam_translation[1],
        &cam_translation[2],
        &cam_rotation.x(),
        &cam_rotation.y(),
        &cam_rotation.z(),
        &cam_rotation.w()) != 8) {
      LOG(ERROR) << "Cannot read poses! Line:";
      LOG(ERROR) << line;
      return false;
    }
    
    Sophus::SE3<T> global_T_frame = Sophus::SE3<T>(cam_rotation.cast<T>(),
                                                   cam_translation.cast<T>());
    
    pose_timestamps->push_back(atof(time_string));
    poses_global_T_frame->push_back(global_T_frame);
    
    getline(trajectory_file, line);
  }
  return true;
}

// Reads a variant of the TUM RGB-D dataset format. Returns true if the data
// was successfully read. Compared to the raw RGB-D datasets (as tgz archives),
// the calibration has to be added in a file calibration.txt, given as
// fx fy cx cy in a single line, and the associate.py tool from the benchmark
// website must be run as follows:
// python associate.py rgb.txt depth.txt > associated.txt
// The trajectory filename can be left empty to not load a trajectory.
template<typename ColorT, typename DepthT>
bool ReadTUMRGBDDatasetAssociatedAndCalibrated(
    const char* dataset_folder_path,
    const char* trajectory_filename,
    RGBDVideo<ColorT, DepthT>* rgbd_video,
    bool is_quaternion_to_be_inverted, 
    double max_interpolation_time_extent = numeric_limits<double>::infinity()) {
  rgbd_video->color_frames_mutable()->clear();
  rgbd_video->depth_frames_mutable()->clear();
  
  // Calibration file: intrinsic camera parameters
  string calibration_path = string(dataset_folder_path) + "/calibration.txt";
  ifstream calibration_file(calibration_path.c_str());
  if (!calibration_file) {
    LOG(ERROR) << "Could not open calibration file: " << calibration_path;
    return false;
  }
  string line_txt;
  getline(calibration_file, line_txt);
  double fx, fy, cx, cy;
  if (sscanf(line_txt.c_str(), "%lf %lf %lf %lf",
      &fx, &fy, &cx, & cy) != 4) {
    LOG(ERROR) << "Cannot read calibration!";
    return false;
  }
  // std::cout << "w/h: " << width << "," << height << std::endl;  // w/h: 640,480
  u32 width = 640;
  u32 height = 480;
  // Set Camera Intrinsic Parameters
  float camera_parameters[4];
  // camera_parameters[0] = argparser.data.camera_fx;
  // camera_parameters[1] = argparser.data.camera_fy;
  // camera_parameters[2] = argparser.data.camera_cx; // cx + 0.5;
  // camera_parameters[3] = argparser.data.camera_cy; // cy + 0.5;
  camera_parameters[0] = fx;
  camera_parameters[1] = fy;
  camera_parameters[2] = cx + 0.5;
  camera_parameters[3] = cy + 0.5;
  rgbd_video->color_camera_mutable()->reset(
      new PinholeCamera4f(width, height, camera_parameters));
  rgbd_video->depth_camera_mutable()->reset(
      new PinholeCamera4f(width, height, camera_parameters));
      
  vector<double> pose_timestamps;
  vector<SE3f> poses_global_T_frame;
  
  if (trajectory_filename != nullptr) {
    string trajectory_path = string(dataset_folder_path) + "/" + trajectory_filename;
    if (!ReadTUMRGBDTrajectory(trajectory_path.c_str(), &pose_timestamps, &poses_global_T_frame)) {
      return false;
    }
  }
  
  // u32 width = 0;
  // u32 height = 0;
  width = 0;
  height = 0;
  
  string associated_filename = string(dataset_folder_path) + "/associated.txt";
  ifstream associated_file(associated_filename.c_str());
  if (!associated_file) {
    LOG(ERROR) << "Could not open associated file: " << associated_filename;
    return false;
  }
  
  std::string line;
  usize frame_index = 0; 
  usize frame_index_pre = 0; 
  while (!associated_file.eof() && !associated_file.bad()) {
    std::getline(associated_file, line);
    if (line.size() == 0 || line[0] == '#') {
      continue;
    }
    
    char rgb_time_string[128];
    char rgb_filename[128];
    char depth_time_string[128];
    char depth_filename[128];
    
    if (sscanf(line.c_str(), "%s %s %s %s",
        rgb_time_string, rgb_filename, depth_time_string, depth_filename) != 4) {
      LOG(ERROR) << "Cannot read association line!";
      return false;
    }
    
    frame_index_pre++; ////////////////

    SE3f rgb_global_T_frame;
    double rgb_timestamp = atof(rgb_time_string);
    if (!poses_global_T_frame.empty()) {
      if (!InterpolatePose(rgb_timestamp, pose_timestamps, poses_global_T_frame, &rgb_global_T_frame, max_interpolation_time_extent)) {
        continue;
      }
    }
    
    SE3f depth_global_T_frame;
    double depth_timestamp = atof(depth_time_string);
    if (!poses_global_T_frame.empty()) {
      if (!InterpolatePose(depth_timestamp, pose_timestamps, poses_global_T_frame, &depth_global_T_frame, max_interpolation_time_extent)) {
        continue;
      }
    }
    
    string color_filepath =
        string(dataset_folder_path) + "/" + rgb_filename;
    ImageFramePtr<ColorT, SE3f> image_frame(new ImageFrame<ColorT, SE3f>(color_filepath, rgb_timestamp, rgb_time_string));
    image_frame->SetGlobalTFrame(rgb_global_T_frame);
    rgbd_video->color_frames_mutable()->push_back(image_frame);
    
    string depth_filepath =
        string(dataset_folder_path) + "/" + depth_filename;
    ImageFramePtr<DepthT, SE3f> depth_frame(new ImageFrame<DepthT, SE3f>(depth_filepath, depth_timestamp, depth_time_string));
    depth_frame->SetGlobalTFrame(depth_global_T_frame);
    rgbd_video->depth_frames_mutable()->push_back(depth_frame);
    
    if (width == 0) {
      // Get width and height by loading one image file.
      shared_ptr<Image<ColorT>> image_ptr =
          image_frame->GetImage();
      if (!image_ptr) {
        LOG(ERROR) << "Cannot load image to determine image dimensions.";
        return false;
      }
      width = image_ptr->width();
      height = image_ptr->height();
      image_frame->ClearImageAndDerivedData();
    }

    

    if (is_quaternion_to_be_inverted){
      SE3f global_T_frame = rgbd_video->color_frame_mutable(frame_index)->global_T_frame();
      global_T_frame.setQuaternion(global_T_frame.unit_quaternion().inverse());
      rgbd_video->color_frame_mutable(frame_index)->SetGlobalTFrame(global_T_frame);
      
      global_T_frame = rgbd_video->depth_frame_mutable(frame_index)->global_T_frame();
      global_T_frame.setQuaternion(global_T_frame.unit_quaternion().inverse());
      rgbd_video->depth_frame_mutable(frame_index)->SetGlobalTFrame(global_T_frame.inverse());
    }
    /////////////////////
    frame_index++; 
    std::cout << frame_index_pre << ", idx: " << frame_index << ", " << rgbd_video->frame_count() << std::endl; 

  } // end of reading associated_file
  
  return true;
}

template<typename ColorT, typename DepthT>
void initializeRGBDVideo(RGBDVideo<ColorT, DepthT>* rgbd_video,
                      float fx, float fy, float cx, float cy, 
                      int width, int height) {
  rgbd_video->color_frames_mutable()->clear();
  rgbd_video->depth_frames_mutable()->clear();
  
  // Set Camera Intrinsic Parameters
  float camera_parameters[4];
  camera_parameters[0] = fx;
  camera_parameters[1] = fy;
  camera_parameters[2] = cx; // cx + 0.5;
  camera_parameters[3] = cy; // cy + 0.5;
  rgbd_video->color_camera_mutable()->reset(
      new PinholeCamera4f(u32(width), u32(height), camera_parameters));
  rgbd_video->depth_camera_mutable()->reset(
      new PinholeCamera4f(u32(width), u32(height), camera_parameters));
}

// Reads a variant of the RealSense RGB-D dataset format. 
// Returns true if the data was successfully read. 
template<typename ColorT, typename DepthT>
void ReadSequentialRGBDDataset(
    std::string& dataset_folder_path,
    RGBDVideo<ColorT, DepthT>* rgbd_video,
    std::vector<double>& pose_timestamps, 
    std::vector<Sophus::SE3f> &poses_global_T_frame, 
    std::vector<std::string> vstrImageFilenamesRGB, 
    std::vector<std::string> vstrImageFilenamesD, 
    bool is_quaternion_to_be_inverted, 
    int end_frame_index) {

  usize frame_index = -1; 
  for(double ti : pose_timestamps) {
    
    frame_index++;
    
    Sophus::SE3f global_T_frame = poses_global_T_frame[frame_index]; 
    if (is_quaternion_to_be_inverted){
      global_T_frame.setQuaternion(global_T_frame.unit_quaternion().inverse());
    }

    std::string color_filepath = dataset_folder_path + "/" + vstrImageFilenamesRGB[frame_index];
    ImageFramePtr<ColorT, SE3f> image_frame(new ImageFrame<ColorT, SE3f>(color_filepath, ti));
    image_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video->color_frames_mutable()->push_back(image_frame);
    
    std::string depth_filepath = dataset_folder_path + "/" + vstrImageFilenamesD[frame_index];
    ImageFramePtr<DepthT, SE3f> depth_frame(new ImageFrame<DepthT, SE3f>(depth_filepath, ti));
    depth_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video->depth_frames_mutable()->push_back(depth_frame);

  } // end of reading associated_file
}

}
