#pragma once

// #include <libvis/rgbd_video_io_tum_dataset.h> // ReadTUMRGBDDatasetAssociatedAndCalibrated

// initializeRGBDVideo and ReadSequentialRGBDDataset is modified from "surfel_meshing/rgbd_realsense_dataset.h"

#include "surfel_meshing/ArgumentParser.h"

namespace vis {

class RGBD_INPUT_CLASS {  
  public:     
  RGBD_INPUT_CLASS(std::string&, std::string&, bool); 
  RGBDVideo<Vec3u8, u16> init(ArgumentParser& argparser); 
  
  template<typename ColorT, typename DepthT>
  void Read1RGBDDataset(usize frame_i, 
    RGBDVideo<ColorT, DepthT>* rgbd_video); 
    
  std::vector<std::string> vstrImageFilenamesRGB; // path to RGB images
  std::vector<std::string> vstrImageFilenamesD; // path to depth images
  std::vector<double> vTimestamps; // path to timestamps of each image
  std::vector<Sophus::SE3f> vTrajectories; // trajectories
  std::string dataset_folder_path_; 
  bool is_quaternion_to_be_inverted_; 
  usize data_size_ = 0;
};


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

// Return only one set of RGB-D dataset
template<typename ColorT, typename DepthT>
void RGBD_INPUT_CLASS::Read1RGBDDataset(usize frame_i, 
    RGBDVideo<ColorT, DepthT>* rgbd_video) 
  {
    double ti = vTimestamps[frame_i];
    Sophus::SE3f global_T_frame = vTrajectories[frame_i];
    if(is_quaternion_to_be_inverted_){
      global_T_frame.setQuaternion(global_T_frame.unit_quaternion().inverse());
    }

    std::string color_filepath = dataset_folder_path_ + "/" + vstrImageFilenamesRGB[frame_i];
    ImageFramePtr<ColorT, SE3f> image_frame(new ImageFrame<ColorT, SE3f>(color_filepath, ti));
    image_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video->color_frames_mutable()->push_back(image_frame);
    
    std::string depth_filepath = dataset_folder_path_ + "/" + vstrImageFilenamesD[frame_i];
    ImageFramePtr<DepthT, SE3f> depth_frame(new ImageFrame<DepthT, SE3f>(depth_filepath, ti));
    depth_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video->depth_frames_mutable()->push_back(depth_frame);
}

// Return only one set of RGB-D dataset
template<typename ColorT, typename DepthT>
void Read1RGBDDataset(
    const std::string &dataset_folder_path, 
    RGBDVideo<ColorT, DepthT>* rgbd_video, 
    double &ti, 
    const Sophus::SE3f &global_T_frame, 
    const std::string &vstrImageFilenamesRGB_i, 
    const std::string &vstrImageFilenamesD_i) 
  {
    std::string color_filepath = dataset_folder_path + "/" + vstrImageFilenamesRGB_i;
    ImageFramePtr<ColorT, SE3f> image_frame(new ImageFrame<ColorT, SE3f>(color_filepath, ti));
    image_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video->color_frames_mutable()->push_back(image_frame);
    
    std::string depth_filepath = dataset_folder_path + "/" + vstrImageFilenamesD_i;
    ImageFramePtr<DepthT, SE3f> depth_frame(new ImageFrame<DepthT, SE3f>(depth_filepath, ti));
    depth_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video->depth_frames_mutable()->push_back(depth_frame);
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
    bool is_quaternion_to_be_inverted) {

  usize frame_index = -1; 
  for(double ti : pose_timestamps) {
    
    frame_index++;
    
    Sophus::SE3f global_T_frame = poses_global_T_frame[frame_index]; 
    if (is_quaternion_to_be_inverted){
      global_T_frame.setQuaternion(global_T_frame.unit_quaternion().inverse());
    }

    Read1RGBDDataset(dataset_folder_path, \
                      rgbd_video, \
                      ti, 
                      global_T_frame, 
                      vstrImageFilenamesRGB[frame_index], 
                      vstrImageFilenamesD[frame_index]); 

  } // end of reading associated_file
}

RGBD_INPUT_CLASS::RGBD_INPUT_CLASS(std::string& dataset_folder_path, std::string& trajectory_filename, bool is_quaternion_to_be_inverted){

  dataset_folder_path_ = dataset_folder_path; 
  is_quaternion_to_be_inverted_ = is_quaternion_to_be_inverted; 

  // Retrieve paths to images  
  std::string strAssociationFilename = dataset_folder_path + "/" + trajectory_filename; // associated txt file connecting RGB and depth files
  // int nImages = LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps); 
  int nImages = Load_ImageName_Trajectory(strAssociationFilename, 
                                      vstrImageFilenamesRGB, vstrImageFilenamesD, 
                                      vTimestamps, vTrajectories); 
  if(nImages < 1){
      LOG(FATAL) << "invalid loading of image sequence.";
      // return EXIT_FAILURE; 
  } else {
    std::cout << "data size:\n" << vTimestamps.size() << ", " 
              << vstrImageFilenamesRGB.size() << ", " << vstrImageFilenamesD.size() << std::endl; 
    if (vTimestamps.size() == vstrImageFilenamesRGB.size() && vTimestamps.size() == vstrImageFilenamesD.size())
      data_size_ = vTimestamps.size(); 
  }
}

RGBDVideo<Vec3u8, u16> RGBD_INPUT_CLASS::init(ArgumentParser& argparser){
  // Load dataset.
  RGBDVideo<Vec3u8, u16> rgbd_video;
  // if (!ReadTUMRGBDDatasetAssociatedAndCalibrated(argparser.data.dataset_folder_path.c_str(), argparser.data.trajectory_filename.c_str(), &rgbd_video, argparser.data.invert_quaternions, argparser.data.max_pose_interpolation_time_extent)) {
  //  LOG(FATAL) << "Could not read dataset.";
  //} else {
  //  CHECK_EQ(rgbd_video.depth_frames_mutable()->size(), rgbd_video.color_frames_mutable()->size());
  //  LOG(INFO) << "Read dataset with " << rgbd_video.frame_count() << " frames";
  //}

  initializeRGBDVideo(&rgbd_video, 
                      argparser.data.camera_fx, argparser.data.camera_fy, 
                      argparser.data.camera_cx, argparser.data.camera_cy, 
                      argparser.data.camera_width, argparser.data.camera_height); 
  
  ReadSequentialRGBDDataset(argparser.data.dataset_folder_path,  
                          &rgbd_video, 
                          vTimestamps, 
                          vTrajectories, 
                          vstrImageFilenamesRGB, 
                          vstrImageFilenamesD, 
                          argparser.data.invert_quaternions); 
  
  CHECK_EQ(rgbd_video.depth_frames_mutable()->size(), rgbd_video.color_frames_mutable()->size());
  LOG(INFO) << "Read dataset with " << rgbd_video.frame_count() << " frames";

  // If end_frame is non-zero, 
  // then remove all frames which would extend beyond
  // this length.
  if (argparser.data.end_frame > 0 &&
      rgbd_video.color_frames_mutable()->size() > static_cast<usize>(argparser.data.end_frame)) 
  {
    rgbd_video.color_frames_mutable()->resize(argparser.data.end_frame);
    rgbd_video.depth_frames_mutable()->resize(argparser.data.end_frame);
  }

  // Check that the RGB-D dataset uses the same intrinsics for color and depth.
  if (!AreCamerasEqual(*rgbd_video.color_camera(), *rgbd_video.depth_camera())) {
    LOG(FATAL) << "The color and depth camera of the RGB-D video must be equal.";
  }
  return rgbd_video; 
}


}
