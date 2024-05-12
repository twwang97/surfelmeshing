#include "mainViewers.h"
#include "struct_utils.h"

namespace vis {

mainViewers_::mainViewers_(bool is_input_viewed, bool is_depth_preprocessing_viewed, 
                            bool is_playback_valid, bool is_camera_followed, 
                            int width, int height, float scaling, 
                            int mesh_window_width, int mesh_window_height, 
                            bool render_new_surfels_as_splats,
                            float splat_half_extent_in_pixels,
                            bool triangle_normal_shading,
                            bool render_camera_frustum){
                            // std::shared_ptr<SurfelMeshingRenderWindow> render_window_01){
  
  is_input_viewed_ = is_input_viewed; 
  scaling_ = scaling; 
  is_depth_preprocessing_viewed_ = is_depth_preprocessing_viewed; 
  width_ = width;
  height_ = height; 

  if (is_playback_valid)
    camera_pose_mode = PlaybackCameraPose; 
  else if (is_camera_followed)
    camera_pose_mode = FollowCameraPose; 
  else
    camera_pose_mode = OtherCameraPose; 

  raw_rgb_display = std::shared_ptr<ImageDisplay>(new ImageDisplay());
  raw_depth_display = std::shared_ptr<ImageDisplay>(new ImageDisplay());
  filtered_depth_display = std::shared_ptr<ImageDisplay>(new ImageDisplay());
  downscaled_depth_display = std::shared_ptr<ImageDisplay>(new ImageDisplay());  

  // Create render window.
  render_window = std::shared_ptr<SurfelMeshingRenderWindow>(
      new SurfelMeshingRenderWindow(render_new_surfels_as_splats,
                                    splat_half_extent_in_pixels,
                                    triangle_normal_shading,
                                    render_camera_frustum));  

  generic_render_window = std::shared_ptr<RenderWindow>(
      RenderWindow::CreateWindow(WindowDisplayMap[WINDOW_OUTPUT_MESH], 
                      mesh_window_width, mesh_window_height, 
                      RenderWindow::API::kOpenGL, 
                      render_window));
}

void mainViewers_::initialize_cuda_opengl(usize max_point_count,
                                cudaGraphicsResource_t& vertex_buffer_resource,
                                OpenGLContext& opengl_context,
                                std::unique_ptr<Camera>& cameraD,
                                bool debug_neighbor_rendering,
                                bool debug_normal_rendering,
                                cudaGraphicsResource_t& neighbor_index_buffer_resource,
                                cudaGraphicsResource_t& normal_vertex_buffer_resource){

  render_window->InitializeForCUDAInterop(
        max_point_count,
        &vertex_buffer_resource,
        &opengl_context,
        *cameraD,
        debug_neighbor_rendering,
        debug_normal_rendering,
        &neighbor_index_buffer_resource,
        &normal_vertex_buffer_resource);
  OpenGLContext no_opengl_context;
  SwitchOpenGLContext(opengl_context, &no_opengl_context);
}
void mainViewers_::setMeshFirstWindowDirection(
                          const Vec3f& frame_position, 
                          const Vec3f& global_position){

  // Set the up direction of the first frame as the global up direction.

  render_window->SetUpDirection(frame_position);
  // Vec3f frame_position_one(0, 0, 1); 
  // Vec3f frame_position_one(-1, 0, 0); 
  // render_window->SetUpDirection(frame_position_one);

  render_window->CenterViewOn(global_position);
}

void mainViewers_::showInputRGBD(ImageFramePtr<Vec3u8, SE3f>& input_color_frame, 
                                  ImageFramePtr<u16, SE3f>& input_depth_frame){
  if (is_input_viewed_){
    raw_rgb_display->Update(*input_color_frame->GetImage(), "inputRGB");
    // raw_depth_display->Update(*input_depth_frame->GetImage(), "inputDepth", static_cast<u16>(0), static_cast<u16>(scaling_));
  }

}

// DEBUG: Preprocessing depth images
void mainViewers_::showPreprocessedDepthImage(Image<u16>& filtered_depth, int depth_type){
  if (is_depth_preprocessing_viewed_){
    std::string window_display_name = WindowDisplayMap[depth_type]; 
    filtered_depth_display->Update(filtered_depth, window_display_name,
                                     static_cast<u16>(0), static_cast<u16>(scaling_));
  }
}

// DEBUG: Preprocessing depth images
void mainViewers_::download_showPreprocessedDepthImage(
                        CUDABuffer<u16>* filtered_depth_buffer_i, 
                        cudaStream_t stream, int depth_type){
  if (is_depth_preprocessing_viewed_){
    Image<u16> filtered_depth(width_, height_);
    filtered_depth_buffer_i->DownloadAsync(stream, &filtered_depth);
    cudaStreamSynchronize(stream);
    std::string window_display_name = WindowDisplayMap[depth_type]; 
    filtered_depth_display->Update(filtered_depth, window_display_name,
                                     static_cast<u16>(0), static_cast<u16>(scaling_));
    ///////////////////////////////
    write_png_count++; 
    std::ostringstream out;
    out << std::setfill('0') << std::setw(5) << std::to_string(write_png_count + 3);
    std::string png_file_name = "../results/d/" + out.str() + ".png"; 
    // filtered_depth.Write(png_file_name); ////////////////////////////////////////////////
  }
}

cameraOrbit mainViewers_::returnKeyframePose(){
  render_window->GetCameraPoseParameters(
                            &camera_free_orbit.offset,
                            &camera_free_orbit.radius,
                            &camera_free_orbit.theta,
                            &camera_free_orbit.phi);

  camera_free_orbit.max_depth = 50; // TODO: output max_depth
  
  return camera_free_orbit; 
}

void mainViewers_::setCameraPose(cameraOrbit camera_free_orbit, SE3f global_T_frame){

  switch(camera_pose_mode) {
    case PlaybackCameraPose:
    { // Determine camera pose from spline-based keyframe playback.

      render_window->SetViewParameters(camera_free_orbit.offset, 
                                        camera_free_orbit.radius, 
                                        camera_free_orbit.theta, 
                                        camera_free_orbit.phi, 
                                        camera_free_orbit.max_depth, 
                                        global_T_frame);
      break;
    }
    case FollowCameraPose:
    { // Use camera pose of frame where all used image data is available.

      Vec3f eye = global_T_frame * Vec3f(0, 0, -0.25f);
      Vec3f look_at = global_T_frame * Vec3f(0, 0, 1.0f);
      Vec3f up = global_T_frame.rotationMatrix() * Vec3f(0, -1.0f, 0);
      
      Vec3f z = (look_at - eye).normalized();  // Forward
      Vec3f x = z.cross(up).normalized(); // Right
      Vec3f y = z.cross(x);
      
      render_window->SetView2(x, y, z, eye, global_T_frame);
      break;
    }
    default: // OtherCameraPose
    {
      // Do not set the visualization camera pose, 
      // so only visualize the input camera. 
      render_window->SetCameraFrustumPose(global_T_frame);
    }
  }

 
}

} // namespace vis