#pragma once

#include <iostream> 
#include <memory>

#include <sstream> 	// std::ostringstream
#include <iomanip> 	// std::setw, std::setfill

#include <libvis/image_display.h> // ImageDisplay
#include <libvis/image_frame.h> // ImageFramePtr
#include <libvis/image.h> // Image
#include <libvis/sophus.h> // SE3f
#include <map> // std::map
#include <libvis/cuda/cuda_buffer.h> // CUDABuffer

#include "struct_utils.h" // struct WindowTriggerCase
#include "surfel_meshing/surfel_meshing_render_window.h" // RenderWindow

namespace vis {

class mainViewers_ {
 public:
    mainViewers_(bool is_input_viewed, bool is_depth_preprocessing_viewed, 
                bool is_playback_valid, bool is_camera_followed, 
                int width, int height, float scaling, 
                int mesh_window_width, int mesh_window_height, 
                bool render_new_surfels_as_splats,
                float splat_half_extent_in_pixels,
                bool triangle_normal_shading,
                bool render_camera_frustum); 
    void showInputRGBD(ImageFramePtr<Vec3u8, SE3f>& color_frame, 
                        ImageFramePtr<u16, SE3f>& input_depth_frame); 
    void showPreprocessedDepthImage(Image<u16>& filtered_depth, int depth_type); 
    void download_showPreprocessedDepthImage(
                        CUDABuffer<u16>* filtered_depth_buffer_i, 
                        cudaStream_t stream, int depth_type); 

    void setCameraPose(cameraOrbit camera_free_orbit, SE3f global_T_frame); 

    void setMeshFirstWindowDirection(const Vec3f& frame_position, const Vec3f& global_position);
    void initialize_cuda_opengl(usize max_point_count,
                                cudaGraphicsResource_t& vertex_buffer_resource,
                                OpenGLContext& opengl_context,
                                std::unique_ptr<Camera>& cameraD,
                                bool debug_neighbor_rendering,
                                bool debug_normal_rendering,
                                cudaGraphicsResource_t& neighbor_index_buffer_resource,
                                cudaGraphicsResource_t& normal_vertex_buffer_resource); 
    cameraOrbit returnKeyframePose(); 

    // Create render windows
    std::shared_ptr<SurfelMeshingRenderWindow> render_window = nullptr;
 private: 
    // Allocate image displays.
    std::shared_ptr<ImageDisplay> raw_rgb_display = nullptr;
    std::shared_ptr<ImageDisplay> raw_depth_display = nullptr;
    std::shared_ptr<ImageDisplay> downscaled_depth_display = nullptr;
    std::shared_ptr<ImageDisplay> filtered_depth_display = nullptr;
    // Create render windows
    std::shared_ptr<RenderWindow> generic_render_window = nullptr; 

    bool is_input_viewed_, is_depth_preprocessing_viewed_; 
    int camera_pose_mode; 
    float scaling_; 
    int width_, height_; 
    cameraOrbit camera_free_orbit; 

    enum CameraPoseMode
    {
        PlaybackCameraPose, 
        FollowCameraPose, 
        OtherCameraPose
    };


    std::map<int, std::string> WindowDisplayMap = {
        { WINDOW_INPUT_RGB, "inputRGB" },
        { WINDOW_INPUT_DEPTH, "inputDepth" },

        { WINDOW_OUTPUT_MESH, "output mesh (3D reconstruction)" },

        // Show downsampled image.
        { WINDOW_DOWNSAMPLED, "downscaled depth" },

        // Show bilateral filtering result.
        { WINDOW_BILATERAL_FILTERING, "CUDA bilateral filtered and cutoff depth" }, 

        // Show outlier filtering result.
        { WINDOW_OUTLIER_FILTERING, "CUDA outlier filtered depth" },

        // Show erosion result.
        { WINDOW_EROSION, "CUDA eroded depth" },

        // Show current depth map result after computing normals
        { WINDOW_NORMALS_COMPUTED, "CUDA bad normal dropped depth" }, 

        // Compute PointRadii and RemoveIsolatedPixels
        { WINDOW_ISOLATED_PIXEL_REMOVAL, "CUDA PointRadii and RemoveIsolatedPixels" }
        
    };

    int write_png_count = 0; 
};

}