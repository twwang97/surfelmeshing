// Settings.h

#pragma once

#include <iostream> 
#include <math.h>
#include <limits>

////////////////////////////////////////
//
// The parameters defined below 
// are not allowed to reset
// in the argument parser. 
//
////////////////////////////////////////

#define USE_TUM_DATA

// #define SYNCHRONOUS_TRIANGULATION
#define ASYNCHRONOUS_TRIANGULATION

#define NEXT_TRIANGULATION_TIME_OFFSET 0.05f


#define LIBVIS_ENABLE_TIMING // typedef Timer ConditionalTimer

#define OUTLIER_FILTERING_FRAME_COUNT_2 1 // 2
#define OUTLIER_FILTERING_FRAME_COUNT_4 2 // 4
#define OUTLIER_FILTERING_FRAME_COUNT_6 3 // 6
#define OUTLIER_FILTERING_FRAME_COUNT_8 4 // 8
#define OUTLIER_FILTERING_FRAME_COUNT_SINGLE_10 10

////////////////////////////////////////
//
// The above parameters
// are not allowed to reset
// in the argument parser. 
//
////////////////////////////////////////

namespace vis {

struct SURFELMESHING_PARAMETERS {


#ifdef USE_TUM_DATA
  
    ///////////////////////////////
    //
    // Camera Models
    //
    ///////////////////////////////
    int camera_width = 640;
    int camera_height = 480;
    float camera_fx = 617.2;
    float camera_fy = 609.5;
    float camera_cx = 315.2;
    float camera_cy = 218.1;

    ///////////////////////////////
    //
    // Dataset playback parameters.
    //
    ///////////////////////////////

    // Input depth scaling: input_depth = depth_scaling * depth_in_meters. 
    float depth_scaling = 5000;  // TUM RGB-D

    // The maximum time (in seconds) between the timestamp of a frame, and the preceding respectively succeeding trajectory pose timestamp, to interpolate the frame's pose. 
    // If this threshold is exceeded, the frame will be dropped since no close-enough pose information is available.
    float max_pose_interpolation_time_extent = 0.05f;
    
    // First frame of the video to process.
    int start_frame = 0;
    
    // If the video is longer, processing stops after end_frame.
    int end_frame = std::numeric_limits<int>::max();
    
    // Specify the scale-space pyramid level to use. 0 uses the original sized images, 1 uses half the original resolution, etc.
    int pyramid_level = 0;
    
    // Restrict the frames per second to at most the given number.
    int fps_restriction = 30;
    
    // Show the log data after some time indices
    int kStatsLogInterval = 200;
    
    bool step_by_step_playback;
        // "Play back video frames step-by-step (do a step by pressing the Return key in the terminal).");
    
    bool invert_quaternions;
        // "Invert the quaternions loaded from the poses file.");
    
    ///////////////////////////////
    //
    // Surfel reconstruction parameters.
    //
    ///////////////////////////////
    int max_surfel_count = 30 * 1000 * 1000;  // 30 million.
        // "Maximum number of surfels. Determines the GPU memory requirements.");
    
    float sensor_noise_factor = 0.05f;
        // "Sensor noise range extent as \"factor times the measured depth\". The real measurement is assumed to be in [(1 - sensor_noise_factor) * depth, (1 + sensor_noise_factor) * depth].");
    
    float max_surfel_confidence = 5.0f;
        // "Maximum value for the surfel confidence. Higher values enable more denoising, lower values faster adaptation to changes.");
    
    float regularizer_weight = 10.0f;
        // "Weight for the regularization term (w_{reg} in the paper).");
    
    float normal_compatibility_threshold_deg = 40;
        // "Angle threshold (in degrees) for considering a measurement normal and a surfel normal to be compatible.");
    
    int regularization_frame_window_size = 30;
        // "Number of frames for which the regularization of a surfel is continued after it goes out of view.");
    
    bool do_blending ;
        // "Disable observation boundary blending.");
    
    int measurement_blending_radius = 12;
        // "Radius for measurement blending in pixels.");
    
    int regularization_iterations_per_integration_iteration = 1;
        // "Number of regularization (gradient descent) iterations performed per depth integration iteration. Set this to zero to disable regularization.");
    
    float radius_factor_for_regularization_neighbors = 2;
        // "Factor on the surfel radius for how far regularization neighbors can be away from a surfel.");
    
    int surfel_integration_active_window_size = std::numeric_limits<int>::max();
        // "Number of frames which need to pass before a surfel becomes inactive. 
        // If there are no loop closures, set this to a value larger than the dataset frame count to disable surfel deactivation.");
    
    ///////////////////////////////
    //
    // Meshing parameters.
    //
    ///////////////////////////////

    float max_angle_between_normals_deg = 90.0f;
        // "Maximum angle between normals of surfels that are connected by triangulation.");

    float max_angle_between_normals = M_PI / 180.0f * max_angle_between_normals_deg;
    
    float min_triangle_angle_deg = 10.0f;
        // "The meshing algorithm attempts to keep triangle angles larger than this.");
    float min_triangle_angle = M_PI / 180.0 * min_triangle_angle_deg;
    
    float max_triangle_angle_deg = 170.0f;
        // "The meshing algorithm attempts to keep triangle angles smaller than this.");
    float max_triangle_angle = M_PI / 180.0 * max_triangle_angle_deg;
    
    float max_neighbor_search_range_increase_factor = 2.0f;
        // "Maximum factor by which the surfel neighbor search range can be increased if the front neighbors are far away.");
    
    float long_edge_tolerance_factor = 1.5f;
        // "Tolerance factor over 'max_neighbor_search_range_increase_factor * surfel_radius' for deciding whether to remesh a triangle with long edges.");
    
    bool asynchronous_triangulation;
        // "Makes the meshing proceed synchronously to the surfel integration (instead of asynchronously).");
    
    bool full_meshing_every_frame;
        // "Instead of partial remeshing, performs full meshing in every frame. Only implemented for using together with --synchronous_meshing.");
    
    bool full_retriangulation_at_end;
        // "Performs a full retriangulation in the end (after the viewer closes, before the mesh is saved).");
    
    ///////////////////////////////
    //
    // Depth preprocessing parameters.
    //
    ///////////////////////////////
    float max_depth = 3.0f;
        // "Maximum input depth in meters.");
    
    float depth_valid_region_radius = 333;
        // "Radius of a circle (centered on the image center) with valid depth. Everything outside the circle is considered to be invalid. Used to discard biased depth at the corners of Kinect v1 depth images.");
    
    float observation_angle_threshold_deg = 85;
        // "If the angle between the inverse observation direction and the measured surface normal is larger than this setting, the surface is discarded.");
    
    int depth_erosion_radius = 2;
        // "Radius for depth map erosion (in [0, 3]). Useful to combat foreground fattening artifacts.");
    
    int median_filter_and_densify_iterations = 0;
        // "Number of iterations of median filtering with hole filling. Disabled by default. Can be useful for noisy time-of-flight data.");
    
    int outlier_filtering_frame_count = 8;
        // "Number of other depth frames to use for outlier filtering of a depth frame. Supported values: 2, 4, 6, 8. Should be reduced if using low-frequency input.");
    
    int outlier_filtering_required_inliers = -1;
        // Number of required inliers for accepting a depth value in outlier filtering. 
        // With the default value of -1, all other frames (outlier_filtering_frame_count) must be inliers.;
    
    float bilateral_filter_sigma_xy = 3;
        // "sigma_xy for depth bilateral filtering, in pixels.");
    
    float bilateral_filter_radius_factor = 2.0f;
        // "Factor on bilateral_filter_sigma_xy to define the kernel radius for depth bilateral filtering.");
    
    float bilateral_filter_sigma_depth_factor = 0.05;
        // "Factor on the depth to compute sigma_depth for depth bilateral filtering.");
    
    float outlier_filtering_depth_tolerance_factor = 0.02f;
        // "Factor on the depth to define the size of the inlier region for outlier filtering.");
    
    float point_radius_extension_factor = 1.5f;
        // "Factor by which a point's radius is extended beyond the distance to its farthest neighbor.");
    
    float point_radius_clamp_factor = std::numeric_limits<float>::infinity();
        // "Factor by which a point's radius can be larger than the distance to its closest neighbor (times sqrt(2)). Larger radii are clamped to this distance.");
    
#endif

    ///////////////////////////////
    //
    // Octree parameters.
    //
    ///////////////////////////////

    int max_surfels_per_node = 50;
        // "Maximum number of surfels per octree node. Should only affect the runtime.");
    
    ///////////////////////////////
    //
    // File export parameters.
    // 
    ///////////////////////////////
    std::string export_mesh_path;
        // "Save the final mesh to the given path (as an OBJ file).");
    
    std::string export_point_cloud_path;
        // "Save the final (surfel) point cloud to the given path (as a PLY file).");
    
    ///////////////////////////////
    //
    // Visualization parameters.
    // 
    ///////////////////////////////

    bool render_camera_frustum;
        // "Hides the input camera frustum rendering.");
    
    bool render_new_surfels_as_splats;
        // "Hides the splat rendering of new surfels which are not meshed yet.");
    
    float splat_half_extent_in_pixels = 3.0f;
        // "Half splat quad extent in pixels.");
    
    bool triangle_normal_shading; 
        // "Colors the mesh triangles based on their triangle normal.");
    
    bool show_input_images; 
        // "Hides the input images (which are normally shown in separate windows).");
    
    int render_window_default_width = 1280; // 1280
        // "Default width of the 3D visualization window.");
    
    int render_window_default_height = 720; // 720
        // "Default height of the 3D visualization window.");
    
    bool show_result;
        // "After processing the video, exit immediately instead of continuing to show the reconstruction.");
    
    bool follow_input_camera;

    std::string follow_input_camera_str;
        // "Make the visualization camera follow the input camera (true / false).");

    
    std::string record_keyframes_path;
        // "Record keyframes for video recording to the given file. It is recommended to also set --step_by_step_playback and --show_result.");
    
    std::string playback_keyframes_path;
        // "Play back keyframes for video recording from the given file.");
    
    ///////////////////////////////
    //
    // Debug and evaluation parameters.
    //
    ///////////////////////////////
    
    // Activates debug display of the depth maps at various stages of pre-processing.
    bool debug_depth_preprocessing; 
    
    // Activates debug rendering of surfel regularization neighbors.
    bool debug_neighbor_rendering;
    
    // Activates debug rendering of surfel normal vectors.
    bool debug_normal_rendering; 
    
    // Show a visualization of the surfel last update timestamps.
    bool visualize_last_update_timestamp;
    
    // Show a visualization of the surfel creation timestamps.
    bool visualize_creation_timestamp;
    
    // Show a visualization of the surfel radii.
    bool visualize_radii;
    
    // Show a visualization of the surfel normals.
    bool visualize_surfel_normals;
    
    // Log the timings to the given file.
    std::string timings_log_path;
    
    ///////////////////////////////
    //
    // Required input paths.
    //
    ///////////////////////////////

    // argv[0] execution file

    // argv[1] 
    // Path to the dataset in TUM RGB-D format.
    std::string dataset_folder_path;

    // argv[2]     
    // Filename of the camera model
    std::string camera_model_filename;
    
    // argv[3] 
    // Filename of the associations (including camera pose estimation) within the dataset_folder_path (for example, 'trajectory.txt').
    std::string trajectory_filename;
};

} // namespace vis