#include "ArgumentParser.h"

#include <libvis/command_line_parser.h>

#include "surfel_meshing/read_yaml_task.h"

namespace vis {

ArgumentParser::ArgumentParser(int argc, char** argv, std::string& yaml_path){

    // ### Parse parameters ###
    CommandLineParser cmd_parser(argc, argv);

    read_parameters_from_yaml(yaml_path, data); 

    // Dataset playback parameters.
    cmd_parser.NamedParameter(
        "--depth_scaling", &(data.depth_scaling), /*required*/ false,
        "Input depth scaling: input_depth = depth_scaling * depth_in_meters. The default is for TUM RGB-D benchmark datasets.");

    cmd_parser.NamedParameter(
        "--max_pose_interpolation_time_extent", 
        &(data.max_pose_interpolation_time_extent), /*required*/ false,
        "The maximum time (in seconds) between the timestamp of a frame, and the preceding respectively succeeding trajectory pose timestamp, to interpolate the frame's pose. If this threshold is exceeded, the frame will be dropped since no close-enough pose information is available.");

    cmd_parser.NamedParameter(
        "--start_frame", 
        &(data.start_frame), /*required*/ false,
        "First frame of the video to process.");

    cmd_parser.NamedParameter(
        "--end_frame", 
        &(data.end_frame), /*required*/ false,
        "If the video is longer, processing stops after end_frame.");

    cmd_parser.NamedParameter(
        "--pyramid_level", 
        &(data.pyramid_level), /*required*/ false,
        "Specify the scale-space pyramid level to use. 0 uses the original sized images, 1 uses half the original resolution, etc.");

    cmd_parser.NamedParameter(
        "--restrict_fps_to", &(data.fps_restriction), /*required*/ false,
        "Restrict the frames per second to at most the given number.");

    data.step_by_step_playback = cmd_parser.Flag(
        "--step_by_step_playback",
        "Play back video frames step-by-step (do a step by pressing the Return key in the terminal).");

    data.invert_quaternions = cmd_parser.Flag(
        "--invert_quaternions",
        "Invert the quaternions loaded from the poses file.");

    // Surfel reconstruction parameters.
    cmd_parser.NamedParameter(
        "--max_surfel_count", 
        &(data.max_surfel_count), /*required*/ false,
        "Maximum number of surfels. Determines the GPU memory requirements.");

    cmd_parser.NamedParameter(
        "--sensor_noise_factor", 
        &(data.sensor_noise_factor), /*required*/ false,
        "Sensor noise range extent as \"factor times the measured depth\". The real measurement is assumed to be in [(1 - sensor_noise_factor) * depth, (1 + sensor_noise_factor) * depth].");

    cmd_parser.NamedParameter(
        "--max_surfel_confidence", 
        &(data.max_surfel_confidence), /*required*/ false,
        "Maximum value for the surfel confidence. Higher values enable more denoising, lower values faster adaptation to changes.");

    cmd_parser.NamedParameter(
        "--regularizer_weight", 
        &(data.regularizer_weight), /*required*/ false,
        "Weight for the regularization term (w_{reg} in the paper).");

    cmd_parser.NamedParameter(
        "--normal_compatibility_threshold_deg", 
        &(data.normal_compatibility_threshold_deg), /*required*/ false,
        "Angle threshold (in degrees) for considering a measurement normal and a surfel normal to be compatible.");

    cmd_parser.NamedParameter(
        "--regularization_frame_window_size", 
        &(data.regularization_frame_window_size), /*required*/ false,
        "Number of frames for which the regularization of a surfel is continued after it goes out of view.");

    data.do_blending = !cmd_parser.Flag(
        "--disable_blending",
        "Disable observation boundary blending.");

    cmd_parser.NamedParameter(
        "--measurement_blending_radius", 
        &(data.measurement_blending_radius), /*required*/ false,
        "Radius for measurement blending in pixels.");

    cmd_parser.NamedParameter(
        "--regularization_iterations_per_integration_iteration",
        &(data.regularization_iterations_per_integration_iteration), /*required*/ false,
        "Number of regularization (gradient descent) iterations performed per depth integration iteration. Set this to zero to disable regularization.");

    cmd_parser.NamedParameter(
        "--radius_factor_for_regularization_neighbors", 
        &(data.radius_factor_for_regularization_neighbors), /*required*/ false,
        "Factor on the surfel radius for how far regularization neighbors can be away from a surfel.");

    cmd_parser.NamedParameter(
        "--surfel_integration_active_window_size", 
        &(data.surfel_integration_active_window_size), /*required*/ false,
        "Number of frames which need to pass before a surfel becomes inactive. If there are no loop closures, set this to a value larger than the dataset frame count to disable surfel deactivation.");

    // Meshing parameters.
    cmd_parser.NamedParameter(
        "--max_angle_between_normals_deg", 
        &(data.max_angle_between_normals_deg), /*required*/ false,
        "Maximum angle between normals of surfels that are connected by triangulation.");
    data.max_angle_between_normals = M_PI / 180.0f * data.max_angle_between_normals_deg;

    cmd_parser.NamedParameter(
        "--min_triangle_angle_deg", 
        &(data.min_triangle_angle_deg), /*required*/ false,
        "The meshing algorithm attempts to keep triangle angles larger than this.");
    data.min_triangle_angle = M_PI / 180.0 * data.min_triangle_angle_deg;

    cmd_parser.NamedParameter(
        "--max_triangle_angle_deg", 
        &(data.max_triangle_angle_deg), /*required*/ false,
        "The meshing algorithm attempts to keep triangle angles smaller than this.");
    data.max_triangle_angle = M_PI / 180.0 * data.max_triangle_angle_deg;

    cmd_parser.NamedParameter(
        "--max_neighbor_search_range_increase_factor", 
        &(data.max_neighbor_search_range_increase_factor), /*required*/ false,
        "Maximum factor by which the surfel neighbor search range can be increased if the front neighbors are far away.");

    cmd_parser.NamedParameter(
        "--long_edge_tolerance_factor", 
        &(data.long_edge_tolerance_factor), /*required*/ false,
        "Tolerance factor over 'max_neighbor_search_range_increase_factor * surfel_radius' for deciding whether to remesh a triangle with long edges.");

    // "data.asynchronous_triangulation": Makes the meshing proceed synchronously 
    //                                   to the surfel integration (instead of asynchronously).
#ifdef ASYNCHRONOUS_TRIANGULATION
    data.asynchronous_triangulation = true; 
#else  // SYNCHRONOUS_TRIANGULATION
    data.asynchronous_triangulation = false; 
#endif

    data.full_meshing_every_frame = cmd_parser.Flag(
        "--full_meshing_every_frame",
        "Instead of partial remeshing, performs full meshing in every frame. Only implemented for using together with --synchronous_meshing.");

    data.full_retriangulation_at_end = cmd_parser.Flag(
        "--full_retriangulation_at_end",
        "Performs a full retriangulation in the end (after the viewer closes, before the mesh is saved).");

    // Depth preprocessing parameters.
    cmd_parser.NamedParameter(
        "--max_depth", 
        &(data.max_depth), /*required*/ false,
        "Maximum input depth in meters.");

    cmd_parser.NamedParameter(
        "--depth_valid_region_radius", 
        &(data.depth_valid_region_radius), /*required*/ false,
        "Radius of a circle (centered on the image center) with valid depth. Everything outside the circle is considered to be invalid. Used to discard biased depth at the corners of Kinect v1 depth images.");

    cmd_parser.NamedParameter(
        "--observation_angle_threshold_deg", 
        &(data.observation_angle_threshold_deg), /*required*/ false,
        "If the angle between the inverse observation direction and the measured surface normal is larger than this setting, the surface is discarded.");

    cmd_parser.NamedParameter(
        "--depth_erosion_radius", 
        &(data.depth_erosion_radius), /*required*/ false,
        "Radius for depth map erosion (in [0, 3]). Useful to combat foreground fattening artifacts.");

    cmd_parser.NamedParameter(
        "--median_filter_and_densify_iterations", 
        &(data.median_filter_and_densify_iterations), /*required*/ false,
        "Number of iterations of median filtering with hole filling. Disabled by default. Can be useful for noisy time-of-flight data.");

    cmd_parser.NamedParameter(
        "--outlier_filtering_frame_count", 
        &(data.outlier_filtering_frame_count), /*required*/ false,
        "Number of other depth frames to use for outlier filtering of a depth frame. Supported values: 2, 4, 6, 8. Should be reduced if using low-frequency input.");

    cmd_parser.NamedParameter(
        "--outlier_filtering_required_inliers", 
        &(data.outlier_filtering_required_inliers), /*required*/ false,
        "Number of required inliers for accepting a depth value in outlier filtering. With the default value of -1, all other frames (outlier_filtering_frame_count) must be inliers.");

    cmd_parser.NamedParameter(
        "--bilateral_filter_sigma_xy", 
        &(data.bilateral_filter_sigma_xy), /*required*/ false,
        "sigma_xy for depth bilateral filtering, in pixels.");

    cmd_parser.NamedParameter(
        "--bilateral_filter_radius_factor", 
        &(data.bilateral_filter_radius_factor), /*required*/ false,
        "Factor on bilateral_filter_sigma_xy to define the kernel radius for depth bilateral filtering.");

    cmd_parser.NamedParameter(
        "--bilateral_filter_sigma_depth_factor", 
        &(data.bilateral_filter_sigma_depth_factor), /*required*/ false,
        "Factor on the depth to compute sigma_depth for depth bilateral filtering.");

    cmd_parser.NamedParameter(
        "--outlier_filtering_depth_tolerance_factor", 
        &(data.outlier_filtering_depth_tolerance_factor), /*required*/ false,
        "Factor on the depth to define the size of the inlier region for outlier filtering.");

    cmd_parser.NamedParameter(
        "--point_radius_extension_factor", 
        &(data.point_radius_extension_factor), /*required*/ false,
        "Factor by which a point's radius is extended beyond the distance to its farthest neighbor.");

    cmd_parser.NamedParameter(
        "--point_radius_clamp_factor", 
        &(data.point_radius_clamp_factor), /*required*/ false,
        "Factor by which a point's radius can be larger than the distance to its closest neighbor (times sqrt(2)). Larger radii are clamped to this distance.");

    // Octree parameters.
    cmd_parser.NamedParameter(
        "--max_surfels_per_node", 
        &(data.max_surfels_per_node), /*required*/ false,
        "Maximum number of surfels per octree node. Should only affect the runtime.");

    // File export parameters.
    cmd_parser.NamedParameter(
        "--export_mesh", 
        &(data.export_mesh_path), /*required*/ false,
        "Save the final mesh to the given path (as an OBJ file).");

    cmd_parser.NamedParameter(
        "--export_point_cloud", 
        &(data.export_point_cloud_path), /*required*/ false,
        "Save the final (surfel) point cloud to the given path (as a PLY file).");

    // Visualization parameters.
    data.render_camera_frustum = !cmd_parser.Flag(
        "--hide_camera_frustum",
        "Hides the input camera frustum rendering.");

    data.render_new_surfels_as_splats = !cmd_parser.Flag(
        "--hide_new_surfel_splats",
        "Hides the splat rendering of new surfels which are not meshed yet.");

    cmd_parser.NamedParameter(
        "--splat_half_extent_in_pixels", 
        &(data.splat_half_extent_in_pixels), /*required*/ false,
        "Half splat quad extent in pixels.");

    data.triangle_normal_shading = cmd_parser.Flag(
        "--triangle_normal_shading",
        "Colors the mesh triangles based on their triangle normal.");

    data.show_input_images = !cmd_parser.Flag(
        "--hide_input_images",
        "Hides the input images (which are normally shown in separate windows).");

    cmd_parser.NamedParameter(
        "--render_window_default_width", 
        &(data.render_window_default_width), /*required*/ false,
        "Default width of the 3D visualization window.");

    cmd_parser.NamedParameter(
        "--render_window_default_height", 
        &(data.render_window_default_height), /*required*/ false,
        "Default height of the 3D visualization window.");

    data.show_result = !cmd_parser.Flag(
        "--exit_after_processing",
        "After processing the video, exit immediately instead of continuing to show the reconstruction.");

    data.follow_input_camera = !(data.step_by_step_playback);
    cmd_parser.NamedParameter(
        "--follow_input_camera", 
        &(data.follow_input_camera_str), /*required*/ false,
        "Make the visualization camera follow the input camera (true / false).");

    if (data.follow_input_camera_str == "true") {
        data.follow_input_camera = true;
    } else if (data.follow_input_camera_str == "false") {
        data.follow_input_camera = false;
    } else if (!(data.follow_input_camera_str).empty()) {
        // LOG(FATAL) << "Unknown value given for --follow_input_camera parameter: " << data.follow_input_camera_str;
        // return EXIT_FAILURE;
        std::cout << "Unknown value given for --follow_input_camera parameter: " << data.follow_input_camera_str << std::endl; 
        is_parsing_valid = false; 
    }

    cmd_parser.NamedParameter(
        "--record_keyframes", 
        &(data.record_keyframes_path), /*required*/ false,
        "Record keyframes for video recording to the given file. It is recommended to also set --step_by_step_playback and --show_result.");

    cmd_parser.NamedParameter(
        "--playback_keyframes", 
        &(data.playback_keyframes_path), /*required*/ false,
        "Play back keyframes for video recording from the given file.");

    // Debug and evaluation parameters.
    data.debug_depth_preprocessing = cmd_parser.Flag(
        "--debug_depth_preprocessing",
        "Activates debug display of the depth maps at various stages of pre-processing.");

    data.debug_neighbor_rendering = cmd_parser.Flag(
        "--debug_neighbor_rendering",
        "Activates debug rendering of surfel regularization neighbors.");

    data.debug_normal_rendering = cmd_parser.Flag(
        "--debug_normal_rendering",
        "Activates debug rendering of surfel normal vectors.");

    data.visualize_last_update_timestamp = cmd_parser.Flag(
        "--visualize_last_update_timestamp",
        "Show a visualization of the surfel last update timestamps.");

    data.visualize_creation_timestamp = cmd_parser.Flag(
        "--visualize_creation_timestamp",
        "Show a visualization of the surfel creation timestamps.");

    data.visualize_radii = cmd_parser.Flag(
        "--visualize_radii",
        "Show a visualization of the surfel radii.");

    data.visualize_surfel_normals = cmd_parser.Flag(
        "--visualize_surfel_normals",
        "Show a visualization of the surfel normals.");

    cmd_parser.NamedParameter(
        "--log_timings", 
        &(data.timings_log_path), /*required*/ false,
        "Log the timings to the given file.");

    // Required input paths.
    cmd_parser.SequentialParameter(
        &(data.dataset_folder_path), "dataset_folder_path", true,
        "Path to the dataset in TUM RGB-D format.");

    cmd_parser.SequentialParameter(
        &(data.camera_model_filename), "camera_model_filename", true,
        "Filename of the camera model.");

    cmd_parser.SequentialParameter(
        &(data.trajectory_filename), "trajectory_filename", true,
        "Filename of the trajectory file in TUM RGB-D format within the dataset_folder_path (for example, 'trajectory.txt').");

    if (!cmd_parser.CheckParameters()) {
        // return EXIT_FAILURE;
        is_parsing_valid = false; 
        std::cout << "invalid parsing" << std::endl; 
    }
}

}