#pragma once

#include <iostream>
#include <opencv2/core/persistence.hpp> // cv::FileStorage
#include <math.h> // M_PI
#include "surfel_meshing/SurfelMeshingSettings.h" // SURFELMESHING_PARAMETERS

namespace vis {

template<typename T>
  T read1Parameter(cv::FileStorage& fSettings, 
                    const std::string& name, 
                    bool& found,
                    const bool required = true); 

template<>
float read1Parameter<float>(cv::FileStorage& fSettings, 
                    const std::string& name, 
                    bool& found,
                    const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0.0f;
        }
    }
    else if(!node.isReal()){
        std::cerr << name << " parameter must be a real number, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.real();
    }
}

template<>
int read1Parameter<int>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else{
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0;
        }
    }
    else if(!node.isInt()){
        std::cerr << name << " parameter must be an integer number, aborting..." << std::endl;
        exit(-1);
    }
    else{
        found = true;
        return node.operator int();
    }
}

void read_parameters_from_yaml(std::string& file_yaml_path, SURFELMESHING_PARAMETERS& params){
  
    
    cv::FileStorage fs1;
    fs1.open(file_yaml_path, cv::FileStorage::READ);

    bool found;

    /////  ORB-SLAM3 /////
    
    params.camera_width = 
            read1Parameter<int>(fs1, "Camera.width", found);
    params.camera_height = 
            read1Parameter<int>(fs1, "Camera.height", found);
    params.camera_fx = 
                read1Parameter<float>(fs1, "Camera1.fx", found);
    params.camera_fy = 
                read1Parameter<float>(fs1, "Camera1.fy", found);
    params.camera_cx = 
                read1Parameter<float>(fs1, "Camera1.cx", found);
    params.camera_cy = 
                read1Parameter<float>(fs1, "Camera1.cy", found);
    
    /////  Surfel Meshing /////

    params.depth_scaling = 
                read1Parameter<float>(fs1, "depth_scaling", found);
    params.max_pose_interpolation_time_extent = 
                read1Parameter<float>(fs1, "max_pose_interpolation_time_extent", found);
    params.start_frame = 
            read1Parameter<int>(fs1,"start_frame", found);
    params.end_frame = 
            read1Parameter<int>(fs1,"end_frame", found);
    params.pyramid_level = 
            read1Parameter<int>(fs1,"pyramid_level", found);  
    params.fps_restriction = 
            read1Parameter<int>(fs1,"fps_restriction", found);  
    params.kStatsLogInterval = 
            read1Parameter<int>(fs1,"kStatsLogInterval", found);  
    params.max_surfel_count = 
            read1Parameter<int>(fs1,"max_surfel_count", found);  
    params.sensor_noise_factor = 
                read1Parameter<float>(fs1, "sensor_noise_factor", found);
    params.max_surfel_confidence = 
                read1Parameter<float>(fs1, "max_surfel_confidence", found);
    params.regularizer_weight = 
                read1Parameter<float>(fs1, "regularizer_weight", found);
    params.normal_compatibility_threshold_deg = 
                read1Parameter<float>(fs1, "normal_compatibility_threshold_deg", found);  
    params.regularization_frame_window_size = 
            read1Parameter<int>(fs1,"regularization_frame_window_size", found);  
    params.measurement_blending_radius = 
            read1Parameter<int>(fs1,"measurement_blending_radius", found);  
    params.regularization_iterations_per_integration_iteration = 
            read1Parameter<int>(fs1,"regularization_iterations_per_integration_iteration", found);  
    params.radius_factor_for_regularization_neighbors = 
            read1Parameter<int>(fs1,"radius_factor_for_regularization_neighbors", found);  
    params.surfel_integration_active_window_size = 
            read1Parameter<int>(fs1,"surfel_integration_active_window_size", found);  

    params.max_angle_between_normals_deg = 
                read1Parameter<float>(fs1, "max_angle_between_normals_deg", found);  
    params.max_angle_between_normals = M_PI / 180.0f * params.max_angle_between_normals_deg; 
    params.min_triangle_angle_deg = 
                read1Parameter<float>(fs1, "min_triangle_angle_deg", found);
    params.min_triangle_angle = M_PI / 180.0f * params.min_triangle_angle_deg; 
    params.max_triangle_angle_deg = 
                read1Parameter<float>(fs1, "max_triangle_angle_deg", found);
    params.max_triangle_angle = M_PI / 180.0f * params.max_triangle_angle_deg; 

    params.max_neighbor_search_range_increase_factor = 
                read1Parameter<float>(fs1, "max_neighbor_search_range_increase_factor", found);
    params.long_edge_tolerance_factor = 
                read1Parameter<float>(fs1, "long_edge_tolerance_factor", found);
    params.max_depth = 
                read1Parameter<float>(fs1, "max_depth", found);
    params.depth_valid_region_radius = 
                read1Parameter<float>(fs1, "depth_valid_region_radius", found);
    params.observation_angle_threshold_deg = 
                read1Parameter<float>(fs1, "observation_angle_threshold_deg", found);
    params.depth_erosion_radius = 
            read1Parameter<int>(fs1, "depth_erosion_radius", found);
    params.median_filter_and_densify_iterations = 
            read1Parameter<int>(fs1, "median_filter_and_densify_iterations", found);
    params.outlier_filtering_frame_count = 
            read1Parameter<int>(fs1, "outlier_filtering_frame_count", found);
    params.outlier_filtering_required_inliers = 
            read1Parameter<int>(fs1, "outlier_filtering_required_inliers", found);
    params.bilateral_filter_sigma_depth_factor = 
                read1Parameter<float>(fs1, "bilateral_filter_sigma_depth_factor", found);
    params.bilateral_filter_radius_factor = 
                read1Parameter<float>(fs1, "bilateral_filter_radius_factor", found);
    params.outlier_filtering_depth_tolerance_factor = 
                read1Parameter<float>(fs1, "outlier_filtering_depth_tolerance_factor", found);
    params.point_radius_extension_factor = 
                read1Parameter<float>(fs1, "point_radius_extension_factor", found);
    params.point_radius_clamp_factor = 
                read1Parameter<float>(fs1, "point_radius_clamp_factor", found); 

    params.max_surfels_per_node = 
            read1Parameter<int>(fs1, "max_surfels_per_node", found);
    params.splat_half_extent_in_pixels = 
                read1Parameter<float>(fs1, "splat_half_extent_in_pixels", found); 
    params.render_window_default_width = 
            read1Parameter<int>(fs1, "render_window_default_width", found);
    params.render_window_default_height = 
            read1Parameter<int>(fs1, "render_window_default_height", found);
            
    std::cout << std::endl << "YAML file: " << file_yaml_path << "\nis read."<< std::endl;
    fs1.release(); 
}
} // namespace vis