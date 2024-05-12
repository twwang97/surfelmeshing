// SurfelMeshing
// https://github.com/puzzlepaint/surfelmeshing.git


// #define LIBVIS_ENABLE_TIMING
#include "surfel_meshing/ArgumentParser.h"
#include "surfel_meshing/main_utils.h"
#include "surfel_meshing/depth_processing_utils.h"
#include "surfel_meshing/struct_utils.h"
#include "surfel_meshing/cuda_c_utils.h"
#include "surfel_meshing/txt_logger.h"
#include "surfel_meshing/playbackKeyframe.h"
#include "surfel_meshing/fps_adjustment.h"
#include "surfel_meshing/mainViewers.h"
#include "surfel_meshing/meshTerminalUI.h"
#include "surfel_meshing/surfel2cpu_task.h"
#include "surfel_meshing/main_camera_model.h"
#include "surfel_meshing/read_rgbd_dataset_task.h"
#include "surfel_meshing/SurfelMeshingSettings.h"


using namespace vis;

int LIBVIS_MAIN(int argc, char** argv) {

  // ### Parse parameters ###
  std::string file_yaml_path = std::string(argv[1]) + "/" + std::string(argv[2]); 
  ArgumentParser argparser(argc, argv, file_yaml_path); 

  // Load dataset.
  RGBD_INPUT_CLASS RGBD_INPUT_DATA(argparser.data.dataset_folder_path, argparser.data.trajectory_filename); 
  RGBDVideo<Vec3u8, u16> rgbd_video = RGBD_INPUT_DATA.init(argparser); 
  
  const usize RGBD_FRAME_COUNTS_OFFSET = argparser.data.outlier_filtering_frame_count; 
  const usize RGBD_TOTAL_FRAMES = rgbd_video.frame_count(); 

  // Get potentially scaled depth camera as pinhole camera, determine input size.
  main_camera_model depth_camera_model(*rgbd_video.depth_camera(), argparser.data.pyramid_level); 

  // Handle keyframe recording or playback.
  TxtLogger KeyframeTrajectoryLogger(argparser.data.record_keyframes_path, KEYFRAME_TXT);
  playbackKeyframe playbackK(argparser.data.playback_keyframes_path); 
  playbackK.read(); 

  // Initialize other classes
  TxtLogger GPUtimeLogger(argparser.data.timings_log_path, TIMER_TXT); 
  fixFPS fpsFixer(argparser.data.fps_restriction, argparser.data.kStatsLogInterval);   
  meshTerminalUI meshTUI( // terminal user interface
                      argparser.data.step_by_step_playback, 
                      argparser.data.show_result); 
  
  
  // Initialize CUDA streams and events.
  CUDA_C_API ccuda(depth_camera_model.height, depth_camera_model.width, argparser.data); 
  ccuda.CreateEventsStreams(); 

  // Allocate CUDA buffers.
  ccuda.AllocateBuffers(); 

  // Allocate image displays and create a window
  mainViewers_ mainViewers(argparser.data.show_input_images, argparser.data.debug_depth_preprocessing, 
                playbackK.isValid(), argparser.data.follow_input_camera, 
                depth_camera_model.width, depth_camera_model.height, 
                argparser.data.depth_scaling * argparser.data.max_depth, 
                argparser.data.render_window_default_width, argparser.data.render_window_default_height, 
                argparser.data.render_new_surfels_as_splats,
                argparser.data.splat_half_extent_in_pixels,
                argparser.data.triangle_normal_shading,
                argparser.data.render_camera_frustum); 

  mainViewers.setMeshFirstWindowDirection(
                rgbd_video.depth_frame_mutable(0)->frame_T_global().rotationMatrix().transpose() * Vec3f(0, 1, 0), 
                rgbd_video.depth_frame_mutable(0)->global_T_frame() * Vec3f(0, 0, 2));

  // Initialize CUDA-OpenGL interoperation.
 mainViewers.initialize_cuda_opengl(argparser.data.max_surfel_count,
                                ccuda.vertex_buffer_resource,
                                ccuda.opengl_context,
                                depth_camera_model.scaled_camera,
                                argparser.data.debug_neighbor_rendering,
                                argparser.data.debug_normal_rendering,
                                ccuda.neighbor_index_buffer_resource,
                                ccuda.normal_vertex_buffer_resource); 
  
  // Allocate reconstruction objects.
  CUDASurfelReconstruction reconstruction(
      argparser.data.max_surfel_count, 
      depth_camera_model.depth_camera, 
      ccuda.vertex_buffer_resource,
      ccuda.neighbor_index_buffer_resource, 
      ccuda.normal_vertex_buffer_resource);
  SurfelMeshing surfel_meshing(
      argparser.data.max_surfels_per_node,
      argparser.data.max_angle_between_normals,
      argparser.data.min_triangle_angle,
      argparser.data.max_triangle_angle,
      argparser.data.max_neighbor_search_range_increase_factor,
      argparser.data.long_edge_tolerance_factor,
      argparser.data.regularization_frame_window_size,
      mainViewers.render_window);
  surfel2cpu_task surfel2cpu(&surfel_meshing,
                            RGBD_TOTAL_FRAMES, 
                            mainViewers.render_window, 
                            argparser);

  // Show memory usage of GPU
  ccuda.ShowGPUusage(); 

  // ### Main loop ###  
  double elapsed_complete_frame_time = 0; 

  // run the image sequentially
  bool quit = false;
  for (usize frame_index = argparser.data.start_frame; frame_index < RGBD_TOTAL_FRAMES && !quit; ++ frame_index) {
    // if (frame_index % 3 == 2 || frame_index % 3 == 1)
    //   continue; //////////////////////////

    Timer frame_rate_timer("");  // "Frame rate timer (with I/O!)"

    bool is_last_frame =
        frame_index == (RGBD_TOTAL_FRAMES - 1);
    
    // ### Input data loading ###
    
    // Since we do not want to measure the time for disk I/O, pre-load the
    // new images for this frame from disk here before starting the frame timer.
    rgbd_video.depth_frame_mutable(frame_index)->GetImage();
    rgbd_video.color_frame_mutable(frame_index)->GetImage();
    
    ConditionalTimer complete_frame_timer("[Integration frame - measured on CPU]");

    cudaEventRecord(ccuda.cudaEvents.upload_finished_event, ccuda.upload_stream);
    
    // Upload all frames up to (frame_index + outlier_filtering_frame_count / 2) to the GPU.
    for (usize test_frame_index = std::max(usize(argparser.data.start_frame), frame_index - RGBD_FRAME_COUNTS_OFFSET); 
          test_frame_index <= frame_index; 
          ++test_frame_index) 
    {
      if (ccuda.frame_index_to_depth_buffer.count(test_frame_index)) {
        continue;
      }
      
      u16** pagelocked_ptr = &ccuda.frame_index_to_depth_buffer_pagelocked[test_frame_index];
      CUDABufferPtr<u16>* buffer_ptr = &ccuda.frame_index_to_depth_buffer[test_frame_index];
      
      if (ccuda.depth_buffers_cache.empty()) {
        cudaHostAlloc(reinterpret_cast<void**>(pagelocked_ptr), depth_camera_model.height * depth_camera_model.width * sizeof(u16), cudaHostAllocWriteCombined);
        
        buffer_ptr->reset(new CUDABuffer<u16>(depth_camera_model.height, depth_camera_model.width));
      } else {
        *pagelocked_ptr = ccuda.depth_buffers_pagelocked_cache.back();
        ccuda.depth_buffers_pagelocked_cache.pop_back();
        
        *buffer_ptr = ccuda.depth_buffers_cache.back();
        ccuda.depth_buffers_cache.pop_back();
      }

      // Perform median filtering and densification.
      // TODO: Do this on the GPU for better performance.
      const Image<u16>* depth_map = rgbd_video.depth_frame_mutable(test_frame_index)->GetImage().get();
      Image<u16> temp_depth_map;
      Image<u16> temp_depth_map_2;
      for (int iteration = 0; iteration < argparser.data.median_filter_and_densify_iterations; ++ iteration) {
        Image<u16>* target_depth_map = (depth_map == &temp_depth_map) ? &temp_depth_map_2 : &temp_depth_map;
        
        target_depth_map->SetSize(depth_map->size());
        MedianFilterAndDensifyDepthMap(*depth_map, target_depth_map);
        
        depth_map = target_depth_map;
      }
      
      if (argparser.data.pyramid_level == 0) {
        memcpy(*pagelocked_ptr,
               depth_map->data(),
               depth_camera_model.height * depth_camera_model.width * sizeof(u16));
      } else {
        if (argparser.data.median_filter_and_densify_iterations > 0) {
          LOG(ERROR) << "Simultaneous downscaling and median filtering of depth maps is not implemented.";
          return EXIT_FAILURE;
        }
        
        Image<u16> downscaled_image(depth_camera_model.width, depth_camera_model.height);
        rgbd_video.depth_frame_mutable(test_frame_index)->GetImage()->DownscaleUsingMedianWhileExcluding(0, depth_camera_model.width, depth_camera_model.height, &downscaled_image);
        
        // DEBUG: (1) Show downsampled image.
        // mainViewers.showPreprocessedDepthImage(downscaled_image, WINDOW_DOWNSAMPLED);  //////////////
        
        memcpy(*pagelocked_ptr,
               downscaled_image.data(),
               depth_camera_model.height * depth_camera_model.width * sizeof(u16));
      }
      cudaEventRecord(ccuda.cudaEvents.depth_image_upload_pre_event, ccuda.upload_stream);
      (*buffer_ptr)->UploadAsync(ccuda.upload_stream, *pagelocked_ptr);
      cudaEventRecord(ccuda.cudaEvents.depth_image_upload_post_event, ccuda.upload_stream);

    } // end for loop (upload some frames to GPU)

    // Swap color image pointers and upload the next color frame to the GPU.
    std::swap(ccuda.next_color_buffer, ccuda.color_buffer);
    std::swap(ccuda.next_color_buffer_pagelocked, ccuda.color_buffer_pagelocked);
    if (argparser.data.pyramid_level == 0) {
      memcpy(ccuda.next_color_buffer_pagelocked,
             rgbd_video.color_frame_mutable(frame_index)->GetImage()->data(),
             depth_camera_model.width * depth_camera_model.height * sizeof(Vec3u8));
    } else {
      memcpy(ccuda.next_color_buffer_pagelocked,
             ImagePyramid(rgbd_video.color_frame_mutable(frame_index).get(), argparser.data.pyramid_level).GetOrComputeResult()->data(),
             depth_camera_model.width * depth_camera_model.height * sizeof(Vec3u8));
    }
    cudaEventRecord(ccuda.cudaEvents.color_image_upload_pre_event, ccuda.upload_stream);
    ccuda.next_color_buffer->UploadAsync(ccuda.upload_stream, ccuda.next_color_buffer_pagelocked);
    cudaEventRecord(ccuda.cudaEvents.color_image_upload_post_event, ccuda.upload_stream);
    
    // If not enough neighboring frames are available for outlier filtering, go to the next frame.
    if (frame_index < static_cast<usize>(argparser.data.start_frame + RGBD_FRAME_COUNTS_OFFSET)) {
      frame_rate_timer.Stop(false);
      complete_frame_timer.Stop(false);
      continue;
    }

    // In the processing stream, wait for this frame's buffers to finish uploading in the upload stream.
    cudaStreamWaitEvent(ccuda.stream, ccuda.cudaEvents.upload_finished_event, 0);
    
    // Get and display input images.
    ImageFramePtr<u16, SE3f> input_depth_frame_i = rgbd_video.depth_frame_mutable(frame_index);
    ImageFramePtr<Vec3u8, SE3f>& input_rgb_frame_i = rgbd_video.color_frame_mutable(frame_index); 
    mainViewers.showInputRGBD(input_rgb_frame_i, input_depth_frame_i); 
    cudaEventRecord(ccuda.cudaEvents.frame_start_event, ccuda.stream);

    // // ### Depth pre-processing ###
    
    // DEBUG: (2) Show bilateral filtering result.
    // Bilateral filtering and depth cutoff.
    CUDABufferPtr<u16> depth_buffer = ccuda.frame_index_to_depth_buffer.at(frame_index);
    BilateralFilteringAndDepthCutoffCUDA(
        ccuda.stream,
        argparser.data.bilateral_filter_sigma_xy,
        argparser.data.bilateral_filter_sigma_depth_factor,
        0, // value_to_ignore
        argparser.data.bilateral_filter_radius_factor,
        argparser.data.depth_scaling * argparser.data.max_depth,
        argparser.data.depth_valid_region_radius,
        depth_buffer->ToCUDA(),
        &ccuda.filtered_depth_buffer_A->ToCUDA());
    cudaEventRecord(ccuda.cudaEvents.bilateral_filtering_post_event, ccuda.stream);
    // mainViewers.download_showPreprocessedDepthImage(ccuda.filtered_depth_buffer_A, ccuda.stream, WINDOW_BILATERAL_FILTERING);   //////////////

    // DEBUG: (3) Show outlier filtering result.
    // Depth outlier filtering.
    // Scale the poses to match the depth scaling. This is faster than scaling the depths of all pixels to match the poses.
    
    SE3f input_depth_frame_scaled_frame_T_global = input_depth_frame_i->frame_T_global();
    input_depth_frame_scaled_frame_T_global.translation() = argparser.data.depth_scaling * input_depth_frame_scaled_frame_T_global.translation();
    
    std::vector< const CUDABuffer_<u16>* > other_depths(argparser.data.outlier_filtering_frame_count);
    std::vector<SE3f> global_TR_others(argparser.data.outlier_filtering_frame_count);
    std::vector<CUDAMatrix3x4> others_TR_reference(argparser.data.outlier_filtering_frame_count);

    for (int i = 0; i < RGBD_FRAME_COUNTS_OFFSET; ++ i) {
      int offset = i + 1;
      other_depths[i] = &ccuda.frame_index_to_depth_buffer.at(frame_index - offset)->ToCUDA();
      global_TR_others[i] = rgbd_video.depth_frame_mutable(frame_index - offset)->global_T_frame();
      global_TR_others[i].translation() = argparser.data.depth_scaling * global_TR_others[i].translation();
      others_TR_reference[i] = CUDAMatrix3x4((input_depth_frame_scaled_frame_T_global * global_TR_others[i]).inverse().matrix3x4());
    }

    if (argparser.data.outlier_filtering_required_inliers == -1 ||
        argparser.data.outlier_filtering_required_inliers == argparser.data.outlier_filtering_frame_count) {

      // Use a macro to pre-compile several versions of the template function.
      #define CALL_OUTLIER_FUSION(other_frame_count) \
          OutlierDepthMapFusionCUDA<other_frame_count + 1, u16>( \
              ccuda.stream, \
              argparser.data.outlier_filtering_depth_tolerance_factor, \
              ccuda.filtered_depth_buffer_A->ToCUDA(), \
              depth_camera_model.camera_4parameters[0], \
              depth_camera_model.camera_4parameters[1], \
              depth_camera_model.camera_4parameters[2], \
              depth_camera_model.camera_4parameters[3], \
              other_depths.data(), \
              others_TR_reference.data(), \
              &ccuda.filtered_depth_buffer_B->ToCUDA())
      if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_2) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_2);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_4) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_4);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_6) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_6);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_8) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_8);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_SINGLE_10) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_SINGLE_10);
      } else {
        LOG(FATAL) << "Unsupported value for outlier_filtering_frame_count: " << argparser.data.outlier_filtering_frame_count;
      }
      #undef CALL_OUTLIER_FUSION
    } else {
      // Use a macro to pre-compile several versions of the template function.
      #define CALL_OUTLIER_FUSION(other_frame_count) \
          OutlierDepthMapFusionCUDA<other_frame_count + 1, u16>( \
              ccuda.stream, \
              argparser.data.outlier_filtering_required_inliers, \
              argparser.data.outlier_filtering_depth_tolerance_factor, \
              ccuda.filtered_depth_buffer_A->ToCUDA(), \
              depth_camera_model.camera_4parameters[0], \
              depth_camera_model.camera_4parameters[1], \
              depth_camera_model.camera_4parameters[2], \
              depth_camera_model.camera_4parameters[3], \
              other_depths.data(), \
              others_TR_reference.data(), \
              &ccuda.filtered_depth_buffer_B->ToCUDA())
      if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_2) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_2);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_4) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_4);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_6) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_6);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_8) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_8);
      } else if (argparser.data.outlier_filtering_frame_count == OUTLIER_FILTERING_FRAME_COUNT_SINGLE_10) {
        CALL_OUTLIER_FUSION(OUTLIER_FILTERING_FRAME_COUNT_SINGLE_10);
      } else {
        LOG(FATAL) << "Unsupported value for outlier_filtering_frame_count: " << argparser.data.outlier_filtering_frame_count;
      }
      #undef CALL_OUTLIER_FUSION
    }
    cudaEventRecord(ccuda.cudaEvents.outlier_filtering_post_event, ccuda.stream);
    // mainViewers.download_showPreprocessedDepthImage(ccuda.filtered_depth_buffer_B, ccuda.stream, WINDOW_OUTLIER_FILTERING);  //////////////

    // DEBUG: (4) Show erosion result.
    // Depth map erosion.
    if (argparser.data.depth_erosion_radius > 0) {
      ErodeDepthMapCUDA(
          ccuda.stream,
          argparser.data.depth_erosion_radius,
          ccuda.filtered_depth_buffer_B->ToCUDA(),
          &ccuda.filtered_depth_buffer_A->ToCUDA());
    } else {
      CopyWithoutBorderCUDA(
          ccuda.stream,
          ccuda.filtered_depth_buffer_B->ToCUDA(),
          &ccuda.filtered_depth_buffer_A->ToCUDA());
    }
    cudaEventRecord(ccuda.cudaEvents.depth_erosion_post_event, ccuda.stream);
    // mainViewers.download_showPreprocessedDepthImage(ccuda.filtered_depth_buffer_A, ccuda.stream, WINDOW_EROSION);  //////////////
    
    // DEBUG: (5) Show current depth map result.
    ComputeNormalsAndDropBadPixelsCUDA(
        ccuda.stream,
        argparser.data.observation_angle_threshold_deg,
        argparser.data.depth_scaling,
        depth_camera_model.camera_4parameters[0],
        depth_camera_model.camera_4parameters[1],
        depth_camera_model.camera_4parameters[2],
        depth_camera_model.camera_4parameters[3],
        ccuda.filtered_depth_buffer_A->ToCUDA(),
        &ccuda.filtered_depth_buffer_B->ToCUDA(),
        &ccuda.normals_buffer->ToCUDA());
    cudaEventRecord(ccuda.cudaEvents.normal_computation_post_event, ccuda.stream);
    // mainViewers.download_showPreprocessedDepthImage(ccuda.filtered_depth_buffer_B, ccuda.stream, WINDOW_NORMALS_COMPUTED);  //////////////
    
    

    // DEBUG: (6) ComputePointRadiiAndRemoveIsolatedPixelsCUDA
    ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
        ccuda.stream,
        argparser.data.point_radius_extension_factor,
        argparser.data.point_radius_clamp_factor,
        argparser.data.depth_scaling,
        depth_camera_model.camera_4parameters[0],
        depth_camera_model.camera_4parameters[1],
        depth_camera_model.camera_4parameters[2],
        depth_camera_model.camera_4parameters[3],
        ccuda.filtered_depth_buffer_B->ToCUDA(),
        &ccuda.radius_buffer->ToCUDA(),
        &ccuda.filtered_depth_buffer_A->ToCUDA());
    // mainViewers.download_showPreprocessedDepthImage(ccuda.filtered_depth_buffer_A, ccuda.stream, WINDOW_ISOLATED_PIXEL_REMOVAL);
    
   
   cudaEventRecord(ccuda.cudaEvents.preprocessing_end_event, ccuda.stream);



    ////////////////////////////////////////////////////
    // cudaEventRecord(ccuda.cudaEvents.bilateral_filtering_post_event, ccuda.stream);
    // cudaEventRecord(ccuda.cudaEvents.outlier_filtering_post_event, ccuda.stream);
    // cudaEventRecord(ccuda.cudaEvents.depth_erosion_post_event, ccuda.stream);
    // cudaEventRecord(ccuda.cudaEvents.normal_computation_post_event, ccuda.stream);
    // cudaEventRecord(ccuda.cudaEvents.preprocessing_end_event, ccuda.stream);

    ///////////////////////////////////////////////////////
    
    // ### Loop closures ###
    
    // Perform surfel deformation if needed.
    // NOTE: This component has been removed to avoid license issues.
    // if (loop_closure) {
    //   // Deform surfels ...
    // }
    
    
    // ### Surfel reconstruction ###
    
    reconstruction.Integrate(
        ccuda.stream,
        frame_index,
        argparser.data.depth_scaling,
        ccuda.filtered_depth_buffer_A,
        *ccuda.normals_buffer,
        *ccuda.radius_buffer,
        *ccuda.color_buffer,
        input_depth_frame_i->global_T_frame(),
        argparser.data.sensor_noise_factor,
        argparser.data.max_surfel_confidence,
        argparser.data.regularizer_weight,
        argparser.data.regularization_frame_window_size,
        argparser.data.do_blending,
        argparser.data.measurement_blending_radius,
        argparser.data.regularization_iterations_per_integration_iteration,
        argparser.data.radius_factor_for_regularization_neighbors,
        argparser.data.normal_compatibility_threshold_deg,
        argparser.data.surfel_integration_active_window_size);
    
    cudaEventRecord(ccuda.cudaEvents.frame_end_event, ccuda.stream);


    SE3f global_T_frame = rgbd_video.depth_frame_mutable(frame_index)->global_T_frame();
    printKeyframeTrajectory(frame_index, RGBD_INPUT_DATA.vTimestamps[frame_index], global_T_frame, rgbd_video.color_frame_mutable(frame_index)->global_T_frame()); 

    // ### Surfels Transfer ###
    // Transfer surfels to the CPU if no meshing is in progress. 
    bool is_surfel_in_CPU = surfel2cpu.run(frame_index, 
                                    ccuda.cudaEvents.surfel_transfer_start_event, 
                                    ccuda.cudaEvents.surfel_transfer_end_event, 
                                    ccuda.stream, 
                                    reconstruction, 
                                    mainViewers.render_window); 

    elapsed_complete_frame_time = complete_frame_timer.Stop(); // stop by Surfel meshing handling
    surfel2cpu.visualize_new_mesh(frame_index, ccuda.stream, reconstruction, mainViewers.render_window); 

    
    // ### Visualization camera pose handling ###

    // render window from the camera pose
    mainViewers.setCameraPose(playbackK.convertSpline2KeyframeTrajectory(frame_index), global_T_frame); 
    
    // For debugging purposes only, notify the render window about the surfel_meshing.
    mainViewers.render_window->SetReconstructionForDebugging(&surfel_meshing);
    
    // ### Profiling ###
    
    // Synchronize with latest event
    ccuda.SynchronizeEvents(is_surfel_in_CPU, frame_index);
    ccuda.PreprocessingElapsedTime(is_surfel_in_CPU); 
    ccuda.ReconstructionElapsedTime(reconstruction); 

    GPUtimeLogger.AllProcessedTime(frame_index, is_surfel_in_CPU, 
                                  reconstruction.surfel_count(), 
                                  elapsed_complete_frame_time, 
                                  ccuda.GetProfilingTimeSet(), 
                                  ccuda.GetReconstructionTimeSet());
                                  
    KeyframeTrajectoryLogger.trajectory(frame_index, mainViewers.returnKeyframePose());
    
    // ### Handle key presses (in the terminal) ###
    if (meshTUI.isTriggered(is_last_frame))
      if(!(meshTUI.handleKeypress(frame_index, surfel_meshing, mainViewers.render_window, 
                  reconstruction, surfel2cpu.MeshINFO.latest_mesh_frame_index, surfel2cpu.MeshINFO.latest_mesh_surfel_count, 
                  ccuda.stream, argparser)))
                  break; 

    // Release frames which are no longer needed.
    ccuda.releaseRGBDframe(rgbd_video, frame_index - RGBD_FRAME_COUNTS_OFFSET); 
    
    // Restrict frame time.
    fpsFixer.sleep(frame_rate_timer.Stop(false), frame_index); 

  }  // End of main loop
  
  
  // ### Save results and cleanup ###

  surfel2cpu.end_triangulation_thread();   
  GPUtimeLogger.write(); 
  KeyframeTrajectoryLogger.write(); 
  
  // Perform retriangulation at end?
  if (argparser.data.full_retriangulation_at_end) {
    surfel_meshing.FullRetriangulation();
  }
  
  // Save the final point cloud.
  if (!argparser.data.export_point_cloud_path.empty()) {
    // meshTUI.SavePointCloudAsPLY(reconstruction, surfel_meshing, argparser.data.export_point_cloud_path); 
    meshTUI.SaveColorfulPointCloudAsPLY(reconstruction, surfel_meshing, argparser.data.export_point_cloud_path, ccuda.stream); 
  }
  
  // Save the final mesh.
  if (!argparser.data.export_mesh_path.empty()) {
    meshTUI.SaveMeshAsOBJ(reconstruction, surfel_meshing, argparser.data.export_mesh_path, ccuda.stream); 
  }

  
  
  // Cleanup
  ccuda.FreeHost(); 
  ccuda.DestroyEventsStreams();   
  
  // Print final timings.
  LOG(INFO) << Timing::print(kSortByTotal);
  

  return EXIT_SUCCESS;
}