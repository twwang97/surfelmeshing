#include "surfel2cpu_task.h"

namespace vis {

#ifdef ASYNCHRONOUS_TRIANGULATION

surfel2cpu_task::surfel2cpu_task( // SurfelMeshing* surfel_meshing,
                        usize total_frames_count, 
                        // const std::shared_ptr<SurfelMeshingRenderWindow>& render_window, 
                        ArgumentParser& argparser){
    
  // cuda_surfels_cpu_buffers_ = new CUDASurfelsCPU(max_surfel_count); 
  cuda_surfels_cpu_buffers_ = new CUDASurfelsCPU(argparser.data.max_surfel_count);  

  log_timings_ = !argparser.data.timings_log_path.empty(); 

  // // Start background thread if using asynchronous meshing.
  // triangulation_thread.reset(
  //             new AsynchronousMeshing(
  //                             surfel_meshing,
  //                             cuda_surfels_cpu_buffers_,
  //                             log_timings_,
  //                             render_window));
  is_run_asynchronously = true; 
  is_final_visualized = (argparser.data.show_result || !argparser.data.export_mesh_path.empty() || !argparser.data.export_point_cloud_path.empty()); 
  LAST_FRAME_INDEX = total_frames_count; // rgbd_video.frame_count() - argparser.data.outlier_filtering_frame_count / 2 - 1; 
  
  surfel_integration_active_window_size = argparser.data.surfel_integration_active_window_size;
  is_last_update_timestamp_visualized = argparser.data.visualize_last_update_timestamp;
  is_creation_timestamp_visualized = argparser.data.visualize_creation_timestamp;
  is_radii_visualized = argparser.data.visualize_radii;
  is_surfel_normals_visualized = argparser.data.visualize_surfel_normals;
  kStatsLogInterval_ = argparser.data.kStatsLogInterval;

  MeshINFO.latest_mesh_frame_index = 0;
  MeshINFO.latest_mesh_surfel_count = 0;
  MeshINFO.latest_mesh_triangle_count = 0;
}

#else // SYNCHRONOUS_TRIANGULATION

surfel2cpu_task::surfel2cpu_task(bool is_final_visualized_0){
  is_run_asynchronously = false; 
  is_final_visualized = is_final_visualized_0; 
  LAST_FRAME_INDEX = total_frames_count; // rgbd_video.frame_count() - argparser.data.outlier_filtering_frame_count / 2 - 1; 

  MeshINFO.latest_mesh_frame_index = 0;
  MeshINFO.latest_mesh_surfel_count = 0;
  MeshINFO.latest_mesh_triangle_count = 0;
}

#endif

void surfel2cpu_task::setWindow(SurfelMeshing* surfel_meshing, 
              const std::shared_ptr<SurfelMeshingRenderWindow>& render_window){

  // Start background thread if using asynchronous meshing.
  triangulation_thread.reset(
              new AsynchronousMeshing(
                              surfel_meshing,
                              cuda_surfels_cpu_buffers_,
                              log_timings_,
                              render_window));
}

void surfel2cpu_task::end_triangulation_thread(){
#ifdef ASYNCHRONOUS_TRIANGULATION
  if (!is_final_visualized) 
  {
    triangulation_thread->RequestExitAndWaitForIt();
  }
#endif
}

bool surfel2cpu_task::run(usize frame_index, 
                    cudaEvent_t& ccudaEvents_surfel_transfer_start_event, 
                    cudaEvent_t& ccudaEvents_surfel_transfer_end_event, 
                    cudaStream_t& ccuda_stream, 
                    CUDASurfelReconstruction& reconstruction, 
                    std::shared_ptr<SurfelMeshingRenderWindow>& render_window){ 
    
    // ### Surfel meshing handling ###
    
    // Transfer surfels to the CPU if no meshing is in progress,
    // if we expect that the next iteration will start very soon,
    // and for the last frame if the final result is needed.

    bool did_surfel_transfer = false; // final output (is initialized at the beginning)

    // true if it does NOT "simultaneously" running synchronously and triangulated
    no_meshing_in_progress = !is_run_asynchronously || !is_triangulation_in_progress;

    bool is_next_meshing_started_soon = false;
    if (!no_meshing_in_progress) { 
        // if the system is "simultaneously" running synchronously and triangulated, then
      double meshing_elapsed_time = 
          1e-9 * chrono::duration<double, nano>(
              chrono::steady_clock::now() -
              triangulation_thread->latest_triangulation_start_time()).count();
      is_next_meshing_started_soon =
          meshing_elapsed_time >
          triangulation_thread->latest_triangulation_duration() - NEXT_TRIANGULATION_TIME_OFFSET;
    }

    is_last_frame =
        frame_index == LAST_FRAME_INDEX; // rgbd_video.frame_count() - argparser.data.outlier_filtering_frame_count / 2 - 1;


    if (no_meshing_in_progress
        || is_next_meshing_started_soon
        || (is_final_visualized && is_last_frame)) 
    {
      cudaEventRecord(ccudaEvents_surfel_transfer_start_event, ccuda_stream);
#ifdef ASYNCHRONOUS_TRIANGULATION
      triangulation_thread->LockInputData();
#endif
      cuda_surfels_cpu_buffers_->LockWriteBuffers();

      reconstruction.TransferAllToCPU(
          ccuda_stream,
          frame_index,
          cuda_surfels_cpu_buffers_);
      
      cudaEventRecord(ccudaEvents_surfel_transfer_end_event, ccuda_stream);
      cudaStreamSynchronize(ccuda_stream);
      
      // Notify the triangulation thread about new input data.
      // NOTE: It must be avoided to send this notification after the thread
      //       has already started working on the input (due to a previous
      //       notification), so do it while the write buffers are locked.
      //       Otherwise, the thread might later continue its
      //       next iteration before the write buffer was updated again,
      //       resulting in wrong data being used, in particular many surfels
      //       might be at (0, 0, 0).
#ifdef ASYNCHRONOUS_TRIANGULATION
      triangulation_thread->NotifyAboutNewInputSurfelsAlreadyLocked();
#endif
      is_triangulation_in_progress = true;
      
      cuda_surfels_cpu_buffers_->UnlockWriteBuffers();
#ifdef ASYNCHRONOUS_TRIANGULATION
      triangulation_thread->UnlockInputData();
#endif
      did_surfel_transfer = true;
    } // valid surfel transfer
    
    cudaStreamSynchronize(ccuda_stream);
    // complete_frame_timer.Stop();
    
    return did_surfel_transfer; 
}

void surfel2cpu_task::visualize_new_mesh(
                      usize frame_index, 
                      cudaStream_t& ccuda_stream, 
                      CUDASurfelReconstruction& reconstruction, 
                      std::shared_ptr<SurfelMeshingRenderWindow>& render_window){

    // Update the visualization if a new mesh is available.
#ifdef ASYNCHRONOUS_TRIANGULATION // Asynchronous triangulation.
      std::shared_ptr<Mesh3fCu8> output_mesh;
      
      if (is_final_visualized && is_last_frame) {
        // No need for efficiency here, use simple polling waiting
        LOG(INFO) << "Waiting for final mesh ...";
        while (!triangulation_thread->all_work_done()) {
          usleep(0);
        }
        triangulation_thread->RequestExitAndWaitForIt();
        LOG(INFO) << "Got final mesh";
      }
      
      // Get new mesh from the triangulation thread?
      u32 output_frame_index;
      u32 output_surfel_count;
      triangulation_thread->GetOutput(&output_frame_index, &output_surfel_count, &output_mesh);
      
      if (output_mesh) {
        // There is a new mesh.
        MeshINFO.latest_mesh_frame_index = output_frame_index;
        MeshINFO.latest_mesh_surfel_count = output_surfel_count;
        MeshINFO.latest_mesh_triangle_count = output_mesh->triangles().size();
      }
      
      // Update visualization.
      unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
      reconstruction.UpdateVisualizationBuffers(
          ccuda_stream,
          frame_index,
          MeshINFO.latest_mesh_frame_index,
          MeshINFO.latest_mesh_surfel_count,
          surfel_integration_active_window_size,
          is_last_update_timestamp_visualized,
          is_creation_timestamp_visualized,
          is_radii_visualized,
          is_surfel_normals_visualized);
      render_window->UpdateVisualizationCloudCUDA(reconstruction.surfels_size(), MeshINFO.latest_mesh_surfel_count);
      if (output_mesh) {
        render_window->UpdateVisualizationMeshCUDA(output_mesh);
      }
      cudaStreamSynchronize(ccuda_stream);
      render_mutex_lock.unlock();
      if (frame_index % kStatsLogInterval_ == 0) {
        LOG(INFO) << "[frame " << frame_index << "] #surfels: " << reconstruction.surfel_count() << ", #triangles (of latest mesh): " << MeshINFO.latest_mesh_triangle_count;
      }

#else // SYNCHRONOUS_TRIANGULATION // Synchronous triangulation.
      cuda_surfels_cpu_buffers_->WaitForLockAndSwapBuffers();
      surfel_meshing.IntegrateCUDABuffers(frame_index, cuda_surfels_cpu_buffers_);
      
      if (argparser.data.full_meshing_every_frame) {
        double full_retriangulation_seconds = surfel_meshing.FullRetriangulation();
        GPUtimeLogger.SynchronousFullMeshingTime(frame_index, full_retriangulation_seconds); 
      } else {
        ConditionalTimer check_remeshing_timer("CheckRemeshing()");
        surfel_meshing.CheckRemeshing();
        double remeshing_seconds = check_remeshing_timer.Stop();
        
        ConditionalTimer triangulate_timer("Triangulate()");
        surfel_meshing.Triangulate();
        double meshing_seconds = triangulate_timer.Stop();
        GPUtimeLogger.SynchronousMeshingTime(frame_index, remeshing_seconds, meshing_seconds); 
      }
      
      // Update cloud and mesh in the display.
      shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
      surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
      unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
      reconstruction.UpdateVisualizationBuffers(
          ccuda_stream,
          frame_index,
          frame_index,
          surfel_meshing.surfels().size(),
          surfel_integration_active_window_size,
          is_last_update_timestamp_visualized,
          is_creation_timestamp_visualized,
          is_radii_visualized,
          is_surfel_normals_visualized);
      render_window->UpdateVisualizationCloudAndMeshCUDA(reconstruction.surfel_count(), visualization_mesh);
      cudaStreamSynchronize(ccuda_stream);
      render_mutex_lock.unlock();
      LOG(INFO) << "[frame " << frame_index << "] #surfels: " << reconstruction.surfel_count() << ", #triangles: " << visualization_mesh->triangles().size();
    
#endif // end of SYNCHRONOUS_TRIANGULATION

}

} // namespace vis