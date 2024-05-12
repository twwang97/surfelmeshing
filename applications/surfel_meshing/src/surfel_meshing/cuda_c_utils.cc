#include "cuda_c_utils.h"

namespace vis {

// CUDA_C_API::CUDA_C_API(int h, int w, int kStatsLogInterval){
CUDA_C_API::CUDA_C_API(int h, int w, SURFELMESHING_PARAMETERS& data){
    height = h;
    width = w; 
    // kStatsLogInterval_ = kStatsLogInterval; 
    data_.kStatsLogInterval = data.kStatsLogInterval;  
}

void CUDA_C_API::CreateEventsStreams(){
    cudaEventCreate(&cudaEvents.depth_image_upload_pre_event);
    cudaEventCreate(&cudaEvents.depth_image_upload_post_event);
    cudaEventCreate(&cudaEvents.color_image_upload_pre_event);
    cudaEventCreate(&cudaEvents.color_image_upload_post_event);
    cudaEventCreate(&cudaEvents.frame_start_event);
    cudaEventCreate(&cudaEvents.bilateral_filtering_post_event);
    cudaEventCreate(&cudaEvents.outlier_filtering_post_event);
    cudaEventCreate(&cudaEvents.depth_erosion_post_event);
    cudaEventCreate(&cudaEvents.normal_computation_post_event);
    cudaEventCreate(&cudaEvents.preprocessing_end_event);
    cudaEventCreate(&cudaEvents.frame_end_event);
    cudaEventCreate(&cudaEvents.surfel_transfer_start_event);
    cudaEventCreate(&cudaEvents.surfel_transfer_end_event);
    cudaEventCreate(&cudaEvents.upload_finished_event);

    cudaStreamCreate(&stream);
    cudaStreamCreate(&upload_stream);

} // CUDA_C_API::CreateEventsStreams


void CUDA_C_API::AllocateBuffers(){
    // allocate CUDA buffers
    filtered_depth_buffer_A = new CUDABuffer<u16>(height, width); 
    filtered_depth_buffer_B = new CUDABuffer<u16>(height, width); 
    normals_buffer = new CUDABuffer<float2>(height, width);
    radius_buffer = new CUDABuffer<float>(height, width);

    cudaHostAlloc(reinterpret_cast<void**>(&color_buffer_pagelocked), height * width * sizeof(Vec3u8), cudaHostAllocWriteCombined);
    cudaHostAlloc(reinterpret_cast<void**>(&next_color_buffer_pagelocked), height * width * sizeof(Vec3u8), cudaHostAllocWriteCombined);

    // color_buffer = new std::shared_ptr<CUDABuffer<Vec3u8>>(new CUDABuffer<Vec3u8>(height, width));
    // next_color_buffer = new std::shared_ptr<CUDABuffer<Vec3u8>>(new CUDABuffer<Vec3u8>(height, width));
    color_buffer = std::shared_ptr<CUDABuffer<Vec3u8>>(new CUDABuffer<Vec3u8>(height, width));
    next_color_buffer = std::shared_ptr<CUDABuffer<Vec3u8>>(new CUDABuffer<Vec3u8>(height, width));
    
} // void CUDA_C_API::AllocateBuffers()

void CUDA_C_API::ShowGPUusage(){
    // Show memory usage of GPU
  size_t free_bytes;
  size_t total_bytes;
  CUDA_CHECKED_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t used_bytes = total_bytes - free_bytes;
  
  constexpr double kBytesToMiB = 1.0 / (1024.0 * 1024.0);
  LOG(INFO) << "GPU memory usage after initialization: used = " <<
               kBytesToMiB * used_bytes << " MiB, free = " <<
               kBytesToMiB * free_bytes << " MiB, total = " <<
               kBytesToMiB * total_bytes << " MiB\n";

}

void CUDA_C_API::DestroyEventsStreams(){
    cudaEventDestroy(cudaEvents.depth_image_upload_pre_event);
    cudaEventDestroy(cudaEvents.depth_image_upload_post_event);
    cudaEventDestroy(cudaEvents.color_image_upload_pre_event);
    cudaEventDestroy(cudaEvents.color_image_upload_post_event);
    cudaEventDestroy(cudaEvents.frame_start_event);
    cudaEventDestroy(cudaEvents.bilateral_filtering_post_event);
    cudaEventDestroy(cudaEvents.outlier_filtering_post_event);
    cudaEventDestroy(cudaEvents.depth_erosion_post_event);
    cudaEventDestroy(cudaEvents.normal_computation_post_event);
    cudaEventDestroy(cudaEvents.preprocessing_end_event);
    cudaEventDestroy(cudaEvents.frame_end_event);
    cudaEventDestroy(cudaEvents.surfel_transfer_start_event);
    cudaEventDestroy(cudaEvents.surfel_transfer_end_event);

    cudaEventDestroy(cudaEvents.upload_finished_event);

    cudaStreamDestroy(stream);
    cudaStreamDestroy(upload_stream);
}

void CUDA_C_API::SynchronizeEvents(bool did_surfel_transfer, usize frame_index){
    frame_index_ = frame_index; 
    if (did_surfel_transfer)
      cudaEventSynchronize(cudaEvents.surfel_transfer_end_event);
    else
      cudaEventSynchronize(cudaEvents.frame_end_event);
    
    cudaEventSynchronize(cudaEvents.depth_image_upload_post_event);
    // cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.depth_image_upload_pre_event, cudaEvents.depth_image_upload_post_event);
    // Timing::addTime(Timing::getHandle("Upload depth image"), 0.001 * profilingTime.elapsed_milliseconds);
    
    cudaEventSynchronize(cudaEvents.color_image_upload_post_event);
    // cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.color_image_upload_pre_event, cudaEvents.color_image_upload_post_event);
    // Timing::addTime(Timing::getHandle("Upload color image"), 0.001 * profilingTime.elapsed_milliseconds);
}

void CUDA_C_API::releaseRGBDframe(RGBDVideo<Vec3u8, u16>& rgbd_video, int last_frame_in_window){
    // Release frames which are no longer needed.
    // int last_frame_in_window = frame_index - RGBD_FRAME_COUNTS_OFFSET + 1;
    if (last_frame_in_window >= 0) {
      rgbd_video.color_frame_mutable(last_frame_in_window)->ClearImageAndDerivedData();
      rgbd_video.depth_frame_mutable(last_frame_in_window)->ClearImageAndDerivedData();
      depth_buffers_pagelocked_cache.push_back(frame_index_to_depth_buffer_pagelocked.at(last_frame_in_window));
      frame_index_to_depth_buffer_pagelocked.erase(last_frame_in_window);
      depth_buffers_cache.push_back(frame_index_to_depth_buffer.at(last_frame_in_window));
      frame_index_to_depth_buffer.erase(last_frame_in_window);
    }
}

ProfilingTimeSet CUDA_C_API::GetProfilingTimeSet(){
    return profilingTime; 
}

ReconstructionTimeSet CUDA_C_API::GetReconstructionTimeSet(){
    return reconstructionTime; 
}

void CUDA_C_API::PreprocessingElapsedTime(bool did_surfel_transfer){
    profilingTime.elapsed_milliseconds = 0;
    profilingTime.frame_time_milliseconds = 0;
    profilingTime.preprocessing_milliseconds = 0;
    profilingTime.surfel_transfer_milliseconds = 0;

    // cudaEventSynchronize(cudaEvents.depth_image_upload_post_event);
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.depth_image_upload_pre_event, cudaEvents.depth_image_upload_post_event);
    Timing::addTime(Timing::getHandle("Upload depth image"), 0.001 * profilingTime.elapsed_milliseconds);
    
    // cudaEventSynchronize(cudaEvents.color_image_upload_post_event);
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.color_image_upload_pre_event, cudaEvents.color_image_upload_post_event);
    Timing::addTime(Timing::getHandle("Upload color image"), 0.001 * profilingTime.elapsed_milliseconds);
    
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.frame_start_event, cudaEvents.bilateral_filtering_post_event);
    profilingTime.frame_time_milliseconds += profilingTime.elapsed_milliseconds;
    profilingTime.preprocessing_milliseconds += profilingTime.elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Depth bilateral filtering"), 0.001 * profilingTime.elapsed_milliseconds);
    
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.bilateral_filtering_post_event, cudaEvents.outlier_filtering_post_event);
    profilingTime.frame_time_milliseconds += profilingTime.elapsed_milliseconds;
    profilingTime.preprocessing_milliseconds += profilingTime.elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Depth outlier filtering"), 0.001 * profilingTime.elapsed_milliseconds);
    
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.outlier_filtering_post_event, cudaEvents.depth_erosion_post_event);
    profilingTime.frame_time_milliseconds += profilingTime.elapsed_milliseconds;
    profilingTime.preprocessing_milliseconds += profilingTime.elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Depth erosion"), 0.001 * profilingTime.elapsed_milliseconds);
    
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.depth_erosion_post_event, cudaEvents.normal_computation_post_event);
    profilingTime.frame_time_milliseconds += profilingTime.elapsed_milliseconds;
    profilingTime.preprocessing_milliseconds += profilingTime.elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Normal computation"), 0.001 * profilingTime.elapsed_milliseconds);
    
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.normal_computation_post_event, cudaEvents.preprocessing_end_event);
    profilingTime.frame_time_milliseconds += profilingTime.elapsed_milliseconds;
    profilingTime.preprocessing_milliseconds += profilingTime.elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Radius computation"), 0.001 * profilingTime.elapsed_milliseconds);
    
    cudaEventElapsedTime(&profilingTime.elapsed_milliseconds, cudaEvents.preprocessing_end_event, cudaEvents.frame_end_event);
    profilingTime.frame_time_milliseconds += profilingTime.elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Integration"), 0.001 * profilingTime.elapsed_milliseconds);
    
    Timing::addTime(Timing::getHandle("[CUDA frame]"), 0.001 * profilingTime.frame_time_milliseconds);
    
    if (did_surfel_transfer) {
      cudaEventElapsedTime(&profilingTime.surfel_transfer_milliseconds, cudaEvents.surfel_transfer_start_event, cudaEvents.surfel_transfer_end_event);
      Timing::addTime(Timing::getHandle("Surfel transfer to CPU"), 0.001 * profilingTime.surfel_transfer_milliseconds);
    }
}

void CUDA_C_API::ReconstructionElapsedTime(CUDASurfelReconstruction& reconstruction){
    
    reconstruction.GetTimings(
        &reconstructionTime.data_association,
        &reconstructionTime.surfel_merging,
        &reconstructionTime.measurement_blending,
        &reconstructionTime.integration,
        &reconstructionTime.neighbor_update,
        &reconstructionTime.new_surfel_creation,
        &reconstructionTime.regularization);
    Timing::addTime(Timing::getHandle("Integration - data_association"), 0.001 * reconstructionTime.data_association);
    Timing::addTime(Timing::getHandle("Integration - surfel_merging"), 0.001 * reconstructionTime.surfel_merging);
    Timing::addTime(Timing::getHandle("Integration - measurement_blending"), 0.001 * reconstructionTime.measurement_blending);
    Timing::addTime(Timing::getHandle("Integration - integration"), 0.001 * reconstructionTime.integration);
    Timing::addTime(Timing::getHandle("Integration - neighbor_update"), 0.001 * reconstructionTime.neighbor_update);
    Timing::addTime(Timing::getHandle("Integration - new_surfel_creation"), 0.001 * reconstructionTime.new_surfel_creation);
    Timing::addTime(Timing::getHandle("Integration - regularization"), 0.001 * reconstructionTime.regularization);

    if (frame_index_ % data_.kStatsLogInterval == 0) {
      LOG(INFO) << Timing::print(kSortByTotal);
    }
}

void CUDA_C_API::FreeHost(){
    for (u16* ptr : depth_buffers_pagelocked_cache) {
        cudaFreeHost(ptr);
    }
    for (std::pair<int, u16*> item : frame_index_to_depth_buffer_pagelocked) {
        cudaFreeHost(item.second);
    }

    cudaFreeHost(color_buffer_pagelocked);
    cudaFreeHost(next_color_buffer_pagelocked);
}

}