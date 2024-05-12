#pragma once

#include "surfel_meshing/SurfelMeshingSettings.h" // LIBVIS_ENABLE_TIMING
#include <cuda_runtime.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/opengl_context.h>
#include <libvis/timing.h>
#include <libvis/rgbd_video.h>
#include <unordered_map>

#include "struct_utils.h"

#include "cuda_surfel_reconstruction.h" // CUDASurfelReconstruction* reconstruction
#include "ArgumentParser.h" // SURFELMESHING_PARAMETERS

namespace vis {

template <typename T> 
struct cudaEventSet {
  T depth_image_upload_pre_event;
  T depth_image_upload_post_event;
  T color_image_upload_pre_event;
  T color_image_upload_post_event;
  T frame_start_event;
  T bilateral_filtering_post_event;
  T outlier_filtering_post_event;
  T depth_erosion_post_event;
  T normal_computation_post_event;
  T preprocessing_end_event;
  T frame_end_event;
  T surfel_transfer_start_event;
  T surfel_transfer_end_event;
  T upload_finished_event;
};

class CUDA_C_API {
    public:

        CUDA_C_API(int h, int w, SURFELMESHING_PARAMETERS& data);

        int height, width; 

        SURFELMESHING_PARAMETERS data_; 
        
        // Initialize CUDA events.
        cudaEventSet<cudaEvent_t> cudaEvents;

        // Initialize CUDA streams.
        cudaStream_t stream;
        cudaStream_t upload_stream;

        void CreateEventsStreams(); 
        void AllocateBuffers(); 
        void ShowGPUusage(); 
        void SynchronizeEvents(bool, usize); 
        void PreprocessingElapsedTime(bool); 
        void ReconstructionElapsedTime(CUDASurfelReconstruction& reconstruction); 
        ProfilingTimeSet GetProfilingTimeSet(); 
        ReconstructionTimeSet GetReconstructionTimeSet(); 
        void DestroyEventsStreams();
        void FreeHost();
        void releaseRGBDframe(RGBDVideo<Vec3u8, u16>& rgbd_video, int last_frame_in_window); 

        // CUDA buffers
        std::unordered_map<int, u16*> frame_index_to_depth_buffer_pagelocked;
        std::unordered_map<int, CUDABufferPtr<u16>> frame_index_to_depth_buffer;
        CUDABuffer<u16>* filtered_depth_buffer_A;
        CUDABuffer<u16>* filtered_depth_buffer_B;
        CUDABuffer<float2>* normals_buffer;
        CUDABuffer<float>* radius_buffer;

        Vec3u8* color_buffer_pagelocked;
        Vec3u8* next_color_buffer_pagelocked;
        std::shared_ptr<CUDABuffer<Vec3u8>> color_buffer = nullptr;
        std::shared_ptr<CUDABuffer<Vec3u8>> next_color_buffer = nullptr;

        std::vector<u16*> depth_buffers_pagelocked_cache;
        std::vector<CUDABufferPtr<u16>> depth_buffers_cache;

        // Initialize CUDA-OpenGL interoperation.
        OpenGLContext opengl_context;
        cudaGraphicsResource_t vertex_buffer_resource = nullptr;
        cudaGraphicsResource_t neighbor_index_buffer_resource = nullptr;
        cudaGraphicsResource_t normal_vertex_buffer_resource = nullptr;

        ProfilingTimeSet profilingTime;
        ReconstructionTimeSet reconstructionTime; 
        int kStatsLogInterval_; 
        usize frame_index_ = 0; 

}; 

} // namespace vis