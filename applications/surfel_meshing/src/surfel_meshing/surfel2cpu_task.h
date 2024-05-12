#pragma once

#include <iostream> 
#include <cuda_runtime.h> // cudaEvent_t, cudaStream_t
#include <mutex>
#include <unistd.h> // usleep

#include "surfel_meshing/ArgumentParser.h"
#include "surfel_meshing/struct_utils.h"

#include "surfel_meshing/surfel_meshing.h" // SurfelMeshing
#include "surfel_meshing/cuda_surfels_cpu.h" // CUDASurfelsCPU
#include "surfel_meshing/asynchronous_meshing.h" // AsynchronousMeshing
#include "surfel_meshing/cuda_surfel_reconstruction.h" // CUDASurfelReconstruction&

namespace vis {

class surfel2cpu_task {
 public:

#ifdef ASYNCHRONOUS_TRIANGULATION
    surfel2cpu_task(// SurfelMeshing* surfel_meshing,
                        usize total_frames_count, 
                        // const std::shared_ptr<SurfelMeshingRenderWindow>& render_window, 
                        ArgumentParser& argparser);
#else
    surfel2cpu_task(bool is_final_visualized_0); 
#endif

    void setWindow(SurfelMeshing* surfel_meshing, 
            const std::shared_ptr<SurfelMeshingRenderWindow>& render_window);

    bool run(usize frame_index, 
            cudaEvent_t& ccudaEvents_surfel_transfer_start_event, 
            cudaEvent_t& ccudaEvents_surfel_transfer_end_event, 
            cudaStream_t& ccuda_stream, 
            CUDASurfelReconstruction& reconstruction, 
            std::shared_ptr<SurfelMeshingRenderWindow>& render_window); 

    void visualize_new_mesh(
                        usize frame_index, 
                        cudaStream_t& ccuda_stream, 
                        CUDASurfelReconstruction& reconstruction, 
                        std::shared_ptr<SurfelMeshingRenderWindow>& render_window); 

    void end_triangulation_thread(); 
    bool is_last_frame; 

    MeshOutputInformation MeshINFO;

 private:
    std::unique_ptr<AsynchronousMeshing> triangulation_thread = nullptr;
    bool is_run_asynchronously; 
    bool is_triangulation_in_progress = false;
    bool no_meshing_in_progress; 
    // bool did_surfel_transfer = false;
    bool is_final_visualized; 
    usize LAST_FRAME_INDEX; 
    CUDASurfelsCPU* cuda_surfels_cpu_buffers_; 

    int kStatsLogInterval_; 
    int surfel_integration_active_window_size;
    bool is_last_update_timestamp_visualized;
    bool is_creation_timestamp_visualized;
    bool is_radii_visualized;
    bool is_surfel_normals_visualized;
    bool log_timings_; 
};

}