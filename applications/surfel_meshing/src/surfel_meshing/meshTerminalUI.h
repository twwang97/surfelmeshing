//
// ** Terminal User Interface (TUI) for mesh visualization ** //
//
//   To Handle key presses (in the terminal)
//

#pragma once

#include <termios.h> // struct termios
#include <stdio.h> // perror
#include <unistd.h> // read
#include <iostream> // std::cout
#include <libvis/logging.h> // LOG(INFO)
#include <libvis/libvis.h> // usize
#include <cuda_runtime.h> // cudaStream_t
#include <signal.h> // to read from or write to the terminal
#include <chrono> // time, sleep
#include <thread> // sleep

#include "surfel_meshing/cuda_surfel_reconstruction.h" // CUDASurfelReconstruction&
#include "surfel_meshing/surfel_meshing.h"
#include "ArgumentParser.h"

namespace vis {

class meshTerminalUI {
 public:
    meshTerminalUI(bool, bool); 
    bool isTriggered(bool is_last_frame); 
    bool handleKeypress(
        usize frame_index, 
        SurfelMeshing& surfel_meshing,
        std::shared_ptr<SurfelMeshingRenderWindow>& render_window, 
        CUDASurfelReconstruction& reconstruction, 
        u32 latest_mesh_frame_index, u32 latest_mesh_surfel_count, 
        cudaStream_t& cuda_stream, ArgumentParser& argparser); 
    char portable_getch(); 
    void ShowMenuOnTerminal(); 

    // Perform a regularization iteration.
    bool RegularizeIterations(
        usize frame_index, 
        SurfelMeshing& surfel_meshing,
        std::shared_ptr<SurfelMeshingRenderWindow>& render_window, 
        CUDASurfelReconstruction& reconstruction, 
        u32 latest_mesh_frame_index, u32 latest_mesh_surfel_count, 
        cudaStream_t& cuda_stream, ArgumentParser& argparser); 

    // Full re-triangulation of all surfels.
    bool FullyRetriangulateSurfels(
        usize frame_index, 
        SurfelMeshing& surfel_meshing,
        std::shared_ptr<SurfelMeshingRenderWindow>& render_window); 

    // Triangulate the selected surfel in debug mode.
    bool TriangulateSurfels(
        SurfelMeshing& surfel_meshing,
        std::shared_ptr<SurfelMeshingRenderWindow>& render_window); 

    // Retriangulate the selected surfel in debug mode.
    bool RetriangulateSurfels(
        SurfelMeshing& surfel_meshing,
        std::shared_ptr<SurfelMeshingRenderWindow>& render_window); 

    // Saves the reconstructed surfels as a point cloud in PLY format.
    bool SavePointCloudAsPLY(
        CUDASurfelReconstruction& reconstruction,
        SurfelMeshing& surfel_meshing,
        const std::string& export_point_cloud_path); 
    
    // Saves the reconstructed colorful surfels as a point cloud in PLY format.
    bool SaveColorfulPointCloudAsPLY(
        CUDASurfelReconstruction& reconstruction,
        SurfelMeshing& surfel_meshing,
        const std::string& export_point_cloud_path, 
        cudaStream_t stream); 
    
    // Saves the reconstructed mesh as an OBJ file.
    bool SaveMeshAsOBJ(
        CUDASurfelReconstruction& reconstruction,
        SurfelMeshing& surfel_meshing,
        const std::string& export_mesh_path,
        cudaStream_t stream); 
        
private:
    bool is_step_by_step_playback_; 
    bool is_show_result_; 
    bool is_last_frame_; 
    bool is_current_triggered; 
}; 

} // namespace vis