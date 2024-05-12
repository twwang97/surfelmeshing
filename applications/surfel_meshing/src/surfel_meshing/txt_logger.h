#pragma once

#include <sstream>      // std::ostringstream
#include <stdio.h> // size_t
#include <iostream> // std::cout
#include <libvis/eigen.h> // Vec3f
#include "struct_utils.h"

namespace vis {

class TxtLogger {
 public:

    TxtLogger(std::string filePath, int file_type);
    void SynchronousFullMeshingTime(size_t, double); 
    void SynchronousMeshingTime(size_t, double, double); 
    void AllProcessedTime(size_t, bool, uint32_t, double, ProfilingTimeSet, ReconstructionTimeSet);  
    void trajectory(size_t frame_index, cameraOrbit camera_free_orbit); 
    void logTitles(); 
    void write();
    
    // txt file to be logged
    std::ostringstream log_file; // timings_log

    // Log the timings to the given file.
    std::string log_filePath; // timings_log_path

    bool is_log_needed; 

    bool is_txt_open; 

    int txt_type_; 
};

}