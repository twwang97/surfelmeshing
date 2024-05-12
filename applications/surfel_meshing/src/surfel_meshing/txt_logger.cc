#include "txt_logger.h"
namespace vis {

TxtLogger::TxtLogger(std::string filePath, int file_type){
    log_filePath = filePath; 
    is_txt_open = false; 
    is_log_needed = true; 
    txt_type_ = file_type; 
    if (log_filePath.empty()) {
        is_log_needed = false; 
    }

    if(!is_txt_open){
        is_txt_open = true; 
        logTitles(); 
    }
}

void TxtLogger::logTitles(){
    if (!log_filePath.empty()) {
        if (txt_type_ == TIMER_TXT){
            log_file << "# frame_index," 
                    << "preprocessing," 
                    << "data_association," 
                    << "surfel_merging,"
                    << "measurement_blending," 
                    << "integration," 
                    << "neighbor_update," 
                    << "new_surfel_creation," 
                    << "regularization,"
                    << "surfel_count,"
                    << "surfel_transfer,"
                    << "CPU_integration"; 

        } else if (txt_type_ == KEYFRAME_TXT){
            log_file << "# keyframe_index, offset_x, offset_y, offset_z, radius, theta, phi, max_depth" 
                    << std::endl;
        }
    }
}
void TxtLogger::SynchronousFullMeshingTime(size_t frame_index, double full_retriangulation_seconds){
    if(is_log_needed){
        log_file << "frame " << frame_index << std::endl;
        log_file << "-full_meshing " << (1000 * full_retriangulation_seconds) << std::endl;
    }
    
} 

void TxtLogger::SynchronousMeshingTime(size_t frame_index, double remeshing_seconds, double meshing_seconds){
    if(is_log_needed){
        log_file << "frame " << frame_index << std::endl;
        log_file << "-remeshing " << (1000 * remeshing_seconds) << std::endl;
        log_file << "-meshing " << (1000 * meshing_seconds) << std::endl;
    }
    
} 

/*
void TxtLogger::AllProcessedTime(size_t frame_index, 
                            bool did_surfel_transfer, 
                            float preprocessing_milliseconds, float data_association,
                            float surfel_merging, float measurement_blending, 
                            float integration, float neighbor_update, 
                            float new_surfel_creation, float regularization,
                            uint32_t reconstruction_surfel_count, float surfel_transfer_milliseconds){
*/
void TxtLogger::AllProcessedTime(size_t frame_index, 
                            bool did_surfel_transfer, 
                            uint32_t reconstruction_surfel_count,
                            double CPU_complete_elapsed_time, 
                            ProfilingTimeSet profilingTime, ReconstructionTimeSet reconstructionTime){
    if(is_log_needed){
        if(did_surfel_transfer){
            log_file << "\n" << frame_index << ", " 
                    << profilingTime.preprocessing_milliseconds << ", " 
                    << reconstructionTime.data_association << ", " 
                    << reconstructionTime.surfel_merging << ", " 
                    << reconstructionTime.measurement_blending << ", " 
                    << reconstructionTime.integration << ", " 
                    << reconstructionTime.neighbor_update << ", " 
                    << reconstructionTime.new_surfel_creation << ", " 
                    << reconstructionTime.regularization << ", " 
                    << reconstruction_surfel_count << ", "
                    << profilingTime.surfel_transfer_milliseconds << ", "
                    << CPU_complete_elapsed_time << ", ";
        } else {
            log_file << "\n" << frame_index << ", " 
                  << profilingTime.preprocessing_milliseconds << ", " 
                  << reconstructionTime.data_association << ", " 
                  << reconstructionTime.surfel_merging << ", " 
                  << reconstructionTime.measurement_blending << ", " 
                  << reconstructionTime.integration << ", " 
                  << reconstructionTime.neighbor_update << ", " 
                  << reconstructionTime.new_surfel_creation << ", " 
                  << reconstructionTime.regularization << ", " 
                  << reconstruction_surfel_count << ", "
                  << int(0) << ", "
                  << CPU_complete_elapsed_time << ", ";
        }


    }
}

void TxtLogger::trajectory(size_t frame_index, cameraOrbit camera_free_orbit){
    if(is_log_needed){
        log_file << frame_index
                << ", " << camera_free_orbit.offset.transpose()
                << ", " << camera_free_orbit.radius
                << ", " << camera_free_orbit.theta
                << ", " << camera_free_orbit.phi
                << ", " << camera_free_orbit.max_depth 
                << std::endl; 
    }
}

void TxtLogger::write(){
    if(is_log_needed){
        FILE* file = fopen(log_filePath.c_str(), "wb");
        std::string str = log_file.str();
        fwrite(str.c_str(), 1, str.size(), file);
        fclose(file);
    }
}

}