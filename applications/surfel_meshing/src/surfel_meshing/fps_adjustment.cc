#include "fps_adjustment.h"
namespace vis {

fixFPS::fixFPS(int fps, int kStatsLogInterval){
    kStatsLogInterval_ = kStatsLogInterval; 
    FPS_CONSTANT = fps; 
    MIN_FRAME_TIME = 1.0 / FPS_CONSTANT;
    times_target_fps_reached = 0;
    times_target_fps_not_reached = 0;
}

void fixFPS::sleep(double actual_frame_time, int frame_index){
    // Restrict frame time.    
    // double actual_frame_time = frame_rate_timer.Stop(false);
    if (actual_frame_time <= MIN_FRAME_TIME) {
      ++ times_target_fps_reached;
    } else {
      ++ times_target_fps_not_reached;
    }
    if (frame_index % kStatsLogInterval_ == 0) {
      std::cout << "Target FPS of " << FPS_CONSTANT 
                    << " for integration reached " << times_target_fps_reached 
                    << " times, failed " << times_target_fps_not_reached << " times" << std::endl;
    }
    
    if (actual_frame_time < MIN_FRAME_TIME) {
      slept_microseconds = kSecondsToMicroSeconds * (MIN_FRAME_TIME - actual_frame_time);
      usleep(slept_microseconds);
    }
}


}