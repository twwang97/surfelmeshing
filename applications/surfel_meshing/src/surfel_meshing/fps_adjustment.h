#pragma once
#include <iostream>
#include <unistd.h> // usleep
#include <libvis/libvis.h> // u32

namespace vis {

class fixFPS {
 public:

    fixFPS(int, int);
    void sleep(double, int); 

    int FPS_CONSTANT; // Hz
    double MIN_FRAME_TIME; // second

    u32 times_target_fps_reached = 0;
    u32 times_target_fps_not_reached = 0;

    const float kSecondsToMicroSeconds = 1000 * 1000;

    usize slept_microseconds; 

    int kStatsLogInterval_; 
};

}