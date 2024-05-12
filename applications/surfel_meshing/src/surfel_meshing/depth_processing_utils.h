
#ifndef _DEPTH_PROCESS_UTILS_H_
#define _DEPTH_PROCESS_UTILS_H_ 

#include <vector>

#include "surfel_meshing/cuda_depth_processing.cuh" // OutlierDepthMapFusionCUDA
#include "surfel_meshing/SurfelMeshingSettings.h"


namespace vis {

// Runs a median filter on the depth map to perform denoising and fill-in.
void MedianFilterAndDensifyDepthMap(const Image<u16>& input, Image<u16>* output) {
  vector<u16> values;
  
  constexpr int kRadius = 1;
  constexpr int kMinNeighbors = 2;
  
  for (int y = 0; y < static_cast<int>(input.height()); ++ y) {
    for (int x = 0; x < static_cast<int>(input.width()); ++ x) {
      values.clear();
      
      int dy_end = std::min<int>(input.height() - 1, y + kRadius);
      for (int dy = std::max<int>(0, static_cast<int>(y) - kRadius);
           dy <= dy_end;
           ++ dy) {
        int dx_end = std::min<int>(input.width() - 1, x + kRadius);
        for (int dx = std::max<int>(0, static_cast<int>(x) - kRadius);
             dx <= dx_end;
             ++ dx) {
          if (input(dx, dy) != 0) {
            values.push_back(input(dx, dy));
          }
        }
      }
      
      if (values.size() >= kMinNeighbors) {
        std::sort(values.begin(), values.end());  // NOTE: slow, need to get center element only
        if (values.size() % 2 == 0) {
          // Take the element which is closer to the average.
          float sum = 0;
          for (u16 value : values) {
            sum += value;
          }
          float average = sum / values.size();
          
          float prev_diff = std::fabs(values[values.size() / 2 - 1] - average);
          float next_diff = std::fabs(values[values.size() / 2] - average);
          (*output)(x, y) = (prev_diff < next_diff) ? values[values.size() / 2 - 1] : values[values.size() / 2];
        } else {
          (*output)(x, y) = values[values.size() / 2];
        }
      } else {
        (*output)(x, y) = input(x, y);
      }
    }
  }
}

// void OutlierRemovalDepthMap(int outlier_frame_count, cudaStream_t stream, \
//                             float outlier_filtering_depth_tolerance_factor, \
//                             CUDABuffer<u16>* filtered_depth_buffer_A, \
//                             CUDABuffer<u16>* filtered_depth_buffer_B, \
//                             float camera_4parameters_0,  \
//                             float camera_4parameters_1,  \
//                             float camera_4parameters_2,  \
//                             float camera_4parameters_3,  \
//                             std::vector< const CUDABuffer_<u16>* >* other_depths,  \
//                             std::vector<CUDAMatrix3x4>* others_TR_reference){
    
//     if (outlier_frame_count == OUTLIER_FILTERING_FRAME_COUNT_2){
//         OutlierDepthMapFusionCUDA<OUTLIER_FILTERING_FRAME_COUNT_2, u16>( \
//                 stream, \
//                 outlier_filtering_depth_tolerance_factor, \
//                 filtered_depth_buffer_A->ToCUDA(), \
//                 camera_4parameters_0, \
//                 camera_4parameters_1, \
//                 camera_4parameters_2, \
//                 camera_4parameters_3, \
//                 other_depths->data(), \
//                 others_TR_reference->data(), \
//                 &filtered_depth_buffer_B->ToCUDA());  
//     }
// }

} // namespace vis

#endif // _DEPTH_PROCESS_UTILS_H_ 
