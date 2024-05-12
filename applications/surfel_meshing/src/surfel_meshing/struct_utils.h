#pragma once

#include <cmath> //std::fabs
#include <libvis/eigen.h> // Vec3f

namespace vis {

// Helper to use splines from the used spline library with single-dimension values.
struct FloatForSpline {
  FloatForSpline(float value)
      : value(value) {}
  
  float length() const {
    return std::fabs(value);
  }
  
  operator float() const {
    return value;
  }
  
  float value;
};


template <typename T> 
struct SplineParameter {
  T offset_x;
  T offset_y;
  T offset_z;
  T radius; ;
  T theta;
  T phi;
  T max_depth;
};

struct cameraOrbit {
  Vec3f offset;
  float radius;
  float theta;
  float phi;
  float max_depth; 
};

struct MeshOutputInformation {
  u32 latest_mesh_frame_index = 0;
  u32 latest_mesh_surfel_count = 0;
  usize latest_mesh_triangle_count = 0;
};

struct ProfilingTimeSet {
  float elapsed_milliseconds = 0;
  float frame_time_milliseconds = 0;
  float preprocessing_milliseconds = 0;
  float surfel_transfer_milliseconds = 0;
};

struct ReconstructionTimeSet {
  float data_association;
  float surfel_merging;
  float measurement_blending;
  float integration;
  float neighbor_update;
  float new_surfel_creation;
  float regularization;
};

enum TXT_TYPES{
TIMER_TXT,
KEYFRAME_TXT
};

enum WindowTriggerCase
{   WINDOW_INPUT_RGB, 
    WINDOW_INPUT_DEPTH, 
    WINDOW_OUTPUT_MESH, 
    WINDOW_DOWNSAMPLED, // (1) Show downsampled image.
    WINDOW_BILATERAL_FILTERING, // (2) Show bilateral filtering result.
    WINDOW_OUTLIER_FILTERING, // (3) Show outlier filtering result.
    WINDOW_EROSION, // (4) Show erosion result.
    WINDOW_NORMALS_COMPUTED, // (5) Show current depth map result.
    WINDOW_ISOLATED_PIXEL_REMOVAL // (6) Compute PointRadii and RemoveIsolatedPixels
};


} // namespace vis