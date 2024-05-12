FIND_PACKAGE( OpenCV REQUIRED ) # modified on Feb. 19, 2023

add_executable(SurfelMeshing
  src/surfel_meshing/asynchronous_meshing.cc
  src/surfel_meshing/asynchronous_meshing.h
  src/surfel_meshing/cuda_util.cuh
  src/surfel_meshing/cuda_depth_processing.cu
  src/surfel_meshing/cuda_depth_processing.cuh
  src/surfel_meshing/cuda_surfel_reconstruction.cc
  src/surfel_meshing/cuda_surfel_reconstruction.h
  src/surfel_meshing/cuda_surfel_reconstruction_kernels.cc
  src/surfel_meshing/cuda_surfel_reconstruction_kernels.cu
  src/surfel_meshing/cuda_surfel_reconstruction_kernels.cuh
  src/surfel_meshing/cuda_surfel_reconstruction_kernels.h
  # src/surfel_meshing/main.cc
  src/main.cc
  src/surfel_meshing/octree.cc
  src/surfel_meshing/octree.h
  src/surfel_meshing/surfel.h
  src/surfel_meshing/surfel_meshing.cc
  src/surfel_meshing/surfel_meshing.h
  src/surfel_meshing/surfel_meshing_render_window.cc
  src/surfel_meshing/surfel_meshing_render_window.h
  
  src/surfel_meshing/ArgumentParser.cc
  src/surfel_meshing/cuda_c_utils.cc
  # src/surfel_meshing/depth_processing_utils.h
  src/surfel_meshing/txt_logger.cc
  src/surfel_meshing/playbackKeyframe.cc
  src/surfel_meshing/fps_adjustment.cc
  src/surfel_meshing/mainViewers.cc
  src/surfel_meshing/meshTerminalUI.cc
  src/surfel_meshing/surfel2cpu_task.cc
  src/surfel_meshing/main_camera_model.cc
  src/surfel_meshing/read_yaml_task.h # OpenCV
)
target_compile_options(SurfelMeshing PUBLIC
  "${LIBVIS_WARNING_OPTIONS}"
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)
target_include_directories(SurfelMeshing PRIVATE
  src
  third_party
  third_party/cub-1.7.4
)
target_link_libraries(SurfelMeshing
  libvis_cuda
  ${BASE_LIB_LIBRARIES} 
  ${X11_LIBRARIES}
  ${OpenCV_LIBS} # modified on Feb. 19, 2023
)
##########################################################################
# Tests.
add_executable(SurfelMeshing_Octree_Test
  src/surfel_meshing/test/test_octree.cc
  src/surfel_meshing/test/main.cc
  src/surfel_meshing/dummy.cu
  # TODO: Compile the file(s) below into a common base lib?
  src/surfel_meshing/octree.cc
)
target_include_directories(SurfelMeshing_Octree_Test PRIVATE
  src
)
target_link_libraries(SurfelMeshing_Octree_Test
  ${BASE_LIB_LIBRARIES}
  gtest
  Threads::Threads
)
add_test(SurfelMeshing_Octree_Test
  SurfelMeshing_Octree_Test
)

add_executable(SurfelMeshing_Triangulation_Test
  src/surfel_meshing/test/test_triangulation.cc
  src/surfel_meshing/test/main.cc
  src/surfel_meshing/dummy.cu
  # TODO: Compile the files below into a common base lib?
  src/surfel_meshing/surfel_meshing.cc
  src/surfel_meshing/octree.cc
  src/surfel_meshing/surfel_meshing_render_window.cc
)
target_include_directories(SurfelMeshing_Triangulation_Test PRIVATE
  src
)
if (UNIX)
  find_package(X11 REQUIRED)
endif()
target_link_libraries(SurfelMeshing_Triangulation_Test
  ${BASE_LIB_LIBRARIES}
  gtest
  Threads::Threads
  ${X11_LIBRARIES}
  libvis_cuda
)
add_test(SurfelMeshing_Triangulation_Test
  SurfelMeshing_Triangulation_Test
)
