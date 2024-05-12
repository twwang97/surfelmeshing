//
// ** Terminal User Interface (TUI) for mesh visualization ** //
//
//   To Handle key presses (in the terminal)
//

#include "meshTerminalUI.h"

namespace vis {

// Show help menu on terminal
void meshTerminalUI::ShowMenuOnTerminal(){
  std::cout << "\n  ===== Help Menu =====\n" << std::endl; 
  std::cout << "Q, q: Quit the program." << std::endl; 
  std::cout << "R, r: disable step-by-step playback." << std::endl; 
  std::cout << "A, a: Stronger regularization." << std::endl; 
  std::cout << "S, s: Weaker regularization." << std::endl; 
  std::cout << "D, d: Perform a regularization iteratin." << std::endl; 
  std::cout << "T, t: Full re-triangulation of all surfels." << std::endl; 
  std::cout << "Y, y: Try to triangulate the selected surfel in debug mode." << std::endl; 
  std::cout << "E, e: Retriangulate the selected surfel in debug mode." << std::endl; 
  std::cout << "P, p: Save the mesh." << std::endl; 
}

meshTerminalUI::meshTerminalUI(bool step_by_step_playback, bool show_result){

  // read from the user’s terminal while it is running as a background job. 
  signal(SIGTTIN, SIG_IGN); 
  // generated when a process in a background job attempts to write to the terminal 
  signal(SIGTTOU, SIG_IGN);

  is_step_by_step_playback_ = step_by_step_playback; 
  is_show_result_ = show_result; 

  is_current_triggered = false; 
  if (is_step_by_step_playback_){
    is_current_triggered = true; 
  }
}

bool meshTerminalUI::isTriggered(bool is_last_frame){
  is_last_frame_ = is_last_frame; 
  if(is_current_triggered)
    return true; 
  else if (is_show_result_ && is_last_frame)
    return true; 
  else
    return false; 
}

// Perform a regularization iteration.
bool meshTerminalUI::RegularizeIterations(
    usize frame_index, 
    SurfelMeshing& surfel_meshing,
    std::shared_ptr<SurfelMeshingRenderWindow>& render_window, 
    CUDASurfelReconstruction& reconstruction, 
    u32 latest_mesh_frame_index, u32 latest_mesh_surfel_count, 
    cudaStream_t& cuda_stream, ArgumentParser& argparser) {
  LOG(INFO) << "Regularization iteration ...";
  reconstruction.Regularize(
      cuda_stream, frame_index, 
      argparser.data.regularizer_weight,
      argparser.data.radius_factor_for_regularization_neighbors,
      argparser.data.regularization_frame_window_size);
  
  unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
  reconstruction.UpdateVisualizationBuffers(
      cuda_stream,
      frame_index,
      argparser.data.asynchronous_triangulation ? latest_mesh_frame_index : frame_index,
      argparser.data.asynchronous_triangulation ? latest_mesh_surfel_count : surfel_meshing.surfels().size(),
      argparser.data.surfel_integration_active_window_size,
      argparser.data.visualize_last_update_timestamp,
      argparser.data.visualize_creation_timestamp,
      argparser.data.visualize_radii,
      argparser.data.visualize_surfel_normals);
  render_window->UpdateVisualizationCloudCUDA(
      reconstruction.surfels_size(),
      argparser.data.asynchronous_triangulation ? latest_mesh_surfel_count : 0);
  cudaStreamSynchronize(cuda_stream);
  render_mutex_lock.unlock();
}

// Full re-triangulation of all surfels.
bool meshTerminalUI::FullyRetriangulateSurfels(
    usize frame_index, 
    SurfelMeshing& surfel_meshing,
    std::shared_ptr<SurfelMeshingRenderWindow>& render_window) {
  surfel_meshing.FullRetriangulation();
  std::shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
  surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
  render_window->UpdateVisualizationMesh(visualization_mesh);
  LOG(INFO) << "[frame " << frame_index << " full retriangulation] Triangle count: " << visualization_mesh->triangles().size();
}

// Triangulate the selected surfel in debug mode.
bool meshTerminalUI::TriangulateSurfels(
    SurfelMeshing& surfel_meshing,
    std::shared_ptr<SurfelMeshingRenderWindow>& render_window) {
  LOG(INFO) << "Trying to triangulate surfel " << render_window->selected_surfel_index() << " ...";
  surfel_meshing.SetSurfelToRemesh(render_window->selected_surfel_index());
  surfel_meshing.Triangulate(true); // true means force_debug

  std::shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
  surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
  render_window->UpdateVisualizationMesh(visualization_mesh);
}

// Retriangulate the selected surfel in debug mode.
bool meshTerminalUI::RetriangulateSurfels(
    SurfelMeshing& surfel_meshing,
    std::shared_ptr<SurfelMeshingRenderWindow>& render_window){
  
  const Surfel* surfel = &surfel_meshing.surfels().at(render_window->selected_surfel_index());
  LOG(INFO) << "Retriangulating surfel " << render_window->selected_surfel_index() << " (radius_squared: " << surfel->radius_squared() << ") ...";
  surfel_meshing.RemeshTrianglesAt(const_cast<Surfel*>(surfel), surfel->radius_squared());  // TODO: avoid const_cast
  surfel_meshing.Triangulate(true); // true means force_debug

  std::shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
  surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
  render_window->UpdateVisualizationMesh(visualization_mesh);

}

// Saves the reconstructed surfels as a point cloud in PLY format.
bool meshTerminalUI::SavePointCloudAsPLY(
    CUDASurfelReconstruction& reconstruction,
    SurfelMeshing& surfel_meshing,
    const std::string& export_point_cloud_path){
  CHECK_EQ(surfel_meshing.surfels().size(), reconstruction.surfels_size());
  
  LOG(INFO) << "Saving the final point cloud ...";

  Point3fCloud cloud;
  surfel_meshing.ConvertToPoint3fCloud_position(&cloud);
  
  // u8* color_buffer_cpu = new u8[3 * reconstruction.surfels_size()];
  // CUDABuffer<u8> color_buffer(1, 3 * reconstruction.surfels_size());
  // color_buffer.DownloadAsync(stream, color_buffer_cpu);


  Point3fC3u8NfCloud final_cloud(cloud.size());
  usize index = 0;
  for (usize i = 0; i < cloud.size(); ++ i) {
    final_cloud[i].position() = cloud[i].position();
    final_cloud[i].color() = Vec3u8(255, 255, 255);
    // final_cloud[i].color() = Vec3u8(color_buffer_cpu[3 * i + 0],
    //                             color_buffer_cpu[3 * i + 1],
    //                             color_buffer_cpu[3 * i + 2]);
    
    while (surfel_meshing.surfels()[index].node() == nullptr) {
      ++ index;
    }
    final_cloud[i].normal() = surfel_meshing.surfels()[index].normal();
    ++ index;
  }
  final_cloud.WriteAsPLY(export_point_cloud_path);
  LOG(INFO) << "Wrote " << export_point_cloud_path << ".";
  return true;
}

// Saves the reconstructed colorful surfels as a point cloud in PLY format.
bool meshTerminalUI::SaveColorfulPointCloudAsPLY(
    CUDASurfelReconstruction& reconstruction,
    SurfelMeshing& surfel_meshing,
    const std::string& export_point_cloud_path, 
    cudaStream_t stream){

  // CHECK_EQ(surfel_meshing.surfels().size(), reconstruction.surfels_size());
  const std::size_t surfels_size_0 = surfel_meshing.surfels().size();  ///////////////
  LOG(INFO) << "Saving the final point cloud ...";

  Point3fCloud cloud;
  surfel_meshing.ConvertToPoint3fCloud_position(&cloud);
  
  // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  CUDABuffer<float> position_buffer(1, 3 * surfels_size_0);
  CUDABuffer<u8> color_buffer(1, 3 * surfels_size_0);
  reconstruction.ExportVertices(stream, &position_buffer, &color_buffer);
  u8* color_buffer_cpu = new u8[3 * surfels_size_0];
  color_buffer.DownloadAsync(stream, color_buffer_cpu);
  cudaStreamSynchronize(stream);

  ///////////////
  std::cerr << "surfel_meshing.surfels().size()" << surfel_meshing.surfels().size() << std::endl; 
  std::cerr << reconstruction.surfels_size() << ", " << cloud.size() << std::endl; 
  
  // const int microsec_per_1000surfels = 65; 
  // int sleep_time = reconstruction.surfels_size() * microsec_per_1000surfels / 1000; 
  // std::cerr << "Start to sleep for " << sleep_time << "[µs]" << std::endl; 
  // std::this_thread::sleep_for(std::chrono::microseconds(sleep_time)); // 1s

  Point3fC3u8NfCloud final_cloud(cloud.size());
  usize index = 0;
  for (usize i = 0; i < cloud.size(); ++ i) {
    final_cloud[i].position() = cloud[i].position();
    // final_cloud[i].color() = Vec3u8(255, 255, 255);
    
    while (surfel_meshing.surfels()[index].node() == nullptr) {
      ++ index;
    }
    
    final_cloud[i].color() = Vec3u8(color_buffer_cpu[3 * index + 0],
                                color_buffer_cpu[3 * index + 1],
                                color_buffer_cpu[3 * index + 2]);
    final_cloud[i].normal() = surfel_meshing.surfels()[index].normal();
    ++ index;
    
  }
  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // int elapsed_micro_second = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(); 
  // std::cout << "Time difference = " << elapsed_micro_second << "[µs]" << std::endl;
  // std::cout << 1000.0f * elapsed_micro_second / reconstruction.surfels_size() << std::endl; 

  final_cloud.WriteAsPLY(export_point_cloud_path);
  LOG(INFO) << "Wrote " << export_point_cloud_path << ".";
  return true;
}

// Saves the reconstructed mesh as an OBJ file.
bool meshTerminalUI::SaveMeshAsOBJ(
    CUDASurfelReconstruction& reconstruction,
    SurfelMeshing& surfel_meshing,
    const std::string& export_mesh_path,
    cudaStream_t stream){
  
  const std::size_t surfels_size_0 = surfel_meshing.surfels().size();  ///////////////
  std::cerr << "CHECK_EQ" << std::endl;  ///////////
  // CHECK_EQ(surfel_meshing.surfels().size(), reconstruction.surfels_size()); ///////////////
  
  LOG(INFO) << "Saving the final mesh ...";

  std::shared_ptr<Mesh3fCu8> mesh(new Mesh3fCu8());
  
  // Also use the positions from the surfel_meshing such that positions
  // and the mesh are from a consistent state.
  surfel_meshing.ConvertToMesh3fCu8(mesh.get());
  
  CUDABuffer<float> position_buffer(1, 3 * surfels_size_0);
  CUDABuffer<u8> color_buffer(1, 3 * surfels_size_0);
  reconstruction.ExportVertices(stream, &position_buffer, &color_buffer);
  float* position_buffer_cpu = new float[3 * surfels_size_0];
  u8* color_buffer_cpu = new u8[3 * surfels_size_0];
  position_buffer.DownloadAsync(stream, position_buffer_cpu);
  color_buffer.DownloadAsync(stream, color_buffer_cpu);
  cudaStreamSynchronize(stream);
  usize index = 0;
  ///////////////////////////////
  // CHECK_EQ(mesh->vertices()->size(), reconstruction.surfel_count());
  std::cerr << "CHECK_EQ" << std::endl;  ///////////
  usize boundary_max = (mesh->vertices()->size() < reconstruction.surfel_count()) ? mesh->vertices()->size() : reconstruction.surfel_count(); 
  for (usize i = 0; i < boundary_max; ++ i) {
    if (isnan(position_buffer_cpu[3 * i + 0])) {
      continue;
    }
    
    Point3fC3u8* point = &(*mesh->vertices_mutable())->at(index);
    point->color() = Vec3u8(color_buffer_cpu[3 * i + 0],
                            color_buffer_cpu[3 * i + 1],
                            color_buffer_cpu[3 * i + 2]);
    ++ index;
  }
  ///////////////////
  std::cerr << "CHECK_EQ" << std::endl;  ///////////
  // CHECK_EQ(index, mesh->vertices()->size());
  delete[] color_buffer_cpu;
  delete[] position_buffer_cpu;
  
  // DEBUG:
  // CHECK(mesh->CheckIndexValidity());
  
  if (mesh->WriteAsOBJ(export_mesh_path.c_str())) {
    LOG(INFO) << "Wrote " << export_mesh_path << ".";
    return true;
  } else {
    LOG(ERROR) << "Writing the mesh failed.";
    return false;
  }
}
    
bool meshTerminalUI::handleKeypress(
        usize frame_index, 
        SurfelMeshing& surfel_meshing,
        std::shared_ptr<SurfelMeshingRenderWindow>& render_window, 
        CUDASurfelReconstruction& reconstruction, 
        u32 latest_mesh_frame_index, u32 latest_mesh_surfel_count, 
        cudaStream_t& cuda_stream, ArgumentParser& argparser){
  bool is_program_continued = true; // true to continue
  while (true) {
    int key = portable_getch();
    
    if (key == 10) {
      // Return key.
      // if (!(is_show_result_ && is_last_frame_)) {
      if (is_step_by_step_playback_) {
        break;
      }
    }
    
    if (key == 'h' || key == 'H') {
      ShowMenuOnTerminal(); 
    } else if (key == 'q' || key == 'Q') {
      // Quit the program.
      is_program_continued = false;
      break;
    } else if (key == 'r' || key == 'R') {
      // Run (i.e., disable step-by-step playback).
      argparser.data.step_by_step_playback = !argparser.data.step_by_step_playback; // return
      is_step_by_step_playback_ = argparser.data.step_by_step_playback; // class variable
      break;
    }  else if (key == 'a' || key =='A') {
      // Stronger regularization.
      argparser.data.regularizer_weight *= 1.1f;
      LOG(INFO) << "regularizer_weight: " << argparser.data.regularizer_weight << std::endl;
    } else if (key == 'd' || key =='D') {
      // Perform a regularization iteration.
      RegularizeIterations(frame_index, surfel_meshing, render_window, 
              reconstruction, latest_mesh_frame_index, latest_mesh_surfel_count, 
              cuda_stream, argparser); 
    } else if (key == 's' || key =='S') {
      // Weaker regularization.
      argparser.data.regularizer_weight *= 1 / 1.1f;
      LOG(INFO) << "regularizer_weight: " << argparser.data.regularizer_weight;
    } else if (key == 't' || key == 'T') {
      // Full re-triangulation of all surfels.
      FullyRetriangulateSurfels(frame_index, surfel_meshing, render_window); 
    } else if (key == 'y' || key == 'Y') {
      // Try to triangulate the selected surfel in debug mode.
      TriangulateSurfels(surfel_meshing, render_window); 
    } else if (key == 'e' || key == 'E') {
      // Retriangulate the selected surfel in debug mode.
      RetriangulateSurfels(surfel_meshing, render_window); 
    } else if (key == 'p' || key == 'P') {
      // Save the mesh.
      SaveMeshAsOBJ(reconstruction, surfel_meshing, argparser.data.export_mesh_path, cuda_stream);
    } 
  }
  return is_program_continued; 
}

// Get a key press from the terminal without requiring the Return key to confirm.
// From https://stackoverflow.com/questions/421860
char meshTerminalUI::portable_getch() {
  char buf = 0;
  struct termios old = {0};
  if (tcgetattr(0, &old) < 0) {
    perror("tcsetattr()");
  }
  old.c_lflag &= ~ICANON;
  old.c_lflag &= ~ECHO;
  old.c_cc[VMIN] = 1;
  old.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &old) < 0) {
    perror("tcsetattr ICANON");
  }
  if (read(0, &buf, 1) < 0) {
    perror ("read()");
  }
  old.c_lflag |= ICANON;
  old.c_lflag |= ECHO;
  if (tcsetattr(0, TCSADRAIN, &old) < 0) {
    perror ("tcsetattr ~ICANON");
  }
  return (buf);
}

} // namespace vis