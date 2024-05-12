#include "main_camera_model.h"

namespace vis {


main_camera_model::main_camera_model(const Camera& generic_depth_camera_0, 
                                    int pyramid_level){

    // Get potentially scaled depth camera as pinhole camera, determine input size.
    std::unique_ptr<Camera> scaled_camera_0(generic_depth_camera_0.Scaled(1.0f / powf(2, pyramid_level)));
    scaled_camera = std::move(scaled_camera_0); 
    
    depth_camera = reinterpret_cast<const PinholeCamera4f&>(*scaled_camera);

    width = depth_camera.width();
    height = depth_camera.height();

    for (int i = 0; i < 4; i++)
        camera_4parameters[i] = depth_camera.parameters()[i]; 
}

} // namespace vis