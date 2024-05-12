// function utilities for main.cc

#ifndef _UTILS_H_
#define _UTILS_H_ 

// #include <iomanip> // setfill

#include "libvis/sophus.h" // SE3f


#include "surfel_meshing/cuda_surfel_reconstruction.h" // CUDASurfelReconstruction

namespace vis {

void printKeyframeTrajectory(usize i, double ti, SE3f global_T_frame, SE3f color_global_T_frame){
  // SE3f global_T_frame = rgbd_video.depth_frame_mutable(frame_index + RGBD_FRAME_COUNTS_OFFSET - 1)->global_T_frame();
  // SE3f color_global_T_frame = rgbd_video.color_frame_mutable(frame_index + 1)->global_T_frame();

  printf("KFtraj %d (%.2fs),\tt:(%.3f,%.3f,%.3f),\tq:(%.4f,%.4f,%.4f,%.4f)\n", 
                              i,
                              ti, // TimeStamp
                              global_T_frame.translation().x() ,
                              global_T_frame.translation().y() ,
                              global_T_frame.translation().z() ,
                              global_T_frame.unit_quaternion().x(), 
                              global_T_frame.unit_quaternion().y(), 
                              global_T_frame.unit_quaternion().z(), 
                              global_T_frame.unit_quaternion().w());   
}

int LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        std::string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            std::string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 0; 
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 0; 
    }
    return nImages; 
} // LoadImages

template <typename T>
int Load_ImageName_Trajectory(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps, 
                std::vector<Sophus::SE3<T>> &vTrajectories)
{
    std::ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        std::string string_line;
        getline(fAssociation, string_line);
        if (string_line.size() == 0 || string_line[0] == '#') {
          continue;
        }
        if(!string_line.empty())
        {
            std::stringstream ss;
            ss << string_line; 
            double t;
            std::string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
            
            Sophus::Vector3f cam_translation;
            Eigen::Quaternionf cam_rotation;
            ss >> cam_translation[0];
            ss >> cam_translation[1];
            ss >> cam_translation[2];
            ss >> cam_rotation.x();
            ss >> cam_rotation.y();
            ss >> cam_rotation.z();
            ss >> cam_rotation.w();
            // cam_rotation.normalize(); 
            Sophus::SE3<T> global_T_frame = 
                          Sophus::SE3<T>(cam_rotation.cast<T>(),
                                        cam_translation.cast<T>());
            vTrajectories.push_back(global_T_frame); 

        }
    }

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 0; 
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 0; 
    }
    return nImages; 
} // Load_ImageName_Trajectory

} // namespace vis

#endif // _UTILS_H_ 