#pragma once

#include "surfel_meshing/SurfelMeshingSettings.h"

namespace vis {


class ArgumentParser {
 public:
    ArgumentParser(int argc, char** argv, std::string& yaml_path);
    struct SURFELMESHING_PARAMETERS data;
    bool is_parsing_valid = true; // true means valid parsing
};

} // namespace vis