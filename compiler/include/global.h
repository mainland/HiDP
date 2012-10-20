#pragma once
#include <vector>
#include <string>

struct Params{
  int OptLevel;
    bool verboseMode;
  std::string OutputFile;
  std::vector<std::string> InputFiles;
    std::vector<std::string> ExecutionModels;
};
