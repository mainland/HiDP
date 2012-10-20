#pragma once
#include <string>

// utility functions
bool underscorePrefix(const std::string &str);

// call it whenever you want a unique id (useful for code generator to get a unique variable name)
class UniqueId {
 public:
  static int getUniqueId()
  {
    return curId++;
  }
 private:
  static int curId;
};


void printLevel(int level);
