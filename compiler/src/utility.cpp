#include "utility.h"
#include <iostream>
using namespace std;

bool underscorePrefix(const string &str)
{
  if (str.length() == 0) return false;
  return (str[0] == '_');
}


int UniqueId::curId = 0;


void printLevel(int level)
{
    for (int i = 0; i < level; i++)
        cout << "---";
}