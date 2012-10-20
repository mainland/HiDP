#include "global.h"
#include <math.h>
#include <iostream>
#include <cstdio>

void runTest(int argc, char **argv);
void runTest_ell(int argc, char **argv);
void runTest_coo(int argc, char **argv);

bool isIdentical(DTYPE *lhs, DTYPE *rhs, int size,  const char *msg)
{
  for (int i = 0; i != size; i++)
    {
      if (fabs(*lhs - *rhs) > 0.1f)
        {
          printf("%f %f \n", *lhs, *rhs);
          printf("Warning: hidp different from cusp starting at %d %s.\n", i, msg);
          return false;
        }
    }
  return true;
}

int main(int argc, char **argv)
{
          runTest(argc, argv);

          //  runTest_ell(argc,argv);
    //         runTest_coo(argc,argv);
  return 0;
 
}
