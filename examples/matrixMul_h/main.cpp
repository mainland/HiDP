
void runTest(int argc, char **argv, int M, int K, int N);


/* Main */
int main(int argc, char** argv)
{    
#if 0
  int K = 2048;
  for (int M = 32; M < 512; M*=2)
    for (int N = 32; N < 512; N*=2)
      runTest(argc, argv, M, K, N);
#else
  // runTest(argc, argv, 2, 2048, 8);
  //  runTest(argc, argv, 16, 32, 16);
  runTest(argc, argv, 2048, 2048, 2048);
#endif
}


