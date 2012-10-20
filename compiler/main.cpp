#if 1
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <string.h>
//#include "ScopeProfile.h"
#include "include/gpu_mode.h"
#include "include/backend.h"
#include "include/global.h"
//#include "grammar.tab.h"
#include <vector>
#include "simpleopt/SimpleOpt.h"
#include <boost/shared_ptr.hpp>
//#include "grammar.tab.h"
extern FILE *yyin;

using namespace std;
using namespace boost;
//std::vector<StepProfile> gProfile;
int gverbose=0;

extern int yyparse();
extern FILE *yyin;

#if defined(_MSC_VER)
# include <windows.h>
# include <tchar.h>
#else
# define TCHAR      char
# define _T(x)      x
# define _tprintf   printf
# define _tmain     main
# define _ttoi      atoi
#endif

enum { OPT_HELP, 
       OPT_OPTI,
       OPT_OUTPUT,
       OPT_MAPPING
};

CSimpleOpt::SOption g_rgOptions[] = 
  {
    { OPT_HELP, _T("--help"), SO_NONE },
    { OPT_OPTI, _T("-O"), SO_OPT },
    { OPT_OUTPUT, _T("-o"), SO_REQ_SEP },
      { OPT_MAPPING, _T("-m"), SO_MULTI},   // e.g. -m 3 block warp thread
    SO_END_OF_OPTIONS
  };

static void ShowUsage() {
  _tprintf(_T("Usage: basicSample [-O0/1/2/3] [-o OUTPUT] [--help] FILES\n"));
}

extern map<string, FuncDefParserNode *> gFunctions;
int gOptLevel = 0;
//string gOutputFile;
//vector<string> gInputFiles;
int main(int argc, char **argv)
{
  Params params;
    params.OutputFile = string("a.cu");
//  gOutputFile = "a.cu";
  CSimpleOpt args(argc, argv, g_rgOptions, SO_O_NOERR|SO_O_SHORTARG);
  
  while (args.Next()) {
    if (args.LastError() == SO_SUCCESS) {
      if (args.OptionId() == OPT_HELP) {
        ShowUsage(); 
        return 0;
      } 
      else if (args.OptionId() == OPT_OPTI) 
        {
        //        gOptLevel = args.Option
        if (args.OptionArg() == 0)
            params.OptLevel = 0;
        else if (args.OptionArg()[0] == '0')
          params.OptLevel = 0;
        else if (args.OptionArg()[0] == '1')
          params.OptLevel = 1;
        else if (args.OptionArg()[0] == '2')
          params.OptLevel = 2;
        else if (args.OptionArg()[0] == '3')
          params.OptLevel = 3;
        }
      else if (args.OptionId() == OPT_OUTPUT)
        {
            // first arg is the number of mappings
             params.OutputFile = args.OptionArg();
        }
      else if (args.OptionId() == OPT_MAPPING)
      {
          char **num = args.MultiArg(1);
          int num_mappings = _ttoi(num[0]);
          
          num = args.MultiArg(num_mappings);
          for (int i = 0; i != num_mappings; i++)
          {
              params.ExecutionModels.push_back(num[i]);

          }
          
      }
    else {
      _tprintf(_T("Invalid argument: %s\n"), args.OptionText());
      return 1;
    }
    }
  }

  printf("optimization level %d, output file name %s.\n", params.OptLevel, params.OutputFile.c_str());
  printf("there are %d input files.\n", args.FileCount());//, args.Files());

  params.InputFiles.resize(args.FileCount());
  for (int i = 0; i != args.FileCount(); i++)
    {
      params.InputFiles[i] = args.File(i);
    }
  
  // currently support one input file
  if (params.InputFiles.size() == 0)
    {
      printf("Please input source file name.\n");
      return 1;
    }

  if (params.InputFiles.size() > 1)
    {
      printf("only support single file.\n");
      return 1;
    }

  FILE *filename = fopen(params.InputFiles[0].c_str(), "r");
  if (filename == NULL) 
    {
      fprintf(stderr, "Can't open file %s.\n", params.InputFiles[0].c_str());
      return 1;
    }

  yyin = filename;
#if _DEBUG

  FILE *f = fopen(gInputFiles[0].c_str(), "r");
  printf("********parsing file:******* \n");
  char c;
  while((c = fgetc(f)) != EOF) {
    printf("%c", c);
  }
  printf("********parsing file end:******* \n");
  fclose(f);
#endif
  if (! (yyparse() == 0 ))
    {
      fprintf(stderr, "parse error.\n");
      exit(1);
    }
    
  // compiler backend
//  params.OptLevel = gOptLevel;
//  params.OutputFile = gOutputFile;
  try {
    // generate intermediate representations
    fuseStatement(params, gFunctions);
  } catch(...)
    {
      printf("backend exception.\n");
      exit(1);
    }

  // configure machine dependent properties
  shared_ptr<HierarchyMode> gpu_mode;
  config_gpu_execution_mode(params, gpu_mode);

//  map_execution_mode(params, gFunctions, gpu_mode);

  printf("gpumode root has %d upper %d below.\n", gpu_mode->getNumUpper(), gpu_mode->getNumBelow());

//  free_gpu_execution_mode(&gpu_mode);
  fclose(filename);
}
#endif
