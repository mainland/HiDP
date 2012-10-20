
#include "BaseSetup.h"

using namespace std;
using namespace boost;

extern string gOutputFile;
extern vector<string> gInputFiles;


static string gemmInput = " \
#c = alpha * a * b + beta * c  \n\
function matrix_multiply  \n\
input _alpha:float, a[I][K]:float, b[K][J]:float, _beta:float \n\
inout c[I][J]:float \n\
# input float _alpha, a[I][K], b[K][J], _beta; \n\
# inout float c[I][J]; \n\
{ \n\
# float c1[I][J]; \n\
#  c0[I][J][K]:float; \n\
    c1[I][J]:float; \n\
    map{ \n\
    i:=[0:I); j:=[0:J); k:= [0:K); \n\
    _c0 = a[i][k] * b[k][j];  \n\
#    scan();\n\
    } reduce(\"+\", c1[i][j], _c0, k:=[*]); \n\
    c = _alpha * c1 + _beta * c;  \n\
                         } \n\
"; 

class GemmUt: public CaseTest{
public:
    virtual void setupString()
    {
        mInputString = gemmInput;
        mParams.OptLevel = 3;
        mParams.verboseMode = true;
        mParams.OutputFile = string("gemm.cu");
        mParams.ExecutionModels.push_back(string("block"));
        mParams.ExecutionModels.push_back(string("warp"));
        mParams.ExecutionModels.push_back(string("thread"));
    }
};



TEST_F(GemmUt, Gemm)
{
    EXPECT_EQ(gFunctions.size(), 1);
    FuncDefParserNode *curFunction = (*gFunctions.begin()).second;
    curFunction->constructSymbolTable();
    
    curFunction->analyzeInOut();
    curFunction->analyzeShape();
    // backend
    curFunction->fuseToSuperStatement(mParams.OptLevel);
    EXPECT_EQ(curFunction->getNumSuperStatement(), 1);
    curFunction->hprint_w(0);
    
    shared_ptr<HierarchyMode> gpu_mode;
    config_gpu_execution_mode(mParams, gpu_mode);
    
    // execution mode mapping
    bool success = curFunction->mapExecution(gpu_mode);
    EXPECT_TRUE(success);
    

    std::cout << "here.\n";
}
