#include "gpu_mode.h"
#include "HierarchyMode.h"
#include "HiExecutionModel.h"
#include <climits>
using namespace std;
using namespace boost;

static shared_ptr<HierarchyMode> genModel(const std::string &name)
{
    shared_ptr<HierarchyMode> re;
    if (name == string("kernel"))
        re.reset(new HiThreadModel());
    else if (name == string("block"))
        re.reset(new HiBlockModel());
    else if (name == string("warp"))
        re.reset(new HiWarpModel());
    else if (name == string("subwarp"))
        re.reset(new HiSubWarpModel());
    else if (name == string("subwarp2"))
        re.reset(new HiSubWarp2Model());
    else if (name == string("thread"))
        re.reset(new HiThreadModel());
    else
        assert(0);
    return re;
}

void  config_gpu_execution_mode(const Params &param, boost::shared_ptr<HierarchyMode> &root)
{
    if (param.ExecutionModels.size() == 0)
    {
        shared_ptr<HierarchyMode> kernel = genModel(string("kernel"));
        printf("kernels's num below %d.\n", kernel.get()->getNumBelow());

        shared_ptr<HierarchyMode> block = genModel(string("block"));
        kernel.get()->push_back(block);

        shared_ptr<HierarchyMode> warp = genModel(string("warp"));
        block.get()->push_back(warp);

        shared_ptr<HierarchyMode> subwarp = genModel(string("subwarp"));
        warp.get()->push_back(subwarp);

        shared_ptr<HierarchyMode> thread = genModel(string("thread"));
        subwarp.get()->push_back(thread);

        root = kernel;
    }
    else
    {
        //else param contain limited execution model
        std::vector<std::string>::const_iterator m = param.ExecutionModels.begin();
        root = genModel(*m);
        shared_ptr<HierarchyMode> cur = root;
        m++;
        while (m != param.ExecutionModels.end())
        {
            shared_ptr<HierarchyMode> c = genModel(*m);
            cur->push_back(c);
            assert(cur.get()->getSuitableMaxRange() < c.get()->getSuitableMaxRange());
            cur = c;
            m++;
        }
    }

}

//void  free_gpu_execution_mode(HierarchyMode **root)
//{
  // todo free resource
//}
