#ifndef PROFILING_H
#define PROFILING_H

#include "nvtx3/nvToolsExt.h"

class ProfilingScope
{
public:
    ProfilingScope() = default;
    ProfilingScope(const char* message){
        start(message);
    }

    ~ProfilingScope(){
        while(isRunning)
            stop();
    }

    void start(const char* message){
        nvtxRangePush(message);
        isRunning++;
    }

    void stop(){
        nvtxRangePop();
        isRunning--;
    }

private:
    int isRunning = 0;
};

#endif // PROFILING_H
