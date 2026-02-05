#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>  
    #include <psapi.h>  
    #include <direct.h>
    #include <process.h>
#else
    #include <sys/stat.h>
    #include <sys/sysinfo.h>
    #include <sys/time.h>
    #include <unistd.h>
    #include <sys/types.h>
#endif

void UseCondition();