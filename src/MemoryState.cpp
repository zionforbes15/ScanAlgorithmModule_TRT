#include "./MemoryState.h"
#include <cstdio>

// 获取当前进程 PID
inline int GetCurrentPid() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

#ifndef _WIN32
#define PROCESS_ITEM 14

static const char* get_items(const char* buffer, unsigned int item) {
    if (buffer == nullptr) return nullptr;
    
    const char* p = buffer;
    size_t len = strlen(buffer);
    size_t count = 0;
    size_t target_count = static_cast<size_t>(item) - 1;
    if (item == 0) return buffer; 

    for (size_t i = 0; i < len; i++) {
        if (' ' == *p) {
            count++;
            if (count == target_count) { 
                p++; 
                break; 
            }
        }
        p++;
    }
    return p;
}

static inline unsigned long get_cpu_total_occupy() {
    unsigned long user, nice, system, idle;
    FILE* fd = fopen("/proc/stat", "r");
    if (!fd) return 0;
    char buff[1024], name[64];
    if (fgets(buff, sizeof(buff), fd)) {
        sscanf(buff, "%s %ld %ld %ld %ld", name, &user, &nice, &system, &idle);
    }
    fclose(fd);
    return (user + nice + system + idle);
}

static inline unsigned long get_cpu_proc_occupy(int pid) {
    unsigned long utime, stime, cutime, cstime;
    char file_name[64];
    sprintf(file_name, "/proc/%d/stat", pid);
    FILE* fd = fopen(file_name, "r");
    if (!fd) return 0;
    char line_buff[1024];
    if (fgets(line_buff, sizeof(line_buff), fd)) {
        const char* q = get_items(line_buff, PROCESS_ITEM);
        sscanf(q, "%ld %ld %ld %ld", &utime, &stime, &cutime, &cstime);
    }
    fclose(fd);
    return (utime + stime + cutime + cstime);
}
#endif

// 获取 CPU 使用率
inline float GetCpuUsageRatio(int pid) {
#ifdef _WIN32
    static int64_t last_time = 0;
	static int64_t last_system_time = 0;

	FILETIME now;
	FILETIME creation_time;
	FILETIME exit_time;
	FILETIME kernel_time;
	FILETIME user_time;
	int64_t system_time;
	int64_t time;
	int64_t system_time_delta;
	int64_t time_delta;

	// get cpu num
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	int cpu_num = info.dwNumberOfProcessors;

	float cpu_ratio = 0.0;

	// get process hanlde by pid
	HANDLE process = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
	// use GetCurrentProcess() can get current process and no need to close handle

	// get now time
	GetSystemTimeAsFileTime(&now);

	if (!GetProcessTimes(process, &creation_time, &exit_time, &kernel_time, &user_time))
	{
		// We don't assert here because in some cases (such as in the Task Manager)  
		// we may call this function on a process that has just exited but we have  
		// not yet received the notification.  
		printf("GetCpuUsageRatio GetProcessTimes failed\n");
		return 0.0;
	}
	// should handle the multiple cpu num
	system_time = (convert_time_format(&kernel_time) + convert_time_format(&user_time)) / cpu_num;
	time = convert_time_format(&now);
	if ((last_system_time == 0) || (last_time == 0))
	{
		// First call, just set the last values.  
		last_system_time = system_time;
		last_time = time;
		return 0.0;
	}
	system_time_delta = system_time - last_system_time;
	time_delta = time - last_time;
	CloseHandle(process);
	if (time_delta == 0)
	{
		printf("GetCpuUsageRatio time_delta is 0, error\n");
		return 0.0;
	}
	// We add time_delta / 2 so the result is rounded.  
	cpu_ratio = (int)((system_time_delta * 100 + time_delta / 2) / time_delta); // the % unit
	last_system_time = system_time;
	last_time = time;
	cpu_ratio /= 100.0; // convert to float number
	return cpu_ratio; 
#else
    unsigned long total1 = get_cpu_total_occupy();
    unsigned long proc1 = get_cpu_proc_occupy(pid);
    usleep(200000); // 采样间隔 200ms
    unsigned long total2 = get_cpu_total_occupy();
    unsigned long proc2 = get_cpu_proc_occupy(pid);
    
    float pcpu = 0.0;
    if (total2 != total1)
        pcpu = (float)(proc2 - proc1) / (float)(total2 - total1);
    
    pcpu *= get_nprocs(); // 乘以 CPU 核心数
    return pcpu;
#endif
}

// 获取内存使用量 (MB)
inline float GetMemoryUsage(int pid) {
#ifdef _WIN32
    uint64_t mem = 0, vmem = 0;
	PROCESS_MEMORY_COUNTERS pmc;
	// get process hanlde by pid
	HANDLE process = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
	if (GetProcessMemoryInfo(process, &pmc, sizeof(pmc)))
	{
		mem = pmc.WorkingSetSize;
		vmem = pmc.PagefileUsage;
	}
	CloseHandle(process);
	// use GetCurrentProcess() can get current process and no need to close handle
	// convert mem from B to MB
	return mem / 1024.0 / 1024.0;
#else
    char file_name[64];
    sprintf(file_name, "/proc/%d/status", pid);
    FILE* fd = fopen(file_name, "r");
    if (!fd) return 0;
    
    char line[512], name[64];
    int vmrss = 0;
    while (fgets(line, sizeof(line), fd)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "%s %d", name, &vmrss);
            break;
        }
    }
    fclose(fd);
    return vmrss / 1024.0f; // KB 转 MB
#endif
}

void UseCondition() {
    int current_pid = GetCurrentPid();
    float cpu = GetCpuUsageRatio(current_pid);
    float mem = GetMemoryUsage(current_pid);
    
    std::cout << "--- 资源监控 ---" << std::endl;
    std::cout << "PID: " << current_pid << std::endl;
    std::cout << "CPU 使用率: " << cpu * 100 << "%" << std::endl;
    std::cout << "内存占用: " << mem << " MB" << std::endl;
}