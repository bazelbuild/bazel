#include <stdio.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <time.h>

__declspec(dllexport) char *get_time() {
  time_t ltime;
  time(&ltime);
  return ctime(&ltime);
}

__declspec(dllexport) void say_hello(char *message) {
  printf("Hello from dll!\n%s", message);
}

#ifdef __cplusplus
}
#endif

