#include <stdio.h>
#include <windows.h>
typedef char *(__cdecl *GET_TIME_PTR)();
typedef void(__cdecl *SAY_HELLO_PTR)(char *);

int main() {
  HINSTANCE hellolib;
  GET_TIME_PTR get_time;
  SAY_HELLO_PTR say_hello;

  bool success = FALSE;

  hellolib = LoadLibrary(TEXT("hellolib.dll"));

  if (hellolib != NULL) {
    get_time = (GET_TIME_PTR)GetProcAddress(hellolib, "get_time");
    say_hello = (SAY_HELLO_PTR)GetProcAddress(hellolib, "say_hello");

    if (NULL != get_time && NULL != say_hello) {
      success = TRUE;
      char *now = get_time();
      say_hello(now);
    }
    FreeLibrary(hellolib);
  }

  if (!success) printf("Failed to load dll and call functions\n");

  return 0;
}
