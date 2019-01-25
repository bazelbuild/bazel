#include <stdio.h>
#include <windows.h>
#include "hello-library.h"

int main() {
  char *now = get_time();
  say_hello(now);
  return 0;
}
