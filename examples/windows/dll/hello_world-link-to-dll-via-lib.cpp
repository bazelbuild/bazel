#include "examples/windows/dll/hello-library.h"

int main() {
  char *now = get_time();
  say_hello(now);
  return 0;
}
