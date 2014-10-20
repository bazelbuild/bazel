#include "examples/cpp/hello-lib.h"

int main(int argc, char** argv) {
  const char* obj = "world";
  if (argc > 1) {
    obj = argv[1];
  }

  greet(obj);
  return 0;
}
