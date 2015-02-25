#include "examples/cpp/hello-lib.h"

int main(int argc, char** argv) {
  const char* obj = "barf";
  if (argc > 1) {
    obj = argv[1];
  }

  greet(obj);
  return 1;
}
