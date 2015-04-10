#include "examples/cpp/hello-lib.h"

#include <string>

using hello::HelloLib;
using std::string;

/**
 * This is a fake test that prints "Hello barf" and then exits with exit code 1.
 * If run as a test (cc_test), the non-0 exit code indicates to Bazel that the
 * "test" has failed.
 */
int main(int argc, char** argv) {
  HelloLib lib("Hello");
  string thing = "barf";
  if (argc > 1) {
    thing = argv[1];
  }

  lib.greet(thing);
  return 1;
}
