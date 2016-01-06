#include "hello-lib.h"

#include <string>

using hello::HelloLib;
using std::string;

/**
 * This prints "Hello world". If it is run with arguments, it will use the first
 * argument instead of "world". Build and run //examples/cpp:hello-world to see
 * this program in action.
 *
 * This file does double-duty as a "test." It is a cc_binary, so it can also be
 * compiled as a cc_test and Bazel will decide on whether it passed or failed
 * based on exit code (which is always 0 here, so the test "passes"). See
 * hello-fail.cc for an example of making a test fail.
 */
int main(int argc, char** argv) {
  HelloLib lib("Hello");
  string thing = "world";
  if (argc > 1) {
    thing = argv[1];
  }
  lib.greet(thing);
  return 0;
}
