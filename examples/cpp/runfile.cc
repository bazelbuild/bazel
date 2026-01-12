#include <fstream>
#include <iostream>
#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"

using rules_cc::cc::runfiles::Runfiles;

inline constexpr std::string_view someFile = "examples/runfile.txt";

int main(int argc, char **argv) {
  std::string error;
  const auto runfiles = Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error);
  if (runfiles == nullptr) {
    std::cerr << "Failed to create Runfiles object" << error << "\n";
    return 1;
  }

  std::string root = BAZEL_CURRENT_REPOSITORY;
  if (root == "") {
    root = "_main";
  }
  const std::string realPathToSomeFile = runfiles->Rlocation(root + "/" + std::string{someFile});

  std::cout << "The content of the runfile is:" << std::endl << std::ifstream(realPathToSomeFile).rdbuf();

  return 0;
}
