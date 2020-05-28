// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// wrapped_clang.cc: Pass args to 'xcrun clang' and zip dsym files.
//
// wrapped_clang passes its args to clang, but also supports a separate set of
// invocations to generate dSYM files.  If "DSYM_HINT" flags are passed in, they
// are used to construct that separate set of invocations (instead of being
// passed to clang).
// The following "DSYM_HINT" flags control dsym generation.  If any one if these
// are passed in, then they all must be passed in.
// "DSYM_HINT_LINKED_BINARY": Workspace-relative path to binary output of the
//    link action generating the dsym file.
// "DSYM_HINT_DSYM_PATH": Workspace-relative path to dSYM dwarf file.
//
// Likewise, this wrapper also contains a workaround for a bug in ld that causes
// flaky builds when using Bitcode symbol maps. ld allows the
// -bitcode_symbol_map to be either a directory (into which the file will be
// written) or a file, but the return value of the call to ::stat is never
// checked so examining the S_ISDIR bit of the struct afterwards returns
// true/false randomly depending on what data happened to be in memory at the
// time it was called:
// https://github.com/michaelweiser/ld64/blob/9c3700b64ed03e2d55ba094176bf6a172bf2bc6b/src/ld/Options.cpp#L3261
// To address this, we prepend a special "BITCODE_TOUCH_SYMBOL_MAP=" flag to the
// symbol map filename and touch it before passing it along to clang, forcing
// the file to exist.

#include <libgen.h>
#include <spawn.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

extern char **environ;

namespace {

constexpr char kAddASTPathPrefix[] = "-Wl,-add_ast_path,";

// Returns the base name of the given filepath. For example, given
// /foo/bar/baz.txt, returns 'baz.txt'.
const char *Basename(const char *filepath) {
  const char *base = strrchr(filepath, '/');
  return base ? (base + 1) : filepath;
}

// Converts an array of string arguments to char *arguments.
// The first arg is reduced to its basename as per execve conventions.
// Note that the lifetime of the char* arguments in the returned array
// are controlled by the lifetime of the strings in args.
std::vector<const char *> ConvertToCArgs(const std::vector<std::string> &args) {
  std::vector<const char *> c_args;
  c_args.push_back(Basename(args[0].c_str()));
  for (int i = 1; i < args.size(); i++) {
    c_args.push_back(args[i].c_str());
  }
  c_args.push_back(nullptr);
  return c_args;
}

// Turn our current process into a new process. Avoids fork overhead.
// Never returns.
void ExecProcess(const std::vector<std::string> &args) {
  std::vector<const char *> exec_argv = ConvertToCArgs(args);
  execv(args[0].c_str(), const_cast<char **>(exec_argv.data()));
  std::cerr << "Error executing child process.'" <<  args[0] << "'. "
            << strerror(errno) << "\n";
  abort();
}

// Spawns a subprocess for given arguments args. The first argument is used
// for the executable path.
void RunSubProcess(const std::vector<std::string> &args) {
  std::vector<const char *> exec_argv = ConvertToCArgs(args);
  pid_t pid;
  int status = posix_spawn(&pid, args[0].c_str(), NULL, NULL,
                           const_cast<char **>(exec_argv.data()), environ);
  if (status == 0) {
    int wait_status;
    do {
      wait_status = waitpid(pid, &status, 0);
    } while ((wait_status == -1) && (errno == EINTR));
    if (wait_status < 0) {
      std::cerr << "Error waiting on child process '" <<  args[0] << "'. "
                << strerror(errno) << "\n";
      abort();
    }
    if (WEXITSTATUS(status) != 0) {
      std::cerr << "Error in child process '" <<  args[0] << "'. "
                << WEXITSTATUS(status) << "\n";
      abort();
    }
  } else {
    std::cerr << "Error forking process '" <<  args[0] << "'. "
              << strerror(status) << "\n";
    abort();
  }
}

// Finds and replaces all instances of oldsub with newsub, in-place on str.
void FindAndReplace(const std::string &oldsub, const std::string &newsub,
                    std::string *str) {
  int start = 0;
  while ((start = str->find(oldsub, start)) != std::string::npos) {
    str->replace(start, oldsub.length(), newsub);
    start += newsub.length();
  }
}

// If arg is of the classic flag form "foo=bar", and flagname is 'foo', sets
// str to point to a new std::string 'bar' and returns true.
// Otherwise, returns false.
bool SetArgIfFlagPresent(const std::string &arg, const std::string &flagname,
                         std::string *str) {
  std::string prefix_string = flagname + "=";
  if (arg.compare(0, prefix_string.length(), prefix_string) == 0) {
    *str = arg.substr(prefix_string.length());
    return true;
  }
  return false;
}

// Returns the DEVELOPER_DIR environment variable in the current process
// environment. Aborts if this variable is unset.
std::string GetMandatoryEnvVar(const std::string &var_name) {
  char *env_value = getenv(var_name.c_str());
  if (env_value == nullptr) {
    std::cerr << "Error: " << var_name << " not set.\n";
    abort();
  }
  return env_value;
}

// Returns true if `str` starts with the specified `prefix`.
bool StartsWith(const std::string &str, const std::string &prefix) {
  return str.compare(0, prefix.size(), prefix) == 0;
}

// If *`str` begins `prefix`, strip it out and return true.
// Otherwise leave *`str` unchanged and return false.
bool StripPrefixStringIfPresent(std::string *str, const std::string &prefix) {
  if (StartsWith(*str, prefix)) {
    *str = str->substr(prefix.size());
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string tool_name;

  std::string binary_name = Basename(argv[0]);
  if (binary_name == "wrapped_clang_pp") {
    tool_name = "clang++";
  } else if (binary_name == "wrapped_clang") {
    tool_name = "clang";
  } else {
    std::cerr << "Binary must either be named 'wrapped_clang' or "
                 "'wrapped_clang_pp', not "
              << binary_name << "\n";
    abort();
  }

  std::string developer_dir = GetMandatoryEnvVar("DEVELOPER_DIR");
  std::string sdk_root = GetMandatoryEnvVar("SDKROOT");

  std::vector<std::string> processed_args = {"/usr/bin/xcrun", tool_name};

  std::string linked_binary, dsym_path, bitcode_symbol_map;
  std::string dest_dir;

  std::unique_ptr<char, decltype(std::free) *> cwd{getcwd(nullptr, 0),
                                                   std::free};
  if (cwd == nullptr) {
    std::cerr << "Error determining current working directory\n";
    abort();
  }

  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (SetArgIfFlagPresent(arg, "DSYM_HINT_LINKED_BINARY", &linked_binary)) {
      continue;
    }
    if (SetArgIfFlagPresent(arg, "DSYM_HINT_DSYM_PATH", &dsym_path)) {
      continue;
    }
    if (SetArgIfFlagPresent(arg, "BITCODE_TOUCH_SYMBOL_MAP",
                            &bitcode_symbol_map)) {
      // Touch bitcode_symbol_map.
      std::ofstream bitcode_symbol_map_file(bitcode_symbol_map);
      arg = bitcode_symbol_map;
    }
    if (SetArgIfFlagPresent(arg, "DEBUG_PREFIX_MAP_PWD", &dest_dir)) {
      arg = "-fdebug-prefix-map=" + std::string(cwd.get()) + "=" + dest_dir;
    }
    FindAndReplace("__BAZEL_XCODE_DEVELOPER_DIR__", developer_dir, &arg);
    FindAndReplace("__BAZEL_XCODE_SDKROOT__", sdk_root, &arg);

    // Make the `add_ast_path` options used to embed Swift module references
    // absolute to enable Swift debugging without dSYMs: see
    // https://forums.swift.org/t/improving-swift-lldb-support-for-path-remappings/22694
    if (StripPrefixStringIfPresent(&arg, kAddASTPathPrefix)) {
      // Only modify relative paths.
      if (!StartsWith(arg, "/")) {
        arg = std::string(kAddASTPathPrefix) +
              std::string(cwd.get()) + "/" + arg;
      } else {
        arg = std::string(kAddASTPathPrefix) + arg;
      }
    }

    processed_args.push_back(arg);
  }

  // Special mode that only prints the command. Used for testing.
  if (getenv("__WRAPPED_CLANG_LOG_ONLY")) {
    for (const std::string &arg : processed_args)
        std::cout << arg << ' ';
    std::cout << "\n";
    return 0;
  }

  // Check to see if we should postprocess with dsymutil.
  bool postprocess = false;
  if ((!linked_binary.empty()) || (!dsym_path.empty())) {
    if ((linked_binary.empty()) || (dsym_path.empty())) {
      const char *missing_dsym_flag;
      if (linked_binary.empty()) {
        missing_dsym_flag = "DSYM_HINT_LINKED_BINARY";
      } else {
        missing_dsym_flag = "DSYM_HINT_DSYM_PATH";
      }
      std::cerr << "Error in clang wrapper: If any dsym "
              "hint is defined, then "
           << missing_dsym_flag << " must be defined\n";
      abort();
    } else {
      postprocess = true;
    }
  }

  if (!postprocess) {
    ExecProcess(processed_args);
    std::cerr << "ExecProcess should not return. Please fix!\n";
    abort();
  }

  RunSubProcess(processed_args);

  std::vector<std::string> dsymutil_args = {"/usr/bin/xcrun", "dsymutil",
                                            linked_binary, "-o", dsym_path,
                                            "--flat"};
  ExecProcess(dsymutil_args);
  std::cerr << "ExecProcess should not return. Please fix!\n";
  abort();

  return 0;
}
