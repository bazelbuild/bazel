// Copyright 2014 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_MAIN_CPP_BLAZE_STARTUP_OPTIONS_H_
#define BAZEL_SRC_MAIN_CPP_BLAZE_STARTUP_OPTIONS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/main/cpp/util/exit_code.h"

namespace blaze {

using std::string;

struct StartupOptions;

// This class holds the parsed startup options for Blaze.
// These options and their defaults must be kept in sync with those
// in java/com/google/devtools/build/lib/blaze/BlazeServerStartupOptions.
// The latter are purely decorative (they affect the help message,
// which displays the defaults).  The actual defaults are defined
// in the constructor.
//
// TODO(bazel-team): The encapsulation is not quite right -- there are some
// places in blaze.cc where some of these fields are explicitly modified. Their
// names also don't conform to the style guide.
class BlazeStartupOptions {
 public:
  BlazeStartupOptions();
  BlazeStartupOptions(const BlazeStartupOptions &rhs);
  ~BlazeStartupOptions();
  BlazeStartupOptions& operator=(const BlazeStartupOptions &rhs);

  // Returns the capitalized name of this binary.
  string GetProductName();

  // Parses a single argument, either from the command line or from the .blazerc
  // "startup" options.
  //
  // rcfile should be an empty string if the option being parsed does not come
  // from a blazerc.
  //
  // Sets "is_space_separated" true if arg is unary and uses the "--foo bar"
  // style, so its value is in next_arg.
  //
  // Sets "is_space_separated" false if arg is either nullary
  // (e.g. "--[no]batch") or is unary but uses the "--foo=bar" style.
  //
  // Returns the exit code after processing the argument. "error" will contain
  // a descriptive string for any return value other than
  // blaze_exit_code::SUCCESS.
  blaze_exit_code::ExitCode ProcessArg(
      const string &arg, const string &next_arg, const string &rcfile,
      bool *is_space_separated, string *error);

  // Adds any other options needed to result.
  void AddExtraOptions(std::vector<string> *result) const;

  // Checks if Blaze needs to be re-executed.  Does not return, if so.
  //
  // Returns the exit code after the check. "error" will contain a descriptive
  // string for any return value other than blaze_exit_code::SUCCESS.
  blaze_exit_code::ExitCode CheckForReExecuteOptions(
      int argc, const char *argv[], string *error);

  // Checks extra fields when processing arg.
  //
  // Returns the exit code after processing the argument. "error" will contain
  // a descriptive string for any return value other than
  // blaze_exit_code::SUCCESS.
  blaze_exit_code::ExitCode ProcessArgExtra(
    const char *arg, const char *next_arg, const string &rcfile,
    const char **value, bool *is_processed, string *error);

  // Return the default path to the JDK used to run Blaze itself
  // (must be an absolute directory).
  string GetDefaultHostJavabase() const;

  // Returns the path to the JVM. This should be called after parsing
  // the startup options.
  string GetJvm();

  // Adds JVM tuning flags for Blaze.
  //
  // Returns the exit code after this operation. "error" will be set to a
  // descriptive string for any value other than blaze_exit_code::SUCCESS.
  blaze_exit_code::ExitCode AddJVMArguments(
    const string &host_javabase, std::vector<string> *result,
    const std::vector<string> &user_options, string *error) const;

  // Blaze's output base.  Everything is relative to this.  See
  // the BlazeDirectories Java class for details.
  string output_base;

  // Installation base for a specific release installation.
  string install_base;

  // The toplevel directory containing Blaze's output.  When Blaze is
  // run by a test, we use TEST_TMPDIR, simplifying the correct
  // hermetic invocation of Blaze from tests.
  string output_root;

  // Blaze's output_user_root. Used only for computing install_base and
  // output_base.
  string output_user_root;

  // Whether to put the execroot at $OUTPUT_BASE/$WORKSPACE_NAME (if false) or
  // $OUTPUT_BASE/execroot/$WORKSPACE_NAME (if true).
  bool deep_execroot;

  // Block for the Blaze server lock. Otherwise,
  // quit with non-0 exit code if lock can't
  // be acquired immediately.
  bool block_for_lock;

  bool host_jvm_debug;

  string host_jvm_profile;

  std::vector<string> host_jvm_args;

  bool batch;

  // From the man page: "This policy is useful for workloads that are
  // non-interactive, but do not want to lower their nice value, and for
  // workloads that want a deterministic scheduling policy without
  // interactivity causing extra preemptions (between the workload's tasks)."
  bool batch_cpu_scheduling;

  // If negative, don't mess with ionice. Otherwise, set a level from 0-7
  // for best-effort scheduling. 0 is highest priority, 7 is lowest.
  int io_nice_level;

  int max_idle_secs;

  bool oom_more_eagerly;

  // If true, Blaze will listen to OS-level file change notifications.
  bool watchfs;

  // Temporary experimental flag that permits configurable attribute syntax
  // in BUILD files. This will be removed when configurable attributes is
  // a more stable feature.
  bool allow_configurable_attributes;

  // Temporary flag for enabling EventBus exceptions to be fatal.
  bool fatal_event_bus_exceptions;

  // A string to string map specifying where each option comes from. If the
  // value is empty, it was on the command line, if it is a string, it comes
  // from a blazerc file, if a key is not present, it is the default.
  std::map<string, string> option_sources;

  // This can be used for site-specific startup options. For Bazel, this is
  // stubbed
  // out.
  std::unique_ptr<StartupOptions> extra_options;

  // Given the working directory, returns the nearest enclosing directory with a
  // WORKSPACE file in it.  If there is no such enclosing directory, returns "".
  //
  // E.g., if there was a WORKSPACE file in foo/bar/build_root:
  // GetWorkspace('foo/bar') --> ''
  // GetWorkspace('foo/bar/build_root') --> 'foo/bar/build_root'
  // GetWorkspace('foo/bar/build_root/biz') --> 'foo/bar/build_root'
  //
  // The returned path is relative or absolute depending on whether cwd was
  // relative or absolute.
  static string GetWorkspace(const string &cwd);

  // Returns if workspace is a valid build workspace.
  static bool InWorkspace(const string &workspace);

  // Returns the basename for the rc file.
  static string RcBasename();

  // Returns the path for the system-wide rc file.
  static string SystemWideRcPath();

  // Returns the candidate pathnames for the RC file in the workspace,
  // the first readable one of which will be chosen.
  // It is ok if no usable candidate exists.
  static void WorkspaceRcFileSearchPath(std::vector<string>* candidates);

  // Turn a %workspace%-relative import into its true name in the filesystem.
  // path_fragment is modified in place.
  // Unlike WorkspaceRcFileSearchPath, it is an error if no import file exists.
  static bool WorkspaceRelativizeRcFilePath(const string &workspace,
                                            string *path_fragment);

  static constexpr char WorkspacePrefix[] = "%workspace%/";
  static const int WorkspacePrefixLength = sizeof WorkspacePrefix - 1;

  // Returns the GetHostJavabase. This should be called after parsing
  // the --host_javabase option.
  string GetHostJavabase();

  // Port for web status server, 0 to disable
  int webstatus_port;

  // Invocation policy proto. May be NULL.
  const char* invocation_policy;

 private:
  string host_javabase;

  // Sets default values for members.
  void Init();

  // Copies member variables from rhs to lhs. This cannot use the compiler-
  // generated copy constructor because extra_options is a unique_ptr and
  // unique_ptr deletes its copy constructor.
  void Copy(const BlazeStartupOptions &rhs, BlazeStartupOptions *lhs);

  // Returns the directory to use for storing outputs.
  string GetOutputRoot();
};

}  // namespace blaze
#endif  // BAZEL_SRC_MAIN_CPP_BLAZE_STARTUP_OPTIONS_H_
