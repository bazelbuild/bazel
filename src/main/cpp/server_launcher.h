// Copyright 2026 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_MAIN_CPP_SERVER_LAUNCHER_H_
#define BAZEL_SRC_MAIN_CPP_SERVER_LAUNCHER_H_

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/option_processor.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/workspace_layout.h"

namespace blaze {

struct ClientLocks {
  std::optional<LockHandle> install_base_lock;
  std::optional<LockHandle> output_base_lock;
};

// Acquires the install base (if configured) and output base client locks.
std::pair<ClientLocks, DurationMillis> AcquireClientLocks(
    const StartupOptions& startup_options);

// Releases the client locks acquired by AcquireClientLocks().
void ReleaseClientLocks(bool lock_install_base, ClientLocks* locks);

// Returns the JVM command argument array.
std::vector<std::string> GetServerExeArgs(
    const blaze_util::Path& jvm_path, const std::string& server_jar_path,
    const std::string& install_md5, const WorkspaceLayout& workspace_layout,
    const std::string& workspace, const StartupOptions& startup_options);

// Join the elements of the specified array with NUL's (\0's), akin to the
// format of /proc/$PID/cmdline.
std::string GetArgumentString(const std::vector<std::string>& argument_array);

void EnsureServerDir(const blaze_util::Path& server_dir);

// Do a chdir into the workspace, and die if it fails.
void GoToWorkspace(const WorkspaceLayout& workspace_layout,
                   const std::string& workspace);

bool IsServerMode(const std::string& command);

// Returns the path of the file that the server JVM reads its arguments from on
// Windows.
blaze_util::Path GetServerJvmArgFilePath(const blaze_util::Path& output_base);

// Returns the server's PID from the pid file, or -1 if there was an error.
int GetServerPid(const blaze_util::Path& server_dir);

bool NeedsNewInstallationSymlink(const StartupOptions& startup_options);

void UpdateInstallationSymlink(const StartupOptions& startup_options);

std::string GetCanonicalCwd();

void PrepareDirectories(StartupOptions* startup_options);

// Prepares the environment to be suitable to start a JVM.
// Changes made to the environment in this function *will not* be part
// of '--client_env'.
std::map<std::string, EnvVarValue> PrepareEnvironmentForJvm();

// Replace this process with the blaze server. Does not exit.
void RunServerMode(const std::string& install_md5,
                   const WorkspaceLayout& workspace_layout,
                   const std::string& workspace,
                   const StartupOptions& startup_options);

// Sets up resource limits, parses startup/command options, configures logging,
// and validates workspace requirements.
StartupOptions* SetupAndParseOptions(int argc, const char* const* argv,
                                     const std::string& cwd,
                                     const std::string& workspace,
                                     WorkspaceLayout* workspace_layout,
                                     OptionProcessor* option_processor);

// Entry point for a standalone exec-server binary. Requires --install_base to
// already be extracted on disk.
int ExecServer(int argc, const char* const* argv,
               WorkspaceLayout* workspace_layout,
               OptionProcessor* option_processor);

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_SERVER_LAUNCHER_H_
