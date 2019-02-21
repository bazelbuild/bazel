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
//
// blaze.cc: bootstrap and client code for Blaze server.
//
// Responsible for:
// - extracting the Python, C++ and Java components.
// - starting the server or finding the existing one.
// - client options parsing.
// - passing the argv array, and printing the out/err streams.
// - signal handling.
// - exiting with the right error/WTERMSIG code.
// - debugger + profiler support.
// - mutual exclusion between batch invocations.
#include "src/main/cpp/blaze.h"

#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grpc/grpc.h>
#include <grpc/support/log.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <algorithm>
#include <chrono>  // NOLINT (gRPC requires this)
#include <cinttypes>
#include <map>
#include <mutex>  // NOLINT
#include <set>
#include <sstream>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/global_variables.h"
#include "src/main/cpp/option_processor.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/bazel_log_handler.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"
#include "third_party/ijar/zip.h"

#include "src/main/protobuf/command_server.grpc.pb.h"

using blaze_util::GetLastErrorString;

extern char** environ;

namespace blaze {

using std::map;
using std::set;
using std::string;
using std::vector;
using command_server::CommandServer;

// The following is a treatise on how the interaction between the client and the
// server works.
//
// First, the client unconditionally acquires an flock() lock on
// $OUTPUT_BASE/lock then verifies if it has already extracted itself by
// checking if the directory it extracts itself to (install base + a checksum)
// is present. If not, then it does the extraction. Care is taken that this
// process is atomic so that Blazen in multiple output bases do not clash.
//
// Then the client tries to connect to the currently executing server and kills
// it if at least one of the following conditions is true:
//
// - The server is of the wrong version (as determined by the
//   $OUTPUT_BASE/install symlink)
// - The server has different startup options than the client wants
// - The client wants to run the command in batch mode
//
// Then, if needed, the client adjusts the install link to indicate which
// version of the server it is running.
//
// In batch mode, the client then simply executes the server while taking care
// that the output base lock is kept until it finishes.
//
// If in server mode, the client starts up a server if needed then sends the
// command to the client and streams back stdout and stderr. The output base
// lock is released after the command is sent to the server (the server
// implements its own locking mechanism).

// Synchronization between the client and the server is a little precarious
// because the client needs to know the PID of the server and it is not
// available using a Java API and we don't have JNI on Windows at the moment,
// so the server can't just communicate this over the communication channel.
// Thus, a PID file is used, but care needs to be taken that the contents of
// this PID file are right.
//
// Upon server startup, the PID file is written before the client spawns the
// server. Thus, when the client can connect, it can be certain that the PID
// file is up to date.
//
// Upon server shutdown, the PID file is deleted using a server shutdown hook.
// However, this happens *after* the server stopped listening, so it's possible
// that a client has already started up a server and written a new PID file.
// In order to avoid this, when the client starts up a new server, it reads the
// contents of the PID file and kills the process indicated in it (it could do
// with a bit more care, since PIDs can be reused, but for now, we just believe
// the PID file)
//
// Some more interesting scenarios:
//
// - The server receives a kill signal and it does not have a chance to delete
//   the PID file: the client cannot connect, reads the PID file, kills the
//   process indicated in it and starts up a new server.
//
// - The server stopped accepting connections but hasn't quit yet and a new
//   client comes around: the new client will kill the server based on the
//   PID file before a new server is started up.
//
// Alternative implementations:
//
// - Don't deal with PIDs at all. This would make it impossible for the client
//   to deliver a SIGKILL to the server after three SIGINTs. It would only be
//   possible with gRPC anyway.
//
// - Have the server check that the PID file contains the correct things
//   before deleting them: there is a window of time between checking the file
//   and deleting it in which a new server can overwrite the PID file. The
//   output base lock cannot be acquired, either, because when starting up a
//   new server, the client already holds it.
//
// - Delete the PID file before stopping to accept connections: then a client
//   could come about after deleting the PID file but before stopping accepting
//   connections. It would also not be resilient against a dead server that
//   left a PID file around.
class BlazeServer {
 public:
  virtual ~BlazeServer() {}

  // Acquire a lock for the server running in this output base. Returns the
  // number of milliseconds spent waiting for the lock.
  uint64_t AcquireLock();

  // Whether there is an active connection to a server.
  bool Connected() const { return connected_; }

  // Connect to the server. Returns if the connection was successful. Only
  // call this when this object is in disconnected state. If it returns true,
  // this object will be in connected state.
  virtual bool Connect() = 0;

  // Disconnects from an existing server. Only call this when this object is in
  // connected state. After this call returns, the object will be in connected
  // state.
  virtual void Disconnect() = 0;

  // Send the command line to the server and forward whatever it says to stdout
  // and stderr. Returns the desired exit code. Only call this when the server
  // is in connected state.
  virtual unsigned int Communicate() = 0;

  // Disconnects and kills an existing server. Only call this when this object
  // is in connected state.
  virtual void KillRunningServer() = 0;

  // Cancel the currently running command. If there is no command currently
  // running, the result is unspecified. When called, this object must be in
  // connected state.
  virtual void Cancel() = 0;

 protected:
  BlazeLock blaze_lock_;
  bool connected_;
};

////////////////////////////////////////////////////////////////////////
// Global Variables
static GlobalVariables *globals;
static BlazeServer *blaze_server;

// TODO(laszlocsomor) 2016-11-24: release the `globals` and `blaze_server`
// objects. Currently nothing deletes them. Be careful that some functions may
// call exit(2) or _exit(2) (attributed with ATTRIBUTE_NORETURN) meaning we have
// to delete the objects before those.

uint64_t BlazeServer::AcquireLock() {
  return blaze::AcquireLock(globals->options->output_base,
                            globals->options->batch,
                            globals->options->block_for_lock, &blaze_lock_);
}

// Communication method that uses gRPC on a socket bound to localhost. More
// documentation is in command_server.proto .
class GrpcBlazeServer : public BlazeServer {
 public:
  GrpcBlazeServer(int connect_timeout_secs);
  virtual ~GrpcBlazeServer();

  virtual bool Connect();
  virtual void Disconnect();
  virtual unsigned int Communicate();
  virtual void KillRunningServer();
  virtual void Cancel();

 private:
  enum CancelThreadAction { NOTHING, JOIN, CANCEL, COMMAND_ID_RECEIVED };

  std::unique_ptr<CommandServer::Stub> client_;
  std::string request_cookie_;
  std::string response_cookie_;
  std::string command_id_;

  // protects command_id_ . Although we always set it before making the cancel
  // thread do something with it, the mutex is still useful because it provides
  // a memory fence.
  std::mutex cancel_thread_mutex_;

  int connect_timeout_secs_;

  // Pipe that the main thread sends actions to and the cancel thread receives
  // actions from.
  blaze_util::IPipe *pipe_;

  bool TryConnect(CommandServer::Stub *client);
  void CancelThread();
  void SendAction(CancelThreadAction action);
  void SendCancelMessage();
};

////////////////////////////////////////////////////////////////////////
// Logic

// A devtools_ijar::ZipExtractorProcessor that has a pure version of Accept.
class PureZipExtractorProcessor : public devtools_ijar::ZipExtractorProcessor {
 public:
  virtual ~PureZipExtractorProcessor() {}

  // Like devtools_ijar::ZipExtractorProcessor::Accept, but is guaranteed to not
  // have side-effects.
  virtual bool AcceptPure(const char *filename,
                          const devtools_ijar::u4 attr) const = 0;
};

// A PureZipExtractorProcessor that adds the names of all the files ZIP up in
// the Blaze binary to the given vector.
class NoteAllFilesZipProcessor : public PureZipExtractorProcessor {
 public:
  explicit NoteAllFilesZipProcessor(std::vector<std::string>* files)
      : files_(files) {}

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override {
    return false;
  }

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    files_->push_back(filename);
    return false;
  }

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "NoteAllFilesZipProcessor::Process shouldn't be called";
  }
 private:
  std::vector<std::string>* files_;
};

// A devtools_ijar::ZipExtractorProcessor that processes the ZIP entries using
// the given PureZipExtractorProcessors.
class CompoundZipProcessor : public devtools_ijar::ZipExtractorProcessor {
 public:
  explicit CompoundZipProcessor(
      const vector<PureZipExtractorProcessor*>& processors)
      : processors_(processors) {}

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    bool should_accept = false;
    for (auto* processor : processors_) {
      if (processor->Accept(filename, attr)) {
        // ZipExtractorProcessor::Accept is allowed to be side-effectful, so
        // we don't want to break out on the first true here.
        should_accept = true;
      }
    }
    return should_accept;
  }

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override {
    for (auto* processor : processors_) {
      if (processor->AcceptPure(filename, attr)) {
        processor->Process(filename, attr, data, size);
      }
    }
  }

 private:
  const vector<PureZipExtractorProcessor*> processors_;
};

static map<string, EnvVarValue> PrepareEnvironmentForJvm();

// A PureZipExtractorProcessor to extract the InstallKeyFile
class GetInstallKeyFileProcessor : public PureZipExtractorProcessor {
 public:
  explicit GetInstallKeyFileProcessor(string *install_base_key)
      : install_base_key_(install_base_key) {}

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override {
    return strcmp(filename, "install_base_key") == 0;
  }

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    return AcceptPure(filename, attr);
  }

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override {
    string str(reinterpret_cast<const char *>(data), size);
    blaze_util::StripWhitespace(&str);
    if (str.size() != 32) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "Failed to extract install_base_key: file size mismatch "
             "(should be 32, is "
          << str.size() << ")";
    }
    *install_base_key_ = str;
  }

 private:
  string *install_base_key_;
};

// Populates globals->install_md5 and globals->extracted_binaries by reading the
// ZIP entries in the Blaze binary.
static void ComputeInstallMd5AndNoteAllFiles(const string &self_path) {
  NoteAllFilesZipProcessor note_all_files_processor(
      &globals->extracted_binaries);
  GetInstallKeyFileProcessor install_key_processor(&globals->install_md5);
  CompoundZipProcessor processor({&note_all_files_processor,
                                  &install_key_processor});
  std::unique_ptr<devtools_ijar::ZipExtractor> extractor(
      devtools_ijar::ZipExtractor::Create(self_path.c_str(), &processor));
  if (extractor == NULL) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to open " << globals->options->product_name
        << " as a zip file: " << GetLastErrorString();
  }
  if (extractor->ProcessAll() < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to extract install_base_key: " << extractor->GetError();
  }

  if (globals->install_md5.empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to find install_base_key's in zip file";
  }
}

// Escapes colons by replacing them with '_C' and underscores by replacing them
// with '_U'. E.g. "name:foo_bar" becomes "name_Cfoo_Ubar"
static string EscapeForOptionSource(const string &input) {
  string result = input;
  blaze_util::Replace("_", "_U", &result);
  blaze_util::Replace(":", "_C", &result);
  return result;
}

// Returns the installed embedded binaries directory, under the shared
// install_base location.
string GetEmbeddedBinariesRoot(const string &install_base) {
  return blaze_util::JoinPath(install_base, "_embedded_binaries");
}

// Returns the JVM command argument array.
static vector<string> GetArgumentArray(
    const WorkspaceLayout *workspace_layout) {
  vector<string> result;

  // e.g. A Blaze server process running in ~/src/build_root (where there's a
  // ~/src/build_root/WORKSPACE file) will appear in ps(1) as "blaze(src)".
  string workspace =
      workspace_layout->GetPrettyWorkspaceName(globals->workspace);
  string product = globals->options->product_name;
  blaze_util::ToLower(&product);
  result.push_back(product + "(" + workspace + ")");
  globals->options->AddJVMArgumentPrefix(
      blaze_util::Dirname(blaze_util::Dirname(globals->jvm_path)), &result);

  result.push_back("-XX:+HeapDumpOnOutOfMemoryError");
  string heap_crash_path = globals->options->output_base;
  result.push_back("-XX:HeapDumpPath=" +
                   blaze_util::PathAsJvmFlag(heap_crash_path));

  // TODO(b/109998449): only assume JDK >= 9 for embedded JDKs
  if (!globals->options->GetEmbeddedJavabase().empty()) {
    // In JDK9 we have seen a slow down when using the default G1 collector
    // and thus switch back to parallel gc.
    result.push_back("-XX:+UseParallelOldGC");
    // see: https://github.com/google/protobuf/issues/3781

    // quiet warnings from com.google.protobuf.UnsafeUtil,
    // see: https://github.com/google/protobuf/issues/3781
    result.push_back("--add-opens=java.base/java.nio=ALL-UNNAMED");
    result.push_back("--add-opens=java.base/java.lang=ALL-UNNAMED");
  }

  result.push_back("-Xverify:none");

  vector<string> user_options;

  user_options.insert(user_options.begin(),
                      globals->options->host_jvm_args.begin(),
                      globals->options->host_jvm_args.end());

  // Add JVM arguments particular to building blaze64 and particular JVM
  // versions.
  string error;
  blaze_exit_code::ExitCode jvm_args_exit_code =
      globals->options->AddJVMArguments(globals->options->GetServerJavabase(),
                                        &result, user_options, &error);
  if (jvm_args_exit_code != blaze_exit_code::SUCCESS) {
    BAZEL_DIE(jvm_args_exit_code) << error;
  }

  // We put all directories on java.library.path that contain .so/.dll files.
  set<string> java_library_paths;
  std::stringstream java_library_path;
  java_library_path << "-Djava.library.path=";
  string real_install_dir =
      GetEmbeddedBinariesRoot(globals->options->install_base);

  bool first = true;
  for (const auto &it : globals->extracted_binaries) {
    if (IsSharedLibrary(it)) {
      string libpath(blaze_util::PathAsJvmFlag(
          blaze_util::JoinPath(real_install_dir, blaze_util::Dirname(it))));
      // Only add the library path if it's not added yet.
      if (java_library_paths.find(libpath) == java_library_paths.end()) {
        java_library_paths.insert(libpath);
        if (!first) {
          java_library_path << kListSeparator;
        }
        first = false;
        java_library_path << libpath;
      }
    }
  }
  result.push_back(java_library_path.str());

  // Force use of latin1 for file names.
  result.push_back("-Dfile.encoding=ISO-8859-1");

  if (globals->options->host_jvm_debug) {
    BAZEL_LOG(USER)
        << "Running host JVM under debugger (listening on TCP port 5005).";
    // Start JVM so that it listens for a connection from a
    // JDWP-compliant debugger:
    result.push_back("-Xdebug");
    result.push_back("-Xrunjdwp:transport=dt_socket,server=y,address=5005");
  }
  result.insert(result.end(), user_options.begin(), user_options.end());

  globals->options->AddJVMArgumentSuffix(real_install_dir,
                                         globals->ServerJarPath(), &result);

  // JVM arguments are complete. Now pass in Blaze startup options.
  // Note that we always use the --flag=ARG form (instead of the --flag ARG one)
  // so that BlazeRuntime#splitStartupOptions has an easy job.

  // TODO(lberki): Test that whatever the list constructed after this line is
  // actually a list of parseable startup options.
  if (!globals->options->batch) {
    result.push_back("--max_idle_secs=" +
                     ToString(globals->options->max_idle_secs));
    result.push_back("--shutdown_on_low_sys_mem=" +
                     ToString(globals->options->shutdown_on_low_sys_mem));
  } else {
    // --batch must come first in the arguments to Java main() because
    // the code expects it to be at args[0] if it's been set.
    result.push_back("--batch");
  }

  if (globals->options->command_port != 0) {
    result.push_back("--command_port=" +
                     ToString(globals->options->command_port));
  }

  result.push_back("--connect_timeout_secs=" +
                   ToString(globals->options->connect_timeout_secs));

  result.push_back("--output_user_root=" +
                   blaze_util::ConvertPath(globals->options->output_user_root));
  result.push_back("--install_base=" +
                   blaze_util::ConvertPath(globals->options->install_base));
  result.push_back("--install_md5=" + globals->install_md5);
  result.push_back("--output_base=" +
                   blaze_util::ConvertPath(globals->options->output_base));
  result.push_back("--workspace_directory=" +
                   blaze_util::ConvertPath(globals->workspace));
  result.push_back("--default_system_javabase=" + GetSystemJavabase());

  if (!globals->options->server_jvm_out.empty()) {
    result.push_back("--server_jvm_out=" + globals->options->server_jvm_out);
  }

  if (globals->options->deep_execroot) {
    result.push_back("--deep_execroot");
  } else {
    result.push_back("--nodeep_execroot");
  }
  if (globals->options->expand_configs_in_place) {
    result.push_back("--expand_configs_in_place");
  } else {
    result.push_back("--noexpand_configs_in_place");
  }
  if (!globals->options->digest_function.empty()) {
    // Only include this if a value is requested - we rely on the empty case
    // being "null" to set the programmatic default in the server.
    result.push_back("--digest_function=" + globals->options->digest_function);
  }
  if (globals->options->idle_server_tasks) {
    result.push_back("--idle_server_tasks");
  } else {
    result.push_back("--noidle_server_tasks");
  }
  if (globals->options->oom_more_eagerly) {
    result.push_back("--experimental_oom_more_eagerly");
  } else {
    result.push_back("--noexperimental_oom_more_eagerly");
  }
  result.push_back("--experimental_oom_more_eagerly_threshold=" +
                   ToString(globals->options->oom_more_eagerly_threshold));

  if (globals->options->write_command_log) {
    result.push_back("--write_command_log");
  } else {
    result.push_back("--nowrite_command_log");
  }

  if (globals->options->watchfs) {
    result.push_back("--watchfs");
  } else {
    result.push_back("--nowatchfs");
  }
  if (globals->options->fatal_event_bus_exceptions) {
    result.push_back("--fatal_event_bus_exceptions");
  } else {
    result.push_back("--nofatal_event_bus_exceptions");
  }

  // We use this syntax so that the logic in AreStartupOptionsDifferent() that
  // decides whether the server needs killing is simpler. This is parsed by the
  // Java code where --noclient_debug and --client_debug=false are equivalent.
  // Note that --client_debug false (separated by space) won't work either,
  // because the logic in AreStartupOptionsDifferent() assumes that every
  // argument is in the --arg=value form.
  if (globals->options->client_debug) {
    result.push_back("--client_debug=true");
  } else {
    result.push_back("--client_debug=false");
  }

  // These flags are passed to the java process only for Blaze reporting
  // purposes; the real interpretation of the jvm flags occurs when we set up
  // the java command line.
  if (!globals->options->GetExplicitServerJavabase().empty()) {
    result.push_back("--server_javabase=" +
                     globals->options->GetExplicitServerJavabase());
  }
  if (globals->options->host_jvm_debug) {
    result.push_back("--host_jvm_debug");
  }
  if (!globals->options->host_jvm_profile.empty()) {
    result.push_back("--host_jvm_profile=" +
                     globals->options->host_jvm_profile);
  }
  if (!globals->options->host_jvm_args.empty()) {
    for (const auto &arg : globals->options->host_jvm_args) {
      result.push_back("--host_jvm_args=" + arg);
    }
  }

  // Pass in invocation policy as a startup argument for batch mode only.
  if (globals->options->batch && !globals->options->invocation_policy.empty()) {
    result.push_back("--invocation_policy=" +
                     globals->options->invocation_policy);
  }

  result.push_back("--product_name=" + globals->options->product_name);

  globals->options->AddExtraOptions(&result);

  // The option sources are transmitted in the following format:
  // --option_sources=option1:source1:option2:source2:...
  string option_sources = "--option_sources=";
  first = true;
  for (const auto &it : globals->options->option_sources) {
    if (!first) {
      option_sources += ":";
    }

    first = false;
    option_sources += EscapeForOptionSource(it.first) + ":" +
                      EscapeForOptionSource(it.second);
  }

  result.push_back(option_sources);
  return result;
}

// Add common command options for logging to the given argument array.
static void AddLoggingArgs(vector<string> *args) {
  args->push_back("--startup_time=" + ToString(globals->startup_time));
  if (globals->command_wait_time != 0) {
    args->push_back("--command_wait_time=" +
                    ToString(globals->command_wait_time));
  }
  if (globals->extract_data_time != 0) {
    args->push_back("--extract_data_time=" +
                    ToString(globals->extract_data_time));
  }
  if (globals->restart_reason != NO_RESTART) {
    const char *reasons[] = {"no_restart",
                             "no_daemon",
                             "new_version",
                             "new_options",
                             "pid_file_but_no_server",
                             "server_vanished",
                             "server_unresponsive"};
    args->push_back(string("--restart_reason=") +
                    reasons[globals->restart_reason]);
  }
  args->push_back(string("--binary_path=") + globals->binary_path);
}

// Join the elements of the specified array with NUL's (\0's), akin to the
// format of /proc/$PID/cmdline.
static string GetArgumentString(const vector<string> &argument_array) {
  string result;
  blaze_util::JoinStrings(argument_array, '\0', &result);
  return result;
}

// Do a chdir into the workspace, and die if it fails.
static void GoToWorkspace(const WorkspaceLayout *workspace_layout) {
  if (workspace_layout->InWorkspace(globals->workspace) &&
      !blaze_util::ChangeDirectory(globals->workspace)) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "changing directory into " << globals->workspace
        << " failed: " << GetLastErrorString();
  }
}

// Starts the Blaze server.
static int StartServer(const WorkspaceLayout *workspace_layout,
                        BlazeServerStartup **server_startup) {
  vector<string> jvm_args_vector = GetArgumentArray(workspace_layout);
  string argument_string = GetArgumentString(jvm_args_vector);
  const string binaries_dir =
      GetEmbeddedBinariesRoot(globals->options->install_base);
  string server_dir =
      blaze_util::JoinPath(globals->options->output_base, "server");
  // Write the cmdline argument string to the server dir. If we get to this
  // point, there is no server running, so we don't overwrite the cmdline file
  // for the existing server. If might be that the server dies and the cmdline
  // file stays there, but that is not a problem, since we always check the
  // server, too.
  blaze_util::WriteFile(argument_string,
                        blaze_util::JoinPath(server_dir, "cmdline"));

  // unless we restarted for a new-version, mark this as initial start
  if (globals->restart_reason == NO_RESTART) {
    globals->restart_reason = NO_DAEMON;
  }

  string exe =
      globals->options->GetExe(globals->jvm_path, globals->ServerJarPath());
  // Go to the workspace before we daemonize, so
  // we can still print errors to the terminal.
  GoToWorkspace(workspace_layout);

  return ExecuteDaemon(exe, jvm_args_vector, PrepareEnvironmentForJvm(),
                       globals->jvm_log_file, globals->jvm_log_file_append,
                       binaries_dir, server_dir, server_startup);
}

// Replace this process with blaze in standalone/batch mode.
// The batch mode blaze process handles the command and exits.
//
// This function passes the commands array to the blaze process.
// This array should start with a command ("build", "info", etc.).
static void StartStandalone(const WorkspaceLayout *workspace_layout,
                            BlazeServer *server, uint64_t start_time) {
  if (server->Connected()) {
    server->KillRunningServer();
  }

  // Wall clock time since process startup.
  globals->startup_time = GetMillisecondsMonotonic() - start_time;

  BAZEL_LOG(INFO) << "Starting " << globals->options->product_name
                  << " in batch mode.";

  string command = globals->option_processor->GetCommand();
  const vector<string> command_arguments =
      globals->option_processor->GetCommandArguments();

  if (!command_arguments.empty() && command == "shutdown") {
    string product = globals->options->product_name;
    blaze_util::ToLower(&product);
    BAZEL_LOG(WARNING)
        << "Running command \"shutdown\" in batch mode.  Batch mode is "
           "triggered\nwhen not running "
        << globals->options->product_name
        << " within a workspace. If you intend to shutdown an\nexisting "
        << globals->options->product_name << " server, run \"" << product
        << " shutdown\" from the directory where\nit was started.";
  }
  vector<string> jvm_args_vector = GetArgumentArray(workspace_layout);
  if (!command.empty()) {
    jvm_args_vector.push_back(command);
    AddLoggingArgs(&jvm_args_vector);
  }

  jvm_args_vector.insert(jvm_args_vector.end(), command_arguments.begin(),
                         command_arguments.end());

  GoToWorkspace(workspace_layout);
  string exe =
      globals->options->GetExe(globals->jvm_path, globals->ServerJarPath());

  {
    WithEnvVars env_obj(PrepareEnvironmentForJvm());
    ExecuteProgram(exe, jvm_args_vector);
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "execv of '" << exe << "' failed: " << GetLastErrorString();
  }
}

static void WriteFileToStderrOrDie(const char *file_name) {
  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "opening " << file_name << " failed: " << GetLastErrorString();
  }
  char buffer[255];
  int num_read;
  while ((num_read = fread(buffer, 1, sizeof buffer, fp)) > 0) {
    if (ferror(fp)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "failed to read from '" << file_name
          << "': " << GetLastErrorString();
    }
    fwrite(buffer, 1, num_read, stderr);
  }
  fclose(fp);
}

// After connecting to the Blaze server, return its PID, or -1 if there was an
// error.
static int GetServerPid(const string &server_dir) {
  // Note: there is no race here on startup since the server creates
  // the pid file strictly before it binds the socket.
  string pid_file = blaze_util::JoinPath(server_dir, kServerPidFile);
  string bufstr;
  int result;
  if (!blaze_util::ReadFile(pid_file, &bufstr, 32) ||
      !blaze_util::safe_strto32(bufstr, &result)) {
    return -1;
  }

  return result;
}

static void SetRestartReasonIfNotSet(RestartReason restart_reason) {
  if (globals->restart_reason == NO_RESTART) {
    globals->restart_reason = restart_reason;
  }
}

// Starts up a new server and connects to it. Exits if it didn't work out.
static void StartServerAndConnect(const WorkspaceLayout *workspace_layout,
                                  BlazeServer *server) {
  string server_dir =
      blaze_util::JoinPath(globals->options->output_base, "server");

  // Delete the old command_port file if it already exists. Otherwise we might
  // run into the race condition that we read the old command_port file before
  // the new server has written the new file and we try to connect to the old
  // port, run into a timeout and try again.
  (void)blaze_util::UnlinkPath(
      blaze_util::JoinPath(server_dir, "command_port"));

  // The server dir has the socket, so we don't allow access by other
  // users.
  if (!blaze_util::MakeDirectories(server_dir, 0700)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "server directory '" << server_dir
        << "' could not be created: " << GetLastErrorString();
  }

  // If we couldn't connect to the server check if there is still a PID file
  // and if so, kill the server that wrote it. This can happen e.g. if the
  // server is in a GC pause and therefore cannot respond to ping requests and
  // having two server instances running in the same output base is a
  // disaster.
  int server_pid = GetServerPid(server_dir);
  if (server_pid > 0) {
    if (VerifyServerProcess(server_pid, globals->options->output_base)) {
      if (KillServerProcess(server_pid, globals->options->output_base)) {
        BAZEL_LOG(USER) << "Killed non-responsive server process (pid="
                        << server_pid << ")";
        SetRestartReasonIfNotSet(SERVER_UNRESPONSIVE);
      } else {
        SetRestartReasonIfNotSet(SERVER_VANISHED);
      }
    } else {
      SetRestartReasonIfNotSet(PID_FILE_BUT_NO_SERVER);
    }
  }

  SetScheduling(globals->options->batch_cpu_scheduling,
                globals->options->io_nice_level);

  BlazeServerStartup *server_startup;
  server_pid = StartServer(workspace_layout, &server_startup);
  BAZEL_LOG(USER) << "Starting local " << globals->options->product_name
                  << " server and connecting to it...";

  // Give the server two minutes to start up. That's enough to connect with a
  // debugger.
  const auto start_time = std::chrono::system_clock::now();
  const auto try_until_time = start_time + std::chrono::seconds(120);
  // Print an update at most once every 10 seconds if we are still trying to
  // connect.
  const auto min_message_interval = std::chrono::seconds(10);
  auto last_message_time = start_time;
  while (std::chrono::system_clock::now() < try_until_time) {
    const auto attempt_time = std::chrono::system_clock::now();
    const auto next_attempt_time =
        attempt_time + std::chrono::milliseconds(100);

    if (server->Connect()) {
      delete server_startup;
      return;
    }

    if (attempt_time >= (last_message_time + min_message_interval)) {
      auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
          attempt_time - start_time);
      BAZEL_LOG(USER) << "... still trying to connect to local "
                      << globals->options->product_name << " server after "
                      << elapsed_time.count() << " seconds ...";
      last_message_time = attempt_time;
    }

    std::this_thread::sleep_until(next_attempt_time);
    if (!server_startup->IsStillAlive()) {
      globals->option_processor->PrintStartupOptionsProvenanceMessage();
      if (globals->jvm_log_file_append) {
        // Don't dump the log if we were appending - the user should know where
        // to find it, and who knows how much content they may have accumulated.
        BAZEL_LOG(USER) << "Server crashed during startup. See "
                        << globals->jvm_log_file;
      } else {
        BAZEL_LOG(USER) << "Server crashed during startup. Now printing "
                        << globals->jvm_log_file;
        WriteFileToStderrOrDie(globals->jvm_log_file.c_str());
      }
      exit(blaze_exit_code::INTERNAL_ERROR);
    }
  }
  BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
      << "couldn't connect to server (" << server_pid << ") after 120 seconds.";
}

// A PureZipExtractorProcessor to extract the files from the blaze zip.
class ExtractBlazeZipProcessor : public PureZipExtractorProcessor {
 public:
  explicit ExtractBlazeZipProcessor(const string &embedded_binaries,
                                    blaze::embedded_binaries::Dumper *dumper)
      : embedded_binaries_(embedded_binaries), dumper_(dumper) {}

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override {
    return !devtools_ijar::zipattr_is_dir(attr);
  }

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    return AcceptPure(filename, attr);
  }

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override {
    dumper_->Dump(data, size,
                  blaze_util::JoinPath(embedded_binaries_, filename));
  }

 private:
  const string embedded_binaries_;
  blaze::embedded_binaries::Dumper *dumper_;
};

// Actually extracts the embedded data files into the tree whose root
// is 'embedded_binaries'.
static void ActuallyExtractData(const string &argv0,
                                const string &embedded_binaries) {
  std::string install_md5;
  GetInstallKeyFileProcessor install_key_processor(&install_md5);

  std::string error;
  std::unique_ptr<blaze::embedded_binaries::Dumper> dumper(
      blaze::embedded_binaries::Create(&error));
  if (dumper == nullptr) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR) << error;
  }
  ExtractBlazeZipProcessor extract_blaze_processor(embedded_binaries,
                                                   dumper.get());

  CompoundZipProcessor processor({&extract_blaze_processor,
                                  &install_key_processor});
  if (!blaze_util::MakeDirectories(embedded_binaries, 0777)) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "couldn't create '" << embedded_binaries
        << "': " << GetLastErrorString();
  }

  BAZEL_LOG(USER) << "Extracting " << globals->options->product_name
                  << " installation...";

  std::unique_ptr<devtools_ijar::ZipExtractor> extractor(
      devtools_ijar::ZipExtractor::Create(argv0.c_str(), &processor));
  if (extractor == NULL) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to open " << globals->options->product_name
        << " as a zip file: " << GetLastErrorString();
  }
  if (extractor->ProcessAll() < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to extract " << globals->options->product_name
        << " as a zip file: " << extractor->GetError();
  }

  if (!dumper->Finish(&error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to extract embedded binaries: " << error;
  }

  if (install_md5 != globals->install_md5) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "The " << globals->options->product_name << " binary at " << argv0
        << " was replaced during the client's self-extraction (old md5: "
        << globals->install_md5 << " new md5: " << install_md5
        << "). If you expected this then you should simply re-run "
        << globals->options->product_name
        << " in order to pick up the different version. If you didn't expect "
           "this then you should investigate what happened.";
  }

  // Set the timestamps of the extracted files to the future and make sure (or
  // at least as sure as we can...) that the files we have written are actually
  // on the disk.

  vector<string> extracted_files;

  // Walks the temporary directory recursively and collects full file paths.
  blaze_util::GetAllFilesUnder(embedded_binaries, &extracted_files);

  std::unique_ptr<blaze_util::IFileMtime> mtime(blaze_util::CreateFileMtime());
  set<string> synced_directories;
  for (const auto &it : extracted_files) {
    const char *extracted_path = it.c_str();

    // Set the time to a distantly futuristic value so we can observe tampering.
    // Note that keeping a static, deterministic timestamp, such as the default
    // timestamp set by unzip (1970-01-01) and using that to detect tampering is
    // not enough, because we also need the timestamp to change between Bazel
    // releases so that the metadata cache knows that the files may have
    // changed. This is essential for the correctness of actions that use
    // embedded binaries as artifacts.
    if (!mtime->SetToDistantFuture(it)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "failed to set timestamp on '" << extracted_path
          << "': " << GetLastErrorString();
    }

    blaze_util::SyncFile(it);

    string directory = blaze_util::Dirname(extracted_path);

    // Now walk up until embedded_binaries and sync every directory in between.
    // synced_directories is used to avoid syncing the same directory twice.
    // The !directory.empty() and !blaze_util::IsRootDirectory(directory)
    // conditions are not strictly needed, but it makes this loop more robust,
    // because otherwise, if due to some glitch, directory was not under
    // embedded_binaries, it would get into an infinite loop.
    while (directory != embedded_binaries &&
           synced_directories.count(directory) == 0 && !directory.empty() &&
           !blaze_util::IsRootDirectory(directory)) {
      blaze_util::SyncFile(directory);
      synced_directories.insert(directory);
      directory = blaze_util::Dirname(directory);
    }
  }

  blaze_util::SyncFile(embedded_binaries);
}

// Installs Blaze by extracting the embedded data files, iff necessary.
// The MD5-named install_base directory on disk is trusted; we assume
// no-one has modified the extracted files beneath this directory once
// it is in place. Concurrency during extraction is handled by
// extracting in a tmp dir and then renaming it into place where it
// becomes visible automically at the new path.
static void ExtractData(const string &self_path) {
  // If the install dir doesn't exist, create it, if it does, we know it's good.
  if (!blaze_util::PathExists(globals->options->install_base)) {
    uint64_t st = GetMillisecondsMonotonic();
    // Work in a temp dir to avoid races.
    string tmp_install = globals->options->install_base + ".tmp." +
                         blaze::GetProcessIdAsString();
    string tmp_binaries =
        blaze_util::JoinPath(tmp_install, "_embedded_binaries");
    ActuallyExtractData(self_path, tmp_binaries);

    uint64_t et = GetMillisecondsMonotonic();
    globals->extract_data_time = et - st;

    // Now rename the completed installation to its final name.
    int attempts = 0;
    while (attempts < 120) {
      int result = blaze_util::RenameDirectory(
          tmp_install.c_str(), globals->options->install_base.c_str());
      if (result == blaze_util::kRenameDirectorySuccess ||
          result == blaze_util::kRenameDirectoryFailureNotEmpty) {
        // If renaming fails because the directory already exists and is not
        // empty, then we assume another good installation snuck in before us.
        break;
      } else {
        // Otherwise the install directory may still be scanned by the antivirus
        // (in case we're running on Windows) so we need to wait for that to
        // finish and try renaming again.
        ++attempts;
        BAZEL_LOG(USER) << "install base directory '" << tmp_install
                        << "' could not be renamed into place after "
                        << attempts << " second(s), trying again\r";
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }

    // Give up renaming after 120 failed attempts / 2 minutes.
    if (attempts == 120) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "install base directory '" << tmp_install
          << "' could not be renamed into place: " << GetLastErrorString();
    }
  } else {
    if (!blaze_util::IsDirectory(globals->options->install_base)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "install base directory '" << globals->options->install_base
          << "' could not be created. It exists but is not a directory.";
    }

    std::unique_ptr<blaze_util::IFileMtime> mtime(
        blaze_util::CreateFileMtime());
    string real_install_dir = blaze_util::JoinPath(
        globals->options->install_base, "_embedded_binaries");
    for (const auto &it : globals->extracted_binaries) {
      string path = blaze_util::JoinPath(real_install_dir, it);
      if (!mtime->IsUntampered(path)) {
        BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
            << "corrupt installation: file '" << path
            << "' is missing or modified.  Please remove '"
            << globals->options->install_base << "' and try again.";
      }
    }
  }
}

// Returns true if the server needs to be restarted to accommodate changes
// between the two argument lists.
static bool AreStartupOptionsDifferent(
    const vector<string> &running_server_args,
    const vector<string> &requested_args) {
  // TODO(ccalvarin) when --batch is gone and the startup_options field in the
  // gRPC message is always set, there is no reason for client options that are
  // not used at server startup to be part of the startup command line. The
  // server command line difference logic can be simplified then.
  static const std::vector<string> volatile_startup_options = {
      "--option_sources=", "--max_idle_secs=", "--connect_timeout_secs=",
      "--client_debug="};

  // We need not worry about one side missing an argument and the other side
  // having the default value, since this command line is the canonical one for
  // this version of Bazel: either the default value is listed explicitly or it
  // is not, but this has nothing to do with the user's command line: it is
  // defined by GetArgumentArray(). Same applies for argument ordering.
  bool options_different = false;
  if (running_server_args.size() != requested_args.size()) {
    BAZEL_LOG(INFO) << "The new command line has a different length from the "
                       "running server's.";
    options_different = true;
  }

  // Args in running_server_args that are not in requested_args.
  bool found_missing_args = false;
  for (const string &arg : running_server_args) {
    // Split arg based on the first "=" if one exists in arg.
    const string::size_type eq_pos = arg.find_first_of('=');
    const string stripped_arg =
        (eq_pos == string::npos) ? arg : arg.substr(0, eq_pos + 1);

    // If arg is not volatile, then check whether or not it's in requested_args.
    if (std::find(volatile_startup_options.begin(),
                  volatile_startup_options.end(),
                  stripped_arg) == volatile_startup_options.end()) {
      if (std::find(requested_args.begin(), requested_args.end(), arg) ==
          requested_args.end()) {
        // If this is the first missing arg we've encountered, then print out
        // the list header.
        if (!found_missing_args) {
          BAZEL_LOG(INFO) << "Args from the running server that are not "
                             "included in the current request:";
          found_missing_args = true;
        }
        BAZEL_LOG(INFO) << "  " << arg;
        options_different = true;
      }
    }
  }

  // Args in requested_args that are not in running_server_args.
  bool found_new_args = false;
  for (const string &arg : requested_args) {
    // Split arg based on the first "=" if one exists in arg.
    const string::size_type eq_pos = arg.find_first_of('=');
    const string stripped_arg =
        (eq_pos == string::npos) ? arg : arg.substr(0, eq_pos + 1);

    // If arg is not volatile, then check whether or not it's in
    // running_server_args.
    if (std::find(volatile_startup_options.begin(),
                  volatile_startup_options.end(),
                  stripped_arg) == volatile_startup_options.end()) {
      if (std::find(running_server_args.begin(), running_server_args.end(),
                    arg) == running_server_args.end()) {
        // If this is the first new arg we've encountered, then print out the
        // list header.
        if (!found_new_args) {
          BAZEL_LOG(INFO) << "Args from the current request that were not "
                             "included when creating the server:";
          found_new_args = true;
        }
        BAZEL_LOG(INFO) << "  " << arg;
        options_different = true;
      }
    }
  }

  return options_different;
}

// Kills the running Blaze server, if any, if the startup options do not match.
static void KillRunningServerIfDifferentStartupOptions(
    const WorkspaceLayout *workspace_layout, BlazeServer *server) {
  if (!server->Connected()) {
    return;
  }

  string cmdline_path =
      blaze_util::JoinPath(globals->options->output_base, "server/cmdline");
  string old_joined_arguments;

  // No, /proc/$PID/cmdline does not work, because it is limited to 4K. Even
  // worse, its behavior differs slightly between kernels (in some, when longer
  // command lines are truncated, the last 4 bytes are replaced with
  // "..." + NUL.
  blaze_util::ReadFile(cmdline_path, &old_joined_arguments);
  vector<string> old_arguments = blaze_util::Split(old_joined_arguments, '\0');

  // These strings contain null-separated command line arguments. If they are
  // the same, the server can stay alive, otherwise, it needs shuffle off this
  // mortal coil.
  if (AreStartupOptionsDifferent(old_arguments,
                                 GetArgumentArray(workspace_layout))) {
    globals->restart_reason = NEW_OPTIONS;
    BAZEL_LOG(WARNING) << "Running " << globals->options->product_name
                       << " server needs to be killed, because the startup "
                          "options are different.";
    server->KillRunningServer();
  }
}

// Kills the old running server if it is not the same version as us,
// dealing with various combinations of installation scheme
// (installation symlink and older MD5_MANIFEST contents).
// This function requires that the installation be complete, and the
// server lock acquired.
static void EnsureCorrectRunningVersion(BlazeServer *server) {
  // Read the previous installation's semaphore symlink in output_base. If the
  // target dirs don't match, or if the symlink was not present, then kill any
  // running servers. Lastly, symlink to our installation so others know which
  // installation is running.
  string installation_path =
      blaze_util::JoinPath(globals->options->output_base, "install");
  string prev_installation;
  bool ok =
      blaze_util::ReadDirectorySymlink(installation_path, &prev_installation);
  if (!ok || !blaze_util::CompareAbsolutePaths(
                 prev_installation, globals->options->install_base)) {
    if (server->Connected()) {
      BAZEL_LOG(INFO)
          << "Killing running server because it is using another version of "
          << globals->options->product_name;
      server->KillRunningServer();
      globals->restart_reason = NEW_VERSION;
    }

    blaze_util::UnlinkPath(installation_path);
    if (!SymlinkDirectories(globals->options->install_base,
                            installation_path)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "failed to create installation symlink '" << installation_path
          << "': " << GetLastErrorString();
    }

    // Update the mtime of the install base so that cleanup tools can
    // find install bases that haven't been used for a long time
    std::unique_ptr<blaze_util::IFileMtime> mtime(
        blaze_util::CreateFileMtime());
    if (!mtime->SetToNow(globals->options->install_base)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "failed to set timestamp on '" << globals->options->install_base
          << "': " << GetLastErrorString();
    }
  }
}

static void CancelServer() { blaze_server->Cancel(); }

// Performs all I/O for a single client request to the server, and
// shuts down the client (by exit or signal).
static ATTRIBUTE_NORETURN void SendServerRequest(
    const WorkspaceLayout *workspace_layout, BlazeServer *server,
    uint64_t start_time) {
  while (true) {
    if (!server->Connected()) {
      StartServerAndConnect(workspace_layout, server);
    }

    // Check for the case when the workspace directory deleted and then gets
    // recreated while the server is running

    string server_cwd = GetProcessCWD(globals->server_pid);
    // If server_cwd is empty, GetProcessCWD failed. This notably occurs when
    // running under Docker because then readlink(/proc/[pid]/cwd) returns
    // EPERM.
    // Docker issue #6687 (https://github.com/docker/docker/issues/6687) fixed
    // this, but one still needs the --cap-add SYS_PTRACE command line flag, at
    // least according to the discussion on Docker issue #6800
    // (https://github.com/docker/docker/issues/6687), and even then, it's a
    // non-default Docker flag. Given that this occurs only in very weird
    // cases, it's better to assume that everything is alright if we can't get
    // the cwd.

    if (!server_cwd.empty() &&
        (server_cwd != globals->workspace ||                // changed
         server_cwd.find(" (deleted)") != string::npos)) {  // deleted.
      // There's a distant possibility that the two paths look the same yet are
      // actually different because the two processes have different mount
      // tables.
      BAZEL_LOG(INFO) << "Server's cwd moved or deleted (" << server_cwd
                      << ").";
      server->KillRunningServer();
    } else {
      break;
    }
  }

  BAZEL_LOG(INFO) << "Connected (server pid=" << globals->server_pid << ").";

  // Wall clock time since process startup.
  globals->startup_time = GetMillisecondsMonotonic() - start_time;

  SignalHandler::Get().Install(globals, CancelServer);
  SignalHandler::Get().PropagateSignalOrExit(server->Communicate());
}

// Parse the options, storing parsed values in globals.
static void ParseOptions(int argc, const char *argv[]) {
  std::string error;
  std::vector<std::string> args;
  args.insert(args.end(), argv, argv + argc);
  const blaze_exit_code::ExitCode parse_exit_code =
      globals->option_processor->ParseOptions(
          args, globals->workspace, globals->cwd, &error);

  if (parse_exit_code != blaze_exit_code::SUCCESS) {
    globals->option_processor->PrintStartupOptionsProvenanceMessage();
    BAZEL_DIE(parse_exit_code) << error;
  }
  globals->options = globals->option_processor->GetParsedStartupOptions();
}

// Compute the globals globals->cwd and globals->workspace.
static void ComputeWorkspace(const WorkspaceLayout *workspace_layout) {
  globals->cwd = blaze_util::MakeCanonical(blaze_util::GetCwd().c_str());
  if (globals->cwd.empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "blaze_util::MakeCanonical('" << blaze_util::GetCwd()
        << "') failed: " << GetLastErrorString();
  }
  globals->workspace = workspace_layout->GetWorkspace(globals->cwd);
}

// Figure out the base directories based on embedded data, username, cwd, etc.
// Ensures that all of globals->options->install_base,
// globals->options->output_base, globals->extracted_binaries,
// globals->lockfile, globals->jvm_log_file, and globals->install_md5 are set.
static void ComputeBaseDirectories(const WorkspaceLayout *workspace_layout,
                                   const string &self_path) {
  // Only start a server when in a workspace because otherwise we won't do more
  // than emit a help message.
  if (!workspace_layout->InWorkspace(globals->workspace)) {
    globals->options->batch = true;
  }

  // The default install_base is <output_user_root>/install/<md5(blaze)>
  // but if an install_base is specified on the command line, we use that as
  // the base instead.
  if (globals->options->install_base.empty()) {
    string install_user_root =
        blaze_util::JoinPath(globals->options->output_user_root, "install");
    ComputeInstallMd5AndNoteAllFiles(self_path);
    globals->options->install_base = blaze_util::JoinPath(install_user_root,
                                                          globals->install_md5);
  } else {
    // We still need to populate globals->install_md5 and
    // globals->extracted_binaries.
    ComputeInstallMd5AndNoteAllFiles(self_path);
  }

  if (globals->options->output_base.empty()) {
    globals->options->output_base = blaze::GetHashedBaseDir(
        globals->options->output_user_root, globals->workspace);
  }

  const char *output_base = globals->options->output_base.c_str();
  if (!blaze_util::PathExists(globals->options->output_base)) {
    if (!blaze_util::MakeDirectories(globals->options->output_base, 0777)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "Output base directory '" << output_base
          << "' could not be created: " << GetLastErrorString();
    }
  } else {
    if (!blaze_util::IsDirectory(globals->options->output_base)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "Output base directory '" << output_base
          << "' could not be created. It exists but is not a directory.";
    }
  }
  if (!blaze_util::CanAccessDirectory(globals->options->output_base)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Output base directory '" << output_base
        << "' must be readable and writable.";
  }
  ExcludePathFromBackup(output_base);

  globals->options->output_base = blaze_util::MakeCanonical(output_base);
  if (globals->options->output_base.empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "blaze_util::MakeCanonical('" << output_base
        << "') failed: " << GetLastErrorString();
  }

  globals->lockfile =
      blaze_util::JoinPath(globals->options->output_base, "lock");
  if (!globals->options->server_jvm_out.empty()) {
    globals->jvm_log_file = globals->options->server_jvm_out;
    globals->jvm_log_file_append = true;
  } else {
    globals->jvm_log_file =
      blaze_util::JoinPath(globals->options->output_base, "server/jvm.out");
    globals->jvm_log_file_append = false;
  }
}

// Prepares the environment to be suitable to start a JVM.
// Changes made to the environment in this function *will not* be part
// of '--client_env'.
static map<string, EnvVarValue> PrepareEnvironmentForJvm() {
  map<string, EnvVarValue> result;

  // Make sure all existing environment variables appear as part of the
  // resulting map unless they are overridden below by UNSET values.
  //
  // Even though the map we return is intended to represent a "delta" of
  // environment variables to modify the current process, we may actually use
  // such map to configure a process from scratch (via interfaces like execvpe
  // or posix_spawn), so we need to inherit any untouched variables.
  for (char** entry = environ; *entry != NULL; entry++) {
    const std::string var_value = *entry;
    std::string::size_type equals = var_value.find('=');
    if (equals == std::string::npos) {
      // Ignore possibly-bad environment. We don't control what we see in this
      // global variable, so it could be invalid.
      continue;
    }
    const std::string var = var_value.substr(0, equals);
    const std::string value = var_value.substr(equals + 1);
    result[var] = EnvVarValue(EnvVarAction::SET, value);
  }

  if (!blaze::GetEnv("LD_ASSUME_KERNEL").empty()) {
    // Fix for bug: if ulimit -s and LD_ASSUME_KERNEL are both
    // specified, the JVM fails to create threads.  See thread_stack_regtest.
    // This is also provoked by LD_LIBRARY_PATH=/usr/lib/debug,
    // or anything else that causes the JVM to use LinuxThreads.
    BAZEL_LOG(WARNING) << "ignoring LD_ASSUME_KERNEL in environment.";
    result["LD_ASSUME_KERNEL"] = EnvVarValue(EnvVarAction::UNSET, "");
  }

  if (!blaze::GetEnv("LD_PRELOAD").empty()) {
    BAZEL_LOG(WARNING) << "ignoring LD_PRELOAD in environment.";
    result["LD_PRELOAD"] = EnvVarValue(EnvVarAction::UNSET, "");
  }

  if (!blaze::GetEnv("_JAVA_OPTIONS").empty()) {
    // This would override --host_jvm_args
    BAZEL_LOG(WARNING) << "ignoring _JAVA_OPTIONS in environment.";
    result["_JAVA_OPTIONS"] = EnvVarValue(EnvVarAction::UNSET, "");
  }

  // TODO(bazel-team):  We've also seen a failure during loading (creating
  // threads?) when ulimit -Hs 8192.  Characterize that and check for it here.

  // Make the JVM use ISO-8859-1 for parsing its command line because "blaze
  // run" doesn't handle non-ASCII command line arguments. This is apparently
  // the most reliable way to select the platform default encoding.
  result["LANG"] = EnvVarValue(EnvVarAction::SET, "en_US.ISO-8859-1");
  result["LANGUAGE"] = EnvVarValue(EnvVarAction::SET, "en_US.ISO-8859-1");
  result["LC_ALL"] = EnvVarValue(EnvVarAction::SET, "en_US.ISO-8859-1");
  result["LC_CTYPE"] = EnvVarValue(EnvVarAction::SET, "en_US.ISO-8859-1");

  return result;
}

static string CheckAndGetBinaryPath(const string &argv0) {
  if (blaze_util::IsAbsolute(argv0)) {
    return argv0;
  } else {
    string abs_path = blaze_util::JoinPath(globals->cwd, argv0);
    string resolved_path = blaze_util::MakeCanonical(abs_path.c_str());
    if (!resolved_path.empty()) {
      return resolved_path;
    } else {
      // This happens during our integration tests, but thats okay, as we won't
      // log the invocation anyway.
      return abs_path;
    }
  }
}

int GetExitCodeForAbruptExit(const GlobalVariables &globals) {
  BAZEL_LOG(INFO) << "Looking for a custom exit-code.";
  std::string filename = blaze_util::JoinPath(
      globals.options->output_base, "exit_code_to_use_on_abrupt_exit");
  std::string content;
  if (!blaze_util::ReadFile(filename, &content)) {
    BAZEL_LOG(INFO) << "Unable to read the custom exit-code file. "
                    << "Exiting with an INTERNAL_ERROR.";
    return blaze_exit_code::INTERNAL_ERROR;
  }
  if (!blaze_util::UnlinkPath(filename)) {
    BAZEL_LOG(INFO) << "Unable to delete the custom exit-code file. "
                    << "Exiting with an INTERNAL_ERROR.";
    return blaze_exit_code::INTERNAL_ERROR;
  }
  int custom_exit_code;
  if (!blaze_util::safe_strto32(content, &custom_exit_code)) {
    BAZEL_LOG(INFO) << "Content of custom exit-code file not an int: "
                    << content << "Exiting with an INTERNAL_ERROR.";
    return blaze_exit_code::INTERNAL_ERROR;
  }
  BAZEL_LOG(INFO) << "Read exit code " << custom_exit_code
                  << " from custom exit-code file. Exiting accordingly.";
  return custom_exit_code;
}

int Main(int argc, const char *argv[], WorkspaceLayout *workspace_layout,
         OptionProcessor *option_processor, uint64_t start_time) {
  // Logging must be set first to assure no log statements are missed.
  std::unique_ptr<blaze_util::BazelLogHandler> default_handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(default_handler));

  globals = new GlobalVariables(option_processor);
  blaze::SetupStdStreams();
  if (argc == 1 && blaze::WarnIfStartedFromDesktop()) {
    // Only check and warn for from-desktop start if there were no args.
    // In this case the user probably clicked Bazel's icon (as opposed to either
    // starting it from a terminal, or as a subprocess with args, or on Windows
    // from a ".lnk" file with some args).
    return blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR;
  }

  // Best-effort operation to raise the resource limits from soft to hard.  We
  // do this early during the main program instead of just before execing the
  // Blaze server binary, because it's easier (for testing purposes) and because
  // the Blaze client also benefits from this (e.g. during installation).
  UnlimitResources();

  // Must be done before command line parsing.
  ComputeWorkspace(workspace_layout);

#if defined(_WIN32) || defined(__CYGWIN__)
  // Must be done before command line parsing.
  // ParseOptions already populate --client_env, so detect bash before it
  // happens.
  DetectBashOrDie();
#endif  // if defined(_WIN32) || defined(__CYGWIN__)

  globals->binary_path = CheckAndGetBinaryPath(argv[0]);
  ParseOptions(argc, argv);

  SetDebugLog(globals->options->client_debug);
  // If client_debug was false, this is ignored, so it's accurate.
  BAZEL_LOG(INFO) << "Debug logging requested, sending all client log "
                     "statements to stderr";
  // TODO(b/79206210): Can't log this before SetDebugLog is called, since the
  // warning might get swallowed. Once the bug is fixed, move this call to
  // OptionProcessor::ParseOptions where the order of operations is more clear.
  globals->options->MaybeLogStartupOptionWarnings();

  if (globals->options->unlimit_coredumps) {
    UnlimitCoredumps();
  }

  blaze::CreateSecureOutputRoot(globals->options->output_user_root);

  const string self_path = GetSelfPath();
  ComputeBaseDirectories(workspace_layout, self_path);

  blaze_server = static_cast<BlazeServer *>(
      new GrpcBlazeServer(globals->options->connect_timeout_secs));

  globals->command_wait_time = blaze_server->AcquireLock();
  BAZEL_LOG(INFO) << "Acquired the client lock, waited "
                  << globals->command_wait_time << " milliseconds";

  WarnFilesystemType(globals->options->output_base);

  ExtractData(self_path);
  globals->jvm_path = globals->options->GetJvm();

  blaze_server->Connect();

  if (!globals->options->batch &&
      "shutdown" == globals->option_processor->GetCommand() &&
      !blaze_server->Connected()) {
    return 0;
  }

  EnsureCorrectRunningVersion(blaze_server);
  KillRunningServerIfDifferentStartupOptions(workspace_layout, blaze_server);

  if (globals->options->batch) {
    SetScheduling(globals->options->batch_cpu_scheduling,
                  globals->options->io_nice_level);
    StartStandalone(workspace_layout, blaze_server, start_time);
  } else {
    SendServerRequest(workspace_layout, blaze_server, start_time);
  }
  return 0;
}

static void null_grpc_log_function(gpr_log_func_args *args) {}

// There might be a mismatch between std::string and the string type returned
// from protos. This function is the safe way to compare such strings.
template <typename StringTypeA, typename StringTypeB>
static bool ProtoStringEqual(const StringTypeA &cookieA,
                             const StringTypeB &cookieB) {
  // use strncmp insted of strcmp to deal with null bytes in the cookie.
  auto cookie_length = cookieA.size();
  if (cookie_length != cookieB.size()) {
    return false;
  }
  return memcmp(cookieA.c_str(), cookieB.c_str(), cookie_length) == 0;
}

GrpcBlazeServer::GrpcBlazeServer(int connect_timeout_secs) {
  connected_ = false;
  connect_timeout_secs_ = connect_timeout_secs;

  gpr_set_log_function(null_grpc_log_function);

  pipe_ = blaze_util::CreatePipe();
  if (pipe_ == NULL) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Couldn't create pipe: " << GetLastErrorString();
  }
}

GrpcBlazeServer::~GrpcBlazeServer() {
  delete pipe_;
  pipe_ = NULL;
}

bool GrpcBlazeServer::TryConnect(
    CommandServer::Stub *client) {
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::seconds(connect_timeout_secs_));

  command_server::PingRequest request;
  command_server::PingResponse response;
  request.set_cookie(request_cookie_);

  BAZEL_LOG(INFO) << "Trying to connect to server (timeout: "
                  << connect_timeout_secs_ << " secs)...";
  grpc::Status status = client->Ping(&context, request, &response);

  if (!status.ok() || !ProtoStringEqual(response.cookie(), response_cookie_)) {
    BAZEL_LOG(INFO) << "Connection to server failed: "
                    << status.error_message().c_str();
    return false;
  }

  return true;
}

bool GrpcBlazeServer::Connect() {
  assert(!connected_);

  std::string server_dir =
      blaze_util::JoinPath(globals->options->output_base, "server");
  std::string port;
  std::string ipv4_prefix = "127.0.0.1:";
  std::string ipv6_prefix_1 = "[0:0:0:0:0:0:0:1]:";
  std::string ipv6_prefix_2 = "[::1]:";

  if (!blaze_util::ReadFile(blaze_util::JoinPath(server_dir, "command_port"),
                            &port)) {
    return false;
  }

  // Make sure that we are being directed to localhost
  if (port.compare(0, ipv4_prefix.size(), ipv4_prefix) &&
      port.compare(0, ipv6_prefix_1.size(), ipv6_prefix_1) &&
      port.compare(0, ipv6_prefix_2.size(), ipv6_prefix_2)) {
    return false;
  }

  if (!blaze_util::ReadFile(blaze_util::JoinPath(server_dir, "request_cookie"),
                            &request_cookie_)) {
    return false;
  }

  if (!blaze_util::ReadFile(blaze_util::JoinPath(server_dir, "response_cookie"),
                            &response_cookie_)) {
    return false;
  }

  pid_t server_pid = GetServerPid(server_dir);
  if (server_pid < 0) {
    return false;
  }

  if (!VerifyServerProcess(server_pid, globals->options->output_base)) {
    return false;
  }

  grpc::ChannelArguments channel_args;
  // Bazel client and server always run on the same machine and communicate
  // locally over gRPC; so we want to ignore any configured proxies when setting
  // up a gRPC channel to the server.
  channel_args.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
  std::shared_ptr<grpc::Channel> channel(grpc::CreateCustomChannel(
      port, grpc::InsecureChannelCredentials(), channel_args));
  std::unique_ptr<CommandServer::Stub> client(
      CommandServer::NewStub(channel));

  if (!TryConnect(client.get())) {
    return false;
  }

  this->client_ = std::move(client);
  connected_ = true;
  globals->server_pid = server_pid;
  return true;
}

// Cancellation works as follows:
//
// When the user presses Ctrl-C, a SIGINT is delivered to the client, which is
// translated into a BlazeServer::Cancel() call. Since it's not a good idea to
// do significant work in signal handlers, all it does is write a byte to an
// unnamed pipe.
//
// This unnamed pipe is used to communicate with the cancel thread. Whenever
// something interesting happens, a byte is written into it, which is read by
// the cancel thread. These commands are available:
//
// - NOP
// - JOIN. The cancel thread needs to be terminated.
// - CANCEL. If the command ID is already available, a cancel request is sent.
// - COMMAND_ID_RECEIVED. The client learned the command ID from the server.
//   If there is a pending cancellation request, it is acted upon.
//
// The only data the cancellation thread shares with the main thread is the
// file descriptor for receiving commands and command_id_, the latter of which
// is protected by a mutex, which mainly serves as a memory fence.
//
// The cancellation thread is joined at the end of the execution of the command.
// The main thread wakes it up just so that it can finish (using the JOIN
// action)
//
// It's conceivable that the server is busy and thus it cannot service the
// cancellation request. In that case, we simply ignore the failure and the both
// the server and the client go on as if nothing had happened (except that this
// Ctrl-C still counts as a SIGINT, three of which result in a SIGKILL being
// delivered to the server)
void GrpcBlazeServer::CancelThread() {
  bool running = true;
  bool cancel = false;
  bool command_id_received = false;
  while (running) {
    char buf;

    int error;
    int bytes_read = pipe_->Receive(&buf, 1, &error);
    if (bytes_read < 0 && error == blaze_util::IPipe::INTERRUPTED) {
      continue;
    } else if (bytes_read != 1) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "Cannot communicate with cancel thread: " << GetLastErrorString();
    }

    switch (buf) {
      case CancelThreadAction::NOTHING:
        break;

      case CancelThreadAction::JOIN:
        running = false;
        break;

      case CancelThreadAction::COMMAND_ID_RECEIVED:
        command_id_received = true;
        if (cancel) {
          SendCancelMessage();
          cancel = false;
        }
        break;

      case CancelThreadAction::CANCEL:
        if (command_id_received) {
          SendCancelMessage();
        } else {
          cancel = true;
        }
        break;
    }
  }
}

void GrpcBlazeServer::SendCancelMessage() {
  std::unique_lock<std::mutex> lock(cancel_thread_mutex_);

  command_server::CancelRequest request;
  request.set_cookie(request_cookie_);
  request.set_command_id(command_id_);
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::seconds(10));
  command_server::CancelResponse response;
  // There isn't a lot we can do if this request fails
  grpc::Status status = client_->Cancel(&context, request, &response);
  if (!status.ok()) {
    BAZEL_LOG(USER) << "\nCould not interrupt server ("
                    << status.error_message().c_str() << ")\n";
  }
}

// This will wait indefinitely until the server shuts down
void GrpcBlazeServer::KillRunningServer() {
  assert(connected_);

  grpc::ClientContext context;
  command_server::RunRequest request;
  command_server::RunResponse response;
  request.set_cookie(request_cookie_);
  request.set_block_for_lock(globals->options->block_for_lock);
  request.set_client_description("pid=" + blaze::GetProcessIdAsString() +
                                 " (for shutdown)");
  request.add_arg("shutdown");
  BAZEL_LOG(INFO) << "Shutting running server with request ["
                  << request.ShortDebugString() << "]";
  std::unique_ptr<grpc::ClientReader<command_server::RunResponse>> reader(
      client_->Run(&context, request));

  // TODO(b/111179585): Swallowing these responses loses potential messages from
  // the server, which may be useful in understanding why a shutdown failed.
  // However, we don't want to spam the user in case the shutdown works
  // perfectly fine, so we discard the information. For --noblock_for_lock, this
  // means that we don't output the PID of the competing client, which isn't
  // great. We could either store the stderr_output returned by the server and
  // output it in the case of a failed shutdown, or we could add a
  // special-cased field in RunResponse for this purpose.
  while (reader->Read(&response)) {
  }

  // Check the final message from the server to see if it exited because another
  // command holds the client lock.
  if (response.finished()) {
    if (response.exit_code() == blaze_exit_code::LOCK_HELD_NOBLOCK_FOR_LOCK) {
      assert(!globals->options->block_for_lock);
      BAZEL_DIE(blaze_exit_code::LOCK_HELD_NOBLOCK_FOR_LOCK)
          << "Exiting because the lock is held and --noblock_for_lock was "
             "given.";
    }
  }

  // If for any reason the shutdown request failed to initiate a termination,
  // this is a bug. Yes, this means the server won't be forced to shut down,
  // which might be the preferred behavior, but it will help identify the bug.
  assert(response.termination_expected());
  // Wait for the server process to terminate (if we know the server PID).
  // If it does not terminate itself gracefully within 1m, terminate it.
  if (globals->server_pid > 0 &&
      !AwaitServerProcessTermination(globals->server_pid,
                                     globals->options->output_base,
                                     kPostShutdownGracePeriodSeconds)) {
    KillServerProcess(globals->server_pid, globals->options->output_base);
  }

  connected_ = false;
}

unsigned int GrpcBlazeServer::Communicate() {
  assert(connected_);
  assert(globals->server_pid > 0);

  vector<string> arg_vector;
  string command = globals->option_processor->GetCommand();
  if (!command.empty()) {
    arg_vector.push_back(command);
    AddLoggingArgs(&arg_vector);
  }

  const vector<string> command_args =
      globals->option_processor->GetCommandArguments();
  if (!command_args.empty()) {
    arg_vector.insert(arg_vector.end(),
                      command_args.begin(),
                      command_args.end());
  }

  command_server::RunRequest request;
  request.set_cookie(request_cookie_);
  request.set_block_for_lock(globals->options->block_for_lock);
  request.set_client_description("pid=" + blaze::GetProcessIdAsString());
  for (const string &arg : arg_vector) {
    request.add_arg(arg);
  }
  if (!globals->options->invocation_policy.empty()) {
    request.set_invocation_policy(globals->options->invocation_policy);
  }

  const StartupOptions *startup_options(
      globals->option_processor->GetParsedStartupOptions());
  for (const auto &startup_option :
       startup_options->original_startup_options_) {
    command_server::StartupOption *proto_option_field =
        request.add_startup_options();
    request.add_startup_options();
    proto_option_field->set_source(startup_option.source);
    proto_option_field->set_option(startup_option.value);
  }

  grpc::ClientContext context;
  command_server::RunResponse response;
  std::unique_ptr<grpc::ClientReader<command_server::RunResponse>> reader(
      client_->Run(&context, request));

  // Release the server lock because the gRPC handles concurrent clients just
  // fine. Note that this may result in two "waiting for other client" messages
  // (one during server startup and one emitted by the server)
  BAZEL_LOG(INFO)
      << "Releasing client lock, let the server manage concurrent requests.";
  blaze::ReleaseLock(&blaze_lock_);

  std::thread cancel_thread(&GrpcBlazeServer::CancelThread, this);
  bool command_id_set = false;
  bool pipe_broken = false;
  command_server::RunResponse final_response;
  bool finished = false;
  bool finished_warning_emitted = false;

  while (reader->Read(&response)) {
    if (finished && !finished_warning_emitted) {
      BAZEL_LOG(USER) << "\nServer returned messages after reporting exit code";
      finished_warning_emitted = true;
    }

    if (!ProtoStringEqual(response.cookie(), response_cookie_)) {
      BAZEL_LOG(USER) << "\nServer response cookie invalid, exiting";
      return blaze_exit_code::INTERNAL_ERROR;
    }

    const char *broken_pipe_name = nullptr;

    if (response.finished()) {
      final_response = response;
      finished = true;
    }

    if (!response.standard_output().empty()) {
      size_t size = response.standard_output().size();
      if (blaze_util::WriteToStdOutErr(response.standard_output().c_str(), size,
                                       /* to_stdout */ true) ==
          blaze_util::WriteResult::BROKEN_PIPE) {
        broken_pipe_name = "standard output";
      }
    }

    if (!response.standard_error().empty()) {
      size_t size = response.standard_error().size();
      if (blaze_util::WriteToStdOutErr(response.standard_error().c_str(), size,
                                       /* to_stdout */ false) ==
          blaze_util::WriteResult::BROKEN_PIPE) {
        broken_pipe_name = "standard error";
      }
    }

    if (broken_pipe_name != nullptr && !pipe_broken) {
      pipe_broken = true;
      BAZEL_LOG(USER) << "\nCannot write to " << broken_pipe_name
                      << "; exiting...\n";
      Cancel();
    }

    if (!command_id_set && !response.command_id().empty()) {
      std::unique_lock<std::mutex> lock(cancel_thread_mutex_);
      command_id_ = response.command_id();
      command_id_set = true;
      SendAction(CancelThreadAction::COMMAND_ID_RECEIVED);
    }
  }

  // If the server has shut down, but does not terminate itself within a 1m
  // grace period, terminate it.
  if (final_response.termination_expected() &&
      !AwaitServerProcessTermination(globals->server_pid,
                                     globals->options->output_base,
                                     kPostShutdownGracePeriodSeconds)) {
    KillServerProcess(globals->server_pid, globals->options->output_base);
  }

  SendAction(CancelThreadAction::JOIN);
  cancel_thread.join();

  grpc::Status status = reader->Finish();
  if (!status.ok()) {
    BAZEL_LOG(USER) << "\nServer terminated abruptly (error code: "
                    << status.error_code() << ", error message: '"
                    << status.error_message() << "', log file: '"
                    << globals->jvm_log_file << "')\n";
    return GetExitCodeForAbruptExit(*globals);
  } else if (!finished) {
    BAZEL_LOG(USER)
        << "\nServer finished RPC without an explicit exit code (log file: '"
        << globals->jvm_log_file << "')\n";
    return GetExitCodeForAbruptExit(*globals);
  } else if (final_response.has_exec_request()) {
    const command_server::ExecRequest& request = final_response.exec_request();
    if (request.argv_size() < 1) {
      BAZEL_LOG(USER)
          << "\nServer requested exec() but did not pass a binary to execute\n";
      return blaze_exit_code::INTERNAL_ERROR;
    }

    vector<string> argv;
    argv.insert(argv.begin(), request.argv().begin(), request.argv().end());
    for (const auto& variable : request.environment_variable()) {
      SetEnv(variable.name(), variable.value());
    }

    if (!blaze_util::ChangeDirectory(request.working_directory())) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "changing directory into " << request.working_directory()
          << " failed: " << GetLastErrorString();
    }

    // Execute the requested program, but before doing so, flush everything
    // we still have to say.
    fflush(NULL);
    ExecuteProgram(request.argv(0), argv);
  }

  // We'll exit with exit code SIGPIPE on Unixes due to PropagateSignalOnExit()
  return pipe_broken
      ? blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR
      : final_response.exit_code();
}

void GrpcBlazeServer::Disconnect() {
  assert(connected_);

  client_.reset();
  request_cookie_ = "";
  response_cookie_ = "";
  connected_ = false;
}

void GrpcBlazeServer::SendAction(CancelThreadAction action) {
  char msg = action;
  if (!pipe_->Send(&msg, 1)) {
    blaze::SigPrintf(
        "\nCould not interrupt server (cannot write to client pipe)\n\n");
  }
}

void GrpcBlazeServer::Cancel() {
  assert(connected_);
  SendAction(CancelThreadAction::CANCEL);
}

}  // namespace blaze
