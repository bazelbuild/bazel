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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grpc++/channel.h>
#include <grpc++/client_context.h>
#include <grpc++/create_channel.h>
#include <grpc++/security/credentials.h>
#include <grpc/grpc.h>
#include <grpc/support/log.h>

#include <algorithm>
#include <chrono>  // NOLINT (gRPC requires this)
#include <mutex>   // NOLINT
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/global_variables.h"
#include "src/main/cpp/option_processor.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"
#include "third_party/ijar/zip.h"

#include "src/main/protobuf/command_server.grpc.pb.h"

using blaze_util::die;
using blaze_util::pdie;
using blaze_util::PrintWarning;

namespace blaze {

using std::set;
using std::string;
using std::vector;

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
// - Have the server check that the PID file containts the correct things
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

  std::unique_ptr<command_server::CommandServer::Stub> client_;
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

  bool TryConnect(command_server::CommandServer::Stub *client);
  void CancelThread();
  void SendAction(CancelThreadAction action);
  void SendCancelMessage();
};

////////////////////////////////////////////////////////////////////////
// Logic

void debug_log(const char *format, ...) {
  if (!globals->options->client_debug) {
    return;
  }

  fprintf(stderr, "CLIENT: ");
  va_list arglist;
  va_start(arglist, format);
  vfprintf(stderr, format, arglist);
  va_end(arglist);
  fprintf(stderr, "%s", "\n");
  fflush(stderr);
}

// A devtools_ijar::ZipExtractorProcessor to extract the InstallKeyFile
class GetInstallKeyFileProcessor : public devtools_ijar::ZipExtractorProcessor {
 public:
  explicit GetInstallKeyFileProcessor(string *install_base_key)
      : install_base_key_(install_base_key) {}

  virtual bool Accept(const char *filename, const devtools_ijar::u4 attr) {
    globals->extracted_binaries.push_back(filename);
    return strcmp(filename, "install_base_key") == 0;
  }

  virtual void Process(const char *filename, const devtools_ijar::u4 attr,
                       const devtools_ijar::u1 *data, const size_t size) {
    string str(reinterpret_cast<const char *>(data), size);
    blaze_util::StripWhitespace(&str);
    if (str.size() != 32) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "\nFailed to extract install_base_key: file size mismatch "
          "(should be 32, is %zd)",
          str.size());
    }
    *install_base_key_ = str;
  }

 private:
  string *install_base_key_;
};

// Returns the install base (the root concatenated with the contents of the file
// 'install_base_key' contained as a ZIP entry in the Blaze binary); as a side
// effect, it also populates the extracted_binaries global variable.
static string GetInstallBase(const string &root, const string &self_path) {
  GetInstallKeyFileProcessor processor(&globals->install_md5);
  std::unique_ptr<devtools_ijar::ZipExtractor> extractor(
      devtools_ijar::ZipExtractor::Create(self_path.c_str(), &processor));
  if (extractor.get() == NULL) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to open %s as a zip file: %s",
        globals->options->product_name.c_str(),
        blaze_util::GetLastErrorString().c_str());
  }
  if (extractor->ProcessAll() < 0) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to extract install_base_key: %s", extractor->GetError());
  }

  if (globals->install_md5.empty()) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to find install_base_key's in zip file");
  }
  return blaze_util::JoinPath(root, globals->install_md5);
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
static vector<string> GetArgumentArray() {
  vector<string> result;

  // e.g. A Blaze server process running in ~/src/build_root (where there's a
  // ~/src/build_root/WORKSPACE file) will appear in ps(1) as "blaze(src)".
  string workspace =
      blaze_util::Basename(blaze_util::Dirname(globals->workspace));
  string product = globals->options->product_name;
  blaze_util::ToLower(&product);
  result.push_back(product + "(" + workspace + ")");
  globals->options->AddJVMArgumentPrefix(
      blaze_util::Dirname(blaze_util::Dirname(globals->jvm_path)), &result);

  result.push_back("-XX:+HeapDumpOnOutOfMemoryError");
  string heap_crash_path = globals->options->output_base;
  result.push_back("-XX:HeapDumpPath=" + blaze::PathAsJvmFlag(heap_crash_path));

  result.push_back("-Xverify:none");

  vector<string> user_options;

  user_options.insert(user_options.begin(),
                      globals->options->host_jvm_args.begin(),
                      globals->options->host_jvm_args.end());

  // Add JVM arguments particular to building blaze64 and particular JVM
  // versions.
  string error;
  blaze_exit_code::ExitCode jvm_args_exit_code =
      globals->options->AddJVMArguments(globals->options->GetHostJavabase(),
                                        &result, user_options, &error);
  if (jvm_args_exit_code != blaze_exit_code::SUCCESS) {
    die(jvm_args_exit_code, "%s", error.c_str());
  }

  // We put all directories on the java.library.path that contain .so files.
  string java_library_path = "-Djava.library.path=";
  string real_install_dir =
      GetEmbeddedBinariesRoot(globals->options->install_base);

  bool first = true;
  for (const auto &it : globals->extracted_binaries) {
    if (IsSharedLibrary(it)) {
      if (!first) {
        java_library_path += kListSeparator;
      }
      first = false;
      java_library_path += blaze::PathAsJvmFlag(
          blaze_util::JoinPath(real_install_dir, blaze_util::Dirname(it)));
    }
  }
  result.push_back(java_library_path);

  // Force use of latin1 for file names.
  result.push_back("-Dfile.encoding=ISO-8859-1");

  if (globals->options->host_jvm_debug) {
    fprintf(stderr,
            "Running host JVM under debugger (listening on TCP port 5005).\n");
    // Start JVM so that it listens for a connection from a
    // JDWP-compliant debugger:
    result.push_back("-Xdebug");
    result.push_back("-Xrunjdwp:transport=dt_socket,server=y,address=5005");
  }
  result.insert(result.end(), user_options.begin(), user_options.end());

  globals->options->AddJVMArgumentSuffix(
      real_install_dir, globals->extracted_binaries[0], &result);

  // JVM arguments are complete. Now pass in Blaze startup options.
  // Note that we always use the --flag=ARG form (instead of the --flag ARG one)
  // so that BlazeRuntime#splitStartupOptions has an easy job.

  // TODO(lberki): Test that whatever the list constructed after this line is
  // actually a list of parseable startup options.
  if (!globals->options->batch) {
    result.push_back("--max_idle_secs=" +
                     ToString(globals->options->max_idle_secs));
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

  result.push_back("--install_base=" +
                   blaze::ConvertPath(globals->options->install_base));
  result.push_back("--install_md5=" + globals->install_md5);
  result.push_back("--output_base=" +
                   blaze::ConvertPath(globals->options->output_base));
  result.push_back("--workspace_directory=" +
                   blaze::ConvertPath(globals->workspace));

  if (globals->options->allow_configurable_attributes) {
    result.push_back("--allow_configurable_attributes");
  }
  if (globals->options->deep_execroot) {
    result.push_back("--deep_execroot");
  } else {
    result.push_back("--nodeep_execroot");
  }
  if (globals->options->oom_more_eagerly) {
    result.push_back("--experimental_oom_more_eagerly");
  }
  result.push_back("--experimental_oom_more_eagerly_threshold=" +
                   ToString(globals->options->oom_more_eagerly_threshold));

  if (!globals->options->write_command_log) {
    result.push_back("--nowrite_command_log");
  }

  if (globals->options->watchfs) {
    result.push_back("--watchfs");
  }
  if (globals->options->fatal_event_bus_exceptions) {
    result.push_back("--fatal_event_bus_exceptions");
  } else {
    result.push_back("--nofatal_event_bus_exceptions");
  }

  // We use this syntax so that the logic in ServerNeedsToBeKilled() that
  // decides whether the server needs killing is simpler. This is parsed by the
  // Java code where --noclient_debug and --client_debug=false are equivalent.
  // Note that --client_debug false (separated by space) won't work either,
  // because the logic in ServerNeedsToBeKilled() assumes that every argument
  // is in the --arg=value form.
  if (globals->options->client_debug) {
    result.push_back("--client_debug=true");
  } else {
    result.push_back("--client_debug=false");
  }

  if (!globals->options->GetExplicitHostJavabase().empty()) {
    result.push_back("--host_javabase=" +
                     globals->options->GetExplicitHostJavabase());
  }

  // This is only for Blaze reporting purposes; the real interpretation of the
  // jvm flags occurs when we set up the java command line.
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
  if (globals->options->batch && globals->options->invocation_policy != NULL &&
      strlen(globals->options->invocation_policy) > 0) {
    result.push_back(string("--invocation_policy=") +
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
    const char *reasons[] = {"no_restart", "no_daemon", "new_version",
                             "new_options"};
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
    pdie(blaze_exit_code::INTERNAL_ERROR, "changing directory into %s failed",
         globals->workspace.c_str());
  }
}

// Check the java version if a java version specification is bundled. On
// success, returns the executable path of the java command.
static void VerifyJavaVersionAndSetJvm() {
  string exe = globals->options->GetJvm();

  string version_spec_file = blaze_util::JoinPath(
      GetEmbeddedBinariesRoot(globals->options->install_base), "java.version");
  string version_spec = "";
  if (blaze_util::ReadFile(version_spec_file, &version_spec)) {
    blaze_util::StripWhitespace(&version_spec);
    // A version specification is given, get version of java.
    string jvm_version = GetJvmVersion(exe);

    // Compare that jvm_version is found and at least the one specified.
    if (jvm_version.size() == 0) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Java version not detected while at least %s is needed.\n"
          "Please set JAVA_HOME.",
          version_spec.c_str());
    } else if (!CheckJavaVersionIsAtLeast(jvm_version, version_spec)) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Java version is %s while at least %s is needed.\n"
          "Please set JAVA_HOME.",
          jvm_version.c_str(), version_spec.c_str());
    }
  }

  globals->jvm_path = exe;
}

// Starts the Blaze server.  Returns a readable fd connected to the server.
// This is currently used only to detect liveness.
static void StartServer(const WorkspaceLayout *workspace_layout,
                        BlazeServerStartup **server_startup) {
  vector<string> jvm_args_vector = GetArgumentArray();
  string argument_string = GetArgumentString(jvm_args_vector);
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

  string exe = globals->options->GetExe(globals->jvm_path,
                                        globals->extracted_binaries[0]);
  // Go to the workspace before we daemonize, so
  // we can still print errors to the terminal.
  GoToWorkspace(workspace_layout);

  ExecuteDaemon(exe, jvm_args_vector, globals->jvm_log_file, server_dir,
                server_startup);
}

// Replace this process with blaze in standalone/batch mode.
// The batch mode blaze process handles the command and exits.
//
// This function passes the commands array to the blaze process.
// This array should start with a command ("build", "info", etc.).
static void StartStandalone(const WorkspaceLayout *workspace_layout,
                            BlazeServer *server) {
  if (server->Connected()) {
    server->KillRunningServer();
  }

  // Wall clock time since process startup.
  globals->startup_time = GetMillisecondsSinceProcessStart();

  if (VerboseLogging()) {
    fprintf(stderr, "Starting %s in batch mode.\n",
            globals->options->product_name.c_str());
  }
  string command = globals->option_processor->GetCommand();
  vector<string> command_arguments;
  globals->option_processor->GetCommandArguments(&command_arguments);

  if (!command_arguments.empty() && command == "shutdown") {
    string product = globals->options->product_name;
    blaze_util::ToLower(&product);
    PrintWarning(
        "Running command \"shutdown\" in batch mode.  Batch mode "
        "is triggered\nwhen not running %s within a workspace. If you "
        "intend to shutdown an\nexisting %s server, run \"%s "
        "shutdown\" from the directory where\nit was started.",
        globals->options->product_name.c_str(),
        globals->options->product_name.c_str(), product.c_str());
  }
  vector<string> jvm_args_vector = GetArgumentArray();
  if (command != "") {
    jvm_args_vector.push_back(command);
    AddLoggingArgs(&jvm_args_vector);
  }

  jvm_args_vector.insert(jvm_args_vector.end(), command_arguments.begin(),
                         command_arguments.end());

  GoToWorkspace(workspace_layout);

  string exe = globals->options->GetExe(globals->jvm_path,
                                        globals->extracted_binaries[0]);
  ExecuteProgram(exe, jvm_args_vector);
  pdie(blaze_exit_code::INTERNAL_ERROR, "execv of '%s' failed", exe.c_str());
}

static void WriteFileToStderrOrDie(const char *file_name) {
  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "opening %s failed",
         file_name);
  }
  char buffer[255];
  int num_read;
  while ((num_read = fread(buffer, 1, sizeof buffer, fp)) > 0) {
    if (ferror(fp)) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to read from '%s'", file_name);
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

// Starts up a new server and connects to it. Exits if it didn't work not.
static void StartServerAndConnect(const WorkspaceLayout *workspace_layout,
                                  BlazeServer *server) {
  string server_dir =
      blaze_util::JoinPath(globals->options->output_base, "server");

  // The server dir has the socket, so we don't allow access by other
  // users.
  if (!blaze_util::MakeDirectories(server_dir, 0700)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "server directory '%s' could not be created", server_dir.c_str());
  }

  // TODO(laszlocsomor) 2016-11-21: remove `pid_symlink` and the `remove` call
  // after 2017-05-01 (~half a year from writing this comment). By that time old
  // Bazel clients that used to write PID symlinks will probably no longer be in
  // use.
  // Until then, defensively delete old PID symlinks that older clients may have
  // left behind.
  string pid_symlink = blaze_util::JoinPath(server_dir, kServerPidSymlink);
  remove(pid_symlink.c_str());

  // If we couldn't connect to the server check if there is still a PID file
  // and if so, kill the server that wrote it. This can happen e.g. if the
  // server is in a GC pause and therefore cannot respond to ping requests and
  // having two server instances running in the same output base is a
  // disaster.
  int server_pid = GetServerPid(server_dir);
  if (server_pid > 0) {
    if (VerifyServerProcess(server_pid, globals->options->output_base,
                            globals->options->install_base) &&
        KillServerProcess(server_pid)) {
      fprintf(stderr, "Killed non-responsive server process (pid=%d)\n",
              server_pid);
    }
  }

  SetScheduling(globals->options->batch_cpu_scheduling,
                globals->options->io_nice_level);

  BlazeServerStartup *server_startup;
  StartServer(workspace_layout, &server_startup);

  // Give the server two minutes to start up. That's enough to connect with a
  // debugger.
  auto try_until_time(std::chrono::system_clock::now() +
                      std::chrono::seconds(120));
  bool had_to_wait = false;
  while (std::chrono::system_clock::now() < try_until_time) {
    auto next_attempt_time(std::chrono::system_clock::now() +
                           std::chrono::milliseconds(100));
    if (server->Connect()) {
      if (had_to_wait && !globals->options->client_debug) {
        fputc('\n', stderr);
        fflush(stderr);
      }
      delete server_startup;
      return;
    }

    had_to_wait = true;
    if (!globals->options->client_debug) {
      fputc('.', stderr);
      fflush(stderr);
    }

    std::this_thread::sleep_until(next_attempt_time);
    if (!server_startup->IsStillAlive()) {
      fprintf(stderr,
              "\nunexpected pipe read status: %s\n"
              "Server presumed dead. Now printing '%s':\n",
              blaze_util::GetLastErrorString().c_str(),
              globals->jvm_log_file.c_str());
      WriteFileToStderrOrDie(globals->jvm_log_file.c_str());
      exit(blaze_exit_code::INTERNAL_ERROR);
    }
  }
  die(blaze_exit_code::INTERNAL_ERROR,
      "\nError: couldn't connect to server after 120 seconds.");
}

// A devtools_ijar::ZipExtractorProcessor to extract the files from the blaze
// zip.
class ExtractBlazeZipProcessor : public devtools_ijar::ZipExtractorProcessor {
 public:
  explicit ExtractBlazeZipProcessor(const string &embedded_binaries)
      : embedded_binaries_(embedded_binaries) {}

  virtual bool Accept(const char *filename, const devtools_ijar::u4 attr) {
    return !devtools_ijar::zipattr_is_dir(attr);
  }

  virtual void Process(const char *filename, const devtools_ijar::u4 attr,
                       const devtools_ijar::u1 *data, const size_t size) {
    string path = blaze_util::JoinPath(embedded_binaries_, filename);
    if (!blaze_util::MakeDirectories(blaze_util::Dirname(path), 0777)) {
      pdie(blaze_exit_code::INTERNAL_ERROR, "couldn't create '%s'",
           path.c_str());
    }

    if (!blaze_util::WriteFile(data, size, path)) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "\nFailed to write zipped file \"%s\": %s", path.c_str(),
          blaze_util::GetLastErrorString().c_str());
    }
  }

 private:
  const string embedded_binaries_;
};

// Actually extracts the embedded data files into the tree whose root
// is 'embedded_binaries'.
static void ActuallyExtractData(const string &argv0,
                                const string &embedded_binaries) {
  ExtractBlazeZipProcessor processor(embedded_binaries);
  if (!blaze_util::MakeDirectories(embedded_binaries, 0777)) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "couldn't create '%s'",
         embedded_binaries.c_str());
  }

  fprintf(stderr, "Extracting %s installation...\n",
          globals->options->product_name.c_str());
  std::unique_ptr<devtools_ijar::ZipExtractor> extractor(
      devtools_ijar::ZipExtractor::Create(argv0.c_str(), &processor));
  if (extractor.get() == NULL) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to open %s as a zip file: %s",
        globals->options->product_name.c_str(),
        blaze_util::GetLastErrorString().c_str());
  }
  if (extractor->ProcessAll() < 0) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to extract %s as a zip file: %s",
        globals->options->product_name.c_str(), extractor->GetError());
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
    if (!mtime.get()->SetToDistantFuture(it)) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to set timestamp on '%s'", extracted_path);
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
// Populates globals->extracted_binaries with their extracted locations.
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
        fprintf(stderr,
                "install base directory '%s' could not be renamed into place"
                "after %d second(s), trying again\r",
                tmp_install.c_str(), attempts);
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }

    // Give up renaming after 120 failed attempts / 2 minutes.
    if (attempts == 120) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "install base directory '%s' could not be renamed into place",
           tmp_install.c_str());
    }
  } else {
    if (!blaze_util::IsDirectory(globals->options->install_base)) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Error: Install base directory '%s' could not be created. "
          "It exists but is not a directory.",
          globals->options->install_base.c_str());
    }

    std::unique_ptr<blaze_util::IFileMtime> mtime(
        blaze_util::CreateFileMtime());
    string real_install_dir = blaze_util::JoinPath(
        globals->options->install_base, "_embedded_binaries");
    for (const auto &it : globals->extracted_binaries) {
      string path = blaze_util::JoinPath(real_install_dir, it);
      // Check that the file exists and is readable.
      if (blaze_util::IsDirectory(path)) {
        continue;
      }
      if (!blaze_util::CanReadFile(path)) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "Error: corrupt installation: file '%s' missing."
            " Please remove '%s' and try again.",
            path.c_str(), globals->options->install_base.c_str());
      }
      // Check that the timestamp is in the future. A past timestamp would
      // indicate that the file has been tampered with.
      // See ActuallyExtractData().
      bool is_in_future = false;
      if (!mtime.get()->GetIfInDistantFuture(path, &is_in_future)) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "Error: could not retrieve mtime of file '%s'. "
            "Please remove '%s' and try again.",
            path.c_str(), globals->options->install_base.c_str());
      }
      if (!is_in_future) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "Error: corrupt installation: file '%s' "
            "modified.  Please remove '%s' and try again.",
            path.c_str(), globals->options->install_base.c_str());
      }
    }
  }
}

const char *volatile_startup_options[] = {
    "--option_sources=",
    "--max_idle_secs=",
    "--connect_timeout_secs=",
    "--client_debug=",
    NULL,
};

// Returns true if the server needs to be restarted to accommodate changes
// between the two argument lists.
static bool ServerNeedsToBeKilled(const vector<string> &args1,
                                  const vector<string> &args2) {
  // We need not worry about one side missing an argument and the other side
  // having the default value, since this command line is already the
  // canonicalized one that always contains every switch (with default values
  // if it was not present on the real command line). Same applies for argument
  // ordering.
  if (args1.size() != args2.size()) {
    return true;
  }

  for (int i = 0; i < args1.size(); i++) {
    bool option_volatile = false;
    for (const char **candidate = volatile_startup_options; *candidate != NULL;
         candidate++) {
      string candidate_string(*candidate);
      if (args1[i].substr(0, candidate_string.size()) == candidate_string &&
          args2[i].substr(0, candidate_string.size()) == candidate_string) {
        option_volatile = true;
        break;
      }
    }

    if (!option_volatile && args1[i] != args2[i]) {
      return true;
    }
  }

  return false;
}

// Kills the running Blaze server, if any, if the startup options do not match.
static void KillRunningServerIfDifferentStartupOptions(BlazeServer *server) {
  if (!server->Connected()) {
    return;
  }

  string cmdline_path =
      blaze_util::JoinPath(globals->options->output_base, "server/cmdline");
  string joined_arguments;

  // No, /proc/$PID/cmdline does not work, because it is limited to 4K. Even
  // worse, its behavior differs slightly between kernels (in some, when longer
  // command lines are truncated, the last 4 bytes are replaced with
  // "..." + NUL.
  blaze_util::ReadFile(cmdline_path, &joined_arguments);
  vector<string> arguments = blaze_util::Split(joined_arguments, '\0');

  // These strings contain null-separated command line arguments. If they are
  // the same, the server can stay alive, otherwise, it needs shuffle off this
  // mortal coil.
  if (ServerNeedsToBeKilled(arguments, GetArgumentArray())) {
    globals->restart_reason = NEW_OPTIONS;
    PrintWarning(
        "Running %s server needs to be killed, because the "
        "startup options are different.",
        globals->options->product_name.c_str());
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
  if (!ok || !CompareAbsolutePaths(prev_installation,
                                   globals->options->install_base)) {
    if (server->Connected()) {
      server->KillRunningServer();
    }

    globals->restart_reason = NEW_VERSION;
    blaze_util::UnlinkPath(installation_path);
    if (!SymlinkDirectories(globals->options->install_base,
                            installation_path)) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to create installation symlink '%s'",
           installation_path.c_str());
    }

    // Update the mtime of the install base so that cleanup tools can
    // find install bases that haven't been used for a long time
    std::unique_ptr<blaze_util::IFileMtime> mtime(
        blaze_util::CreateFileMtime());
    if (!mtime.get()->SetToNow(globals->options->install_base)) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to set timestamp on '%s'",
           globals->options->install_base.c_str());
    }
  }
}

static void CancelServer() { blaze_server->Cancel(); }

// Performs all I/O for a single client request to the server, and
// shuts down the client (by exit or signal).
static ATTRIBUTE_NORETURN void SendServerRequest(
    const WorkspaceLayout *workspace_layout, BlazeServer *server) {
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
      if (VerboseLogging()) {
        fprintf(stderr, "Server's cwd moved or deleted (%s).\n",
                server_cwd.c_str());
      }
      server->KillRunningServer();
    } else {
      break;
    }
  }

  if (VerboseLogging()) {
    fprintf(stderr, "Connected (server pid=%d).\n", globals->server_pid);
  }

  // Wall clock time since process startup.
  globals->startup_time = GetMillisecondsSinceProcessStart();

  SignalHandler::Get().Install(globals, CancelServer);
  SignalHandler::Get().PropagateSignalOrExit(server->Communicate());
}

// Parse the options, storing parsed values in globals.
static void ParseOptions(int argc, const char *argv[]) {
  string error;
  blaze_exit_code::ExitCode parse_exit_code =
      globals->option_processor->ParseOptions(argc, argv, globals->workspace,
                                              globals->cwd, &error);
  if (parse_exit_code != blaze_exit_code::SUCCESS) {
    die(parse_exit_code, "%s", error.c_str());
  }
  globals->options = globals->option_processor->GetParsedStartupOptions();
}

// Compute the globals globals->cwd and globals->workspace.
static void ComputeWorkspace(const WorkspaceLayout *workspace_layout) {
  globals->cwd = blaze_util::MakeCanonical(blaze_util::GetCwd().c_str());
  if (globals->cwd.empty()) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "blaze_util::MakeCanonical('%s') failed",
         blaze_util::GetCwd().c_str());
  }
  globals->workspace = workspace_layout->GetWorkspace(globals->cwd);
}

// Figure out the base directories based on embedded data, username, cwd, etc.
// Sets globals->options->install_base, globals->options->output_base,
// globals->lockfile, globals->jvm_log_file.
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
    globals->options->install_base =
        GetInstallBase(install_user_root, self_path);
  } else {
    // We call GetInstallBase anyway to populate extracted_binaries and
    // install_md5.
    GetInstallBase("", self_path);
  }

  if (globals->options->output_base.empty()) {
    globals->options->output_base = blaze::GetHashedBaseDir(
        globals->options->output_user_root, globals->workspace);
  }

  const char *output_base = globals->options->output_base.c_str();
  if (!blaze_util::PathExists(globals->options->output_base)) {
    if (!blaze_util::MakeDirectories(globals->options->output_base, 0777)) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "Output base directory '%s' could not be created", output_base);
    }
  } else {
    if (!blaze_util::IsDirectory(globals->options->output_base)) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Error: Output base directory '%s' could not be created. "
          "It exists but is not a directory.",
          output_base);
    }
  }
  if (!blaze_util::CanAccessDirectory(globals->options->output_base)) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Error: Output base directory '%s' must be readable and writable.",
        output_base);
  }
  ExcludePathFromBackup(output_base);

  globals->options->output_base = blaze_util::MakeCanonical(output_base);
  if (globals->options->output_base.empty()) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "blaze_util::MakeCanonical('%s') failed", output_base);
  }

  globals->lockfile =
      blaze_util::JoinPath(globals->options->output_base, "lock");
  globals->jvm_log_file =
      blaze_util::JoinPath(globals->options->output_base, "server/jvm.out");
}

static void CheckEnvironment() {
  if (!blaze::GetEnv("http_proxy").empty()) {
    PrintWarning("ignoring http_proxy in environment.");
    blaze::UnsetEnv("http_proxy");
  }

  if (!blaze::GetEnv("LD_ASSUME_KERNEL").empty()) {
    // Fix for bug: if ulimit -s and LD_ASSUME_KERNEL are both
    // specified, the JVM fails to create threads.  See thread_stack_regtest.
    // This is also provoked by LD_LIBRARY_PATH=/usr/lib/debug,
    // or anything else that causes the JVM to use LinuxThreads.
    PrintWarning("ignoring LD_ASSUME_KERNEL in environment.");
    blaze::UnsetEnv("LD_ASSUME_KERNEL");
  }

  if (!blaze::GetEnv("LD_PRELOAD").empty()) {
    PrintWarning("ignoring LD_PRELOAD in environment.");
    blaze::UnsetEnv("LD_PRELOAD");
  }

  if (!blaze::GetEnv("_JAVA_OPTIONS").empty()) {
    // This would override --host_jvm_args
    PrintWarning("ignoring _JAVA_OPTIONS in environment.");
    blaze::UnsetEnv("_JAVA_OPTIONS");
  }

  if (!blaze::GetEnv("TEST_TMPDIR").empty()) {
    fprintf(stderr,
            "INFO: $TEST_TMPDIR defined: output root default is '%s'.\n",
            globals->options->output_root.c_str());
  }

  // TODO(bazel-team):  We've also seen a failure during loading (creating
  // threads?) when ulimit -Hs 8192.  Characterize that and check for it here.

  // Make the JVM use ISO-8859-1 for parsing its command line because "blaze
  // run" doesn't handle non-ASCII command line arguments. This is apparently
  // the most reliable way to select the platform default encoding.
  blaze::SetEnv("LANG", "en_US.ISO-8859-1");
  blaze::SetEnv("LANGUAGE", "en_US.ISO-8859-1");
  blaze::SetEnv("LC_ALL", "en_US.ISO-8859-1");
  blaze::SetEnv("LC_CTYPE", "en_US.ISO-8859-1");
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
         OptionProcessor *option_processor,
         std::unique_ptr<blaze_util::LogHandler> log_handler) {
  // Logging must be set first to assure no log statements are missed.
  blaze_util::SetLogHandler(std::move(log_handler));

  globals = new GlobalVariables(option_processor);
  blaze::SetupStdStreams();

  // Must be done before command line parsing.
  ComputeWorkspace(workspace_layout);
  globals->binary_path = CheckAndGetBinaryPath(argv[0]);
  ParseOptions(argc, argv);

  debug_log("Debug logging active");

  CheckEnvironment();
  blaze::CreateSecureOutputRoot(globals->options->output_user_root);

  const string self_path = GetSelfPath();
  ComputeBaseDirectories(workspace_layout, self_path);

  blaze_server = static_cast<BlazeServer *>(
      new GrpcBlazeServer(globals->options->connect_timeout_secs));

  globals->command_wait_time = blaze_server->AcquireLock();

  WarnFilesystemType(globals->options->output_base);

  ExtractData(self_path);
  VerifyJavaVersionAndSetJvm();

  blaze_server->Connect();
  EnsureCorrectRunningVersion(blaze_server);
  KillRunningServerIfDifferentStartupOptions(blaze_server);

  if (globals->options->batch) {
    SetScheduling(globals->options->batch_cpu_scheduling,
                  globals->options->io_nice_level);
    StartStandalone(workspace_layout, blaze_server);
  } else {
    SendServerRequest(workspace_layout, blaze_server);
  }
  return 0;
}

static void null_grpc_log_function(gpr_log_func_args *args) {}

GrpcBlazeServer::GrpcBlazeServer(int connect_timeout_secs) {
  connected_ = false;
  connect_timeout_secs_ = connect_timeout_secs;

  gpr_set_log_function(null_grpc_log_function);

  pipe_ = blaze_util::CreatePipe();
  if (pipe_ == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "Couldn't create pipe");
  }
}

GrpcBlazeServer::~GrpcBlazeServer() {
  delete pipe_;
  pipe_ = NULL;
}

bool GrpcBlazeServer::TryConnect(command_server::CommandServer::Stub *client) {
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::seconds(connect_timeout_secs_));

  command_server::PingRequest request;
  command_server::PingResponse response;
  request.set_cookie(request_cookie_);

  debug_log("Trying to connect to server (timeout: %d secs)...",
            connect_timeout_secs_);
  grpc::Status status = client->Ping(&context, request, &response);

  if (!status.ok() || response.cookie() != response_cookie_) {
    debug_log("Connection to server failed: %s",
              status.error_message().c_str());
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

  std::shared_ptr<grpc::Channel> channel(
      grpc::CreateChannel(port, grpc::InsecureChannelCredentials()));
  std::unique_ptr<command_server::CommandServer::Stub> client(
      command_server::CommandServer::NewStub(channel));

  if (!TryConnect(client.get())) {
    return false;
  }

  this->client_ = std::move(client);
  connected_ = true;

  globals->server_pid = GetServerPid(server_dir);
  if (globals->server_pid <= 0) {
    fprintf(stderr,
            "Can't get PID of existing server (server dir=%s). "
            "Shutting it down and starting a new one...\n",
            server_dir.c_str());
    // This means that we have a server we could connect to but without a PID
    // file, which in turn means that something went wrong before. Kill the
    // server so that we can start with as clean a slate as possible. This may
    // happen if someone (e.g. a client or server that's very old and uses an
    // AF_UNIX socket instead of gRPC) deletes the server.pid.txt file.
    KillRunningServer();
    // Then wait until it actually dies
    do {
      auto next_attempt_time(std::chrono::system_clock::now() +
                             std::chrono::milliseconds(1000));
      std::this_thread::sleep_until(next_attempt_time);
    } while (TryConnect(client_.get()));

    return false;
  }

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
      pdie(blaze_exit_code::INTERNAL_ERROR,
           "Cannot communicate with cancel thread");
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
    fprintf(stderr, "\nCould not interrupt server (%s)\n\n",
            status.error_message().c_str());
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
  std::unique_ptr<grpc::ClientReader<command_server::RunResponse>> reader(
      client_->Run(&context, request));

  while (reader->Read(&response)) {
  }

  // Kill the server process for good measure (if we know the server PID)
  if (globals->server_pid > 0 &&
      VerifyServerProcess(globals->server_pid, globals->options->output_base,
                          globals->options->install_base)) {
    KillServerProcess(globals->server_pid);
  }

  connected_ = false;
}

unsigned int GrpcBlazeServer::Communicate() {
  assert(connected_);

  vector<string> arg_vector;
  string command = globals->option_processor->GetCommand();
  if (command != "") {
    arg_vector.push_back(command);
    AddLoggingArgs(&arg_vector);
  }

  globals->option_processor->GetCommandArguments(&arg_vector);

  command_server::RunRequest request;
  request.set_cookie(request_cookie_);
  request.set_block_for_lock(globals->options->block_for_lock);
  request.set_client_description("pid=" + blaze::GetProcessIdAsString());
  for (const string &arg : arg_vector) {
    request.add_arg(arg);
  }
  if (globals->options->invocation_policy != NULL &&
      strlen(globals->options->invocation_policy) > 0) {
    request.set_invocation_policy(globals->options->invocation_policy);
  }

  grpc::ClientContext context;
  command_server::RunResponse response;
  std::unique_ptr<grpc::ClientReader<command_server::RunResponse>> reader(
      client_->Run(&context, request));

  // Release the server lock because the gRPC handles concurrent clients just
  // fine. Note that this may result in two "waiting for other client" messages
  // (one during server startup and one emitted by the server)
  blaze::ReleaseLock(&blaze_lock_);

  std::thread cancel_thread(&GrpcBlazeServer::CancelThread, this);
  bool command_id_set = false;
  bool pipe_broken = false;
  int exit_code = -1;
  bool finished = false;
  bool finished_warning_emitted = false;

  while (reader->Read(&response)) {
    if (finished && !finished_warning_emitted) {
      fprintf(stderr, "\nServer returned messages after reporting exit code\n");
      finished_warning_emitted = true;
    }

    if (response.cookie() != response_cookie_) {
      fprintf(stderr, "\nServer response cookie invalid, exiting\n");
      return blaze_exit_code::INTERNAL_ERROR;
    }

    const char *broken_pipe_name = nullptr;

    if (response.finished()) {
      exit_code = response.exit_code();
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
      fprintf(stderr, "\nCannot write to %s; exiting...\n\n", broken_pipe_name);
      Cancel();
    }

    if (!command_id_set && response.command_id().size() > 0) {
      std::unique_lock<std::mutex> lock(cancel_thread_mutex_);
      command_id_ = response.command_id();
      command_id_set = true;
      SendAction(CancelThreadAction::COMMAND_ID_RECEIVED);
    }
  }

  SendAction(CancelThreadAction::JOIN);
  cancel_thread.join();

  grpc::Status status = reader->Finish();
  if (!status.ok()) {
    fprintf(stderr,
            "\nServer terminated abruptly "
            "(error code: %d, error message: '%s', log file: '%s')\n\n",
            status.error_code(), status.error_message().c_str(),
            globals->jvm_log_file.c_str());
    return GetExitCodeForAbruptExit(*globals);
  } else if (!finished) {
    fprintf(stderr,
            "\nServer finished RPC without an explicit exit code "
            "(log file: '%s')\n\n",
            globals->jvm_log_file.c_str());
    return GetExitCodeForAbruptExit(*globals);
  }

  // We'll exit with exit code SIGPIPE on Unixes due to PropagateSignalOnExit()
  return pipe_broken ? blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR : exit_code;
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
