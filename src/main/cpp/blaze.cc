// Copyright 2014 Google Inc. All rights reserved.
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

#include <assert.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>
#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "src/main/cpp/blaze_exit_code.h"
#include "src/main/cpp/blaze_startup_options.h"
#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/option_processor.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/md5.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"
#include "archive.h"
#include "archive_entry.h"

using blaze_util::Md5Digest;
using blaze_util::die;
using blaze_util::pdie;
using std::set;
using std::vector;

// This should already be defined in sched.h, but it's not.
#ifndef SCHED_BATCH
#define SCHED_BATCH 3
#endif

namespace blaze {

extern char **environ;

////////////////////////////////////////////////////////////////////////
// Global Variables

// The reason for a blaze server restart.
// Keep in sync with logging.proto
enum RestartReason {
  NO_RESTART = 0,
  NO_DAEMON,
  NEW_VERSION,
  NEW_OPTIONS
};

struct GlobalVariables {
  // Used to make concurrent invocations of this program safe.
  string lockfile;  // = <output_base>/lock
  int lockfd;

  string jvm_log_file;  // = <output_base>/server/jvm.out

  string cwd;

  // The nearest enclosing workspace directory, starting from cwd.
  // If not under a workspace directory, this is equal to cwd.
  string workspace;

  // Option processor responsible for parsing RC files and converting them into
  // the argument list passed on to the server.
  OptionProcessor option_processor;

  pid_t server_pid;

  volatile sig_atomic_t sigint_count;

  // The number of the last received signal that should cause the client
  // to shutdown.  This is saved so that the client's WTERMSIG can be set
  // correctly.  (Currently only SIGPIPE uses this mechanism.)
  volatile sig_atomic_t received_signal;

  // Contains the relative paths of all the files in the attached zip, and is
  // populated during GetInstallDir().
  vector<string> extracted_binaries;

  // Parsed startup options
  BlazeStartupOptions options;

  // The time in ms the launcher spends before sending the request to the Blaze
  uint64 startup_time;

  // The time spent on extracting the new blaze version
  // This is part of startup_time
  uint64 extract_data_time;

  // The time in ms if a command had to wait on a busy Blaze server process
  // This is part of startup_time
  uint64 command_wait_time;

  RestartReason restart_reason;

  // Absolute path of the blaze binary
  string binary_path;
};

static GlobalVariables *globals;

void InitGlobals() {
  globals = new GlobalVariables;
  globals->sigint_count = 0;
  globals->startup_time = 0;
  globals->extract_data_time = 0;
  globals->command_wait_time = 0;
  globals->restart_reason = NO_RESTART;
}

////////////////////////////////////////////////////////////////////////
// Logic


// Returns the canonical form of the base dir given a root and a hashable
// string. The resulting dir is composed of the root + md5(hashable)
static string GetHashedBaseDir(const string &root,
                               const string &hashable) {
  unsigned char buf[Md5Digest::kDigestLength];
  Md5Digest digest;
  digest.Update(hashable.data(), hashable.size());
  digest.Finish(buf);
  return root + "/" + digest.String();
}

// Returns the install base (the root concatenated with the contents of the file
// 'install_base_key' contained as a ZIP entry in the Blaze binary); as a side
// effect, it also populates the extracted_binaries global variable.
static string GetInstallBase(const string &root, const string &self_path) {
  string key_file = "install_base_key";
  struct archive *blaze_zip = archive_read_new();
  archive_read_support_format_zip(blaze_zip);
  int retval = archive_read_open_filename(blaze_zip, self_path.c_str(), 10240);
  if (retval != ARCHIVE_OK) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to open %s as a zip file: (%d) %s",
        globals->options.GetProductName().c_str(), archive_errno(blaze_zip),
        archive_error_string(blaze_zip));
  }

  struct archive_entry *entry;
  string install_base_key;
  while (archive_read_next_header(blaze_zip, &entry) == ARCHIVE_OK) {
    string pathname = archive_entry_pathname(entry);
    globals->extracted_binaries.push_back(pathname);

    if (key_file == pathname) {
      const int size = 32;
      char buf[size];
      int bytesRead = archive_read_data(blaze_zip, &buf, size);
      if (bytesRead < 0) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "\nFailed to extract install_base_key: (%d) %s",
            archive_errno(blaze_zip), archive_error_string(blaze_zip));
      }
      if (bytesRead < 32) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "\nFailed to extract install_base_key: file too short");
      }
      install_base_key = string(buf, bytesRead);
    }
  }
  retval = archive_read_free(blaze_zip);
  if (retval != ARCHIVE_OK) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to close install_base_key's containing zip file");
  }

  return root + "/" + install_base_key;
}

// Escapes colons by replacing them with '_C' and underscores by replacing them
// with '_U'. E.g. "name:foo_bar" becomes "name_Cfoo_Ubar"
static string EscapeForOptionSource(const string& input) {
  string result = input;
  blaze_util::Replace("_", "_U", &result);
  blaze_util::Replace(":", "_C", &result);
  return result;
}

// Returns the JVM command argument array.
static vector<string> GetArgumentArray() {
  vector<string> result;

  // e.g. A Blaze server process running in ~/src/build_root (where there's a
  // ~/src/build_root/WORKSPACE file) will appear in ps(1) as "blaze(src)".
  string workspace =
      blaze_util::Basename(blaze_util::Dirname(globals->workspace));
  string product = globals->options.GetProductName();
  blaze_util::ToLower(&product);
  result.push_back(product + "(" + workspace + ")");
  if (globals->options.batch) {
    result.push_back("-client");
    result.push_back("-Xms256m");
    result.push_back("-XX:NewRatio=4");
  } else {
    result.push_back("-server");
  }

  result.push_back("-XX:+HeapDumpOnOutOfMemoryError");
  string heap_crash_path = globals->options.output_base;
  result.push_back("-XX:HeapDumpPath=" + heap_crash_path);

  result.push_back("-Xverify:none");

  // Add JVM arguments particular to building blaze64 and particular JVM
  // versions.
  string error;
  blaze_exit_code::ExitCode jvm_args_exit_code =
      globals->options.AddJVMArguments(globals->options.GetHostJavabase(),
                                       &result, &error);
  if (jvm_args_exit_code != blaze_exit_code::SUCCESS) {
    die(jvm_args_exit_code, "%s", error.c_str());
  }

  // We put all directories on the java.library.path that contain .so files.
  string java_library_path = "-Djava.library.path=";
  string real_install_dir = blaze_util::JoinPath(globals->options.install_base,
                                                 "_embedded_binaries");
  bool first = true;
  for (const auto& it : globals->extracted_binaries) {
    if (IsSharedLibrary(it)) {
      if (!first) {
        java_library_path += ":";
      }
      first = false;
      java_library_path += blaze_util::JoinPath(real_install_dir,
                                                blaze_util::Dirname(it));
    }
  }
  result.push_back(java_library_path);

  // Force use of latin1 for file names.
  result.push_back("-Dfile.encoding=ISO-8859-1");

  if (globals->options.host_jvm_debug) {
    fprintf(stderr,
            "Running host JVM under debugger (listening on TCP port 5005).\n");
    // Start JVM so that it listens for a connection from a
    // JDWP-compliant debugger:
    result.push_back("-Xdebug");
    result.push_back("-Xrunjdwp:transport=dt_socket,server=y,address=5005");
  }

  blaze_util::SplitQuotedStringUsing(globals->options.host_jvm_args, ' ',
                                     &result);

  result.push_back("-jar");
  result.push_back(blaze_util::JoinPath(real_install_dir,
                                        globals->extracted_binaries[0]));

  if (!globals->options.batch) {
    result.push_back("--max_idle_secs");
    result.push_back(std::to_string(globals->options.max_idle_secs));
  } else {
    // --batch must come first in the arguments to Java main() because
    // the code expects it to be at args[0] if it's been set.
    result.push_back("--batch");
  }
  result.push_back("--install_base=" + globals->options.install_base);
  result.push_back("--output_base=" + globals->options.output_base);
  result.push_back("--workspace_directory=" + globals->workspace);
  if (!globals->options.skyframe.empty()) {
    result.push_back("--skyframe=" + globals->options.skyframe);
  }
  if (globals->options.blaze_cpu) {
    result.push_back("--blaze_cpu=true");
  }

  if (globals->options.allow_configurable_attributes) {
    result.push_back("--allow_configurable_attributes");
  }
  if (globals->options.watchfs) {
    result.push_back("--watchfs");
  }
  if (globals->options.fatal_event_bus_exceptions) {
    result.push_back("--fatal_event_bus_exceptions");
  } else {
    result.push_back("--nofatal_event_bus_exceptions");
  }
  if (globals->options.webstatus_port) {
    result.push_back("--use_webstatusserver=" + \
                     std::to_string(globals->options.webstatus_port));
  }

  // This is only for Blaze reporting purposes; the real interpretation of the
  // jvm flags occurs when we set up the java command line.
  if (globals->options.host_jvm_debug) {
    result.push_back("--host_jvm_debug");
  }
  if (!globals->options.host_jvm_profile.empty()) {
    result.push_back("--host_jvm_profile=" + globals->options.host_jvm_profile);
  }
  if (!globals->options.host_jvm_args.empty()) {
    result.push_back("--host_jvm_args=" + globals->options.host_jvm_args);
  }
  globals->options.AddExtraOptions(&result);

  // The option sources are transmitted in the following format:
  // --option_sources=option1:source1:option2:source2:...
  string option_sources = "--option_sources=";
  first = true;
  for (const auto& it : globals->options.option_sources) {
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

// Add commom command options for logging to the given argument array.
static void AddLoggingArgs(vector<string>* args) {
  args->push_back("--startup_time=" + std::to_string(globals->startup_time));
  if (globals->command_wait_time != 0) {
    args->push_back("--command_wait_time=" +
                    std::to_string(globals->command_wait_time));
  }
  if (globals->extract_data_time != 0) {
    args->push_back("--extract_data_time=" +
                    std::to_string(globals->extract_data_time));
  }
  if (globals->restart_reason != NO_RESTART) {
    const char *reasons[] = {
        "no_restart", "no_daemon", "new_version", "new_options"
    };
    args->push_back(
        string("--restart_reason=") + reasons[globals->restart_reason]);
  }
  args->push_back(
      string("--binary_path=") + globals->binary_path);
}


// Join the elements of the specified array with NUL's (\0's), akin to the
// format of /proc/$PID/cmdline.
string GetArgumentString(const vector<string>& argument_array) {
  string result;
  blaze_util::JoinStrings(argument_array, '\0', &result);
  return result;
}

// Causes the current process to become a daemon (i.e. a child of
// init, detached from the terminal, in its own session.)  We don't
// change cwd, though.
static void Daemonize(int socket) {
  // Don't call die() or exit() in this function; we're already in a
  // child process so it won't work as expected.  Just don't do
  // anything that can possibly fail. :)

  signal(SIGHUP, SIG_IGN);
  if (fork() > 0) {
    // This second fork is required iff there's any chance cmd will
    // open an specific tty explicitly, e.g., open("/dev/tty23"). If
    // not, this fork can be removed.
    _exit(blaze_exit_code::SUCCESS);
  }

  setsid();

  close(0);
  close(1);
  close(2);
  close(socket);

  open("/dev/null", O_RDONLY);  // stdin
  // stdout:
  if (open(globals->jvm_log_file.c_str(),
           O_WRONLY | O_CREAT | O_TRUNC, 0666) == -1) {
    // In a daemon, no-one can hear you scream.
    open("/dev/null", O_WRONLY);
  }
  dup(STDOUT_FILENO);  // stderr (2>&1)

  // Keep server from inheriting a useless fd.
  // The file lock was already lost at fork().
  close(globals->lockfd);
}

// Do a chdir into the workspace, and die if it fails.
static void GoToWorkspace() {
  if (BlazeStartupOptions::InWorkspace(globals->workspace) &&
      chdir(globals->workspace.c_str()) != 0) {
    pdie(blaze_exit_code::INTERNAL_ERROR,
         "chdir() into %s failed", globals->workspace.c_str());
  }
}

// Check the java version if a java version specification is bundled. On
// success,
// return the executable path of the java command.
static string VerifyJavaVersionAndGetJvm() {
  string exe = globals->options.GetJvm();

  string version_spec_file = blaze_util::JoinPath(
      blaze_util::JoinPath(globals->options.install_base, "_embedded_binaries"),
      "java.version");
  string version_spec = "";
  if (ReadFile(version_spec_file, &version_spec)) {
    blaze_util::StripWhitespace(&version_spec);
    // A version specification is given, get version of java.
    string jvm_version = GetJvmVersion(exe);

    // Compare that jvm_version is found and at least the one specified.
    if (jvm_version.size() == 0) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Java version not detected while at least %s is needed.\n"
          "Please set JAVA_HOME.", version_spec.c_str());
    } else if (!CheckJavaVersionIsAtLeast(jvm_version, version_spec)) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "Java version is %s while at least %s is needed.\n"
           "Please set JAVA_HOME.",
           jvm_version.c_str(), version_spec.c_str());
    }
  }

  return exe;
}

// Starts the Blaze server.  Returns a readable fd connected to the server.
// This is currently used only to detect liveness.
static int StartServer(int socket) {
  vector<string> jvm_args_vector = GetArgumentArray();
  string argument_string = GetArgumentString(jvm_args_vector);

  // Write the cmdline argument string to the server dir. If we get to this
  // point, there is no server running, so we don't overwrite the cmdline file
  // for the existing server. If might be that the server dies and the cmdline
  // file stays there, but that is not a problem, since we always check the
  // server, too.
  WriteFile(argument_string, globals->options.output_base + "/server/cmdline");

  // unless we restarted for a new-version, mark this as initial start
  if (globals->restart_reason == NO_RESTART) {
    globals->restart_reason = NO_DAEMON;
  }

  // Computing this path may report a fatal error, so do it before forking.
  string exe = VerifyJavaVersionAndGetJvm();

  // Go to the workspace before we daemonize, so
  // we can still print errors to the terminal.
  GoToWorkspace();

  int fds[2];
  if (pipe(fds)) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "pipe creation failed");
  }
  int child = fork();
  if (child == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "fork() failed");
  } else if (child > 0) {  // we're the parent
    close(fds[1]);  // parent keeps only the reading side
    return fds[0];
  } else {
    close(fds[0]);  // child keeps only the writing side
  }

  Daemonize(socket);
  ExecuteProgram(exe, jvm_args_vector);
  pdie(blaze_exit_code::INTERNAL_ERROR, "execv of '%s' failed", exe.c_str());
}

static bool KillRunningServerIfAny();

// Replace this process with blaze in standalone/batch mode.
// The batch mode blaze process handles the command and exits.
//
// This function passes the commands array to the blaze process.
// This array should start with a command ("build", "info", etc.).
static void StartStandalone() {
  KillRunningServerIfAny();

  // Wall clock time since process startup.
  globals->startup_time = ProcessClock() / 1000000LL;

  if (VerboseLogging()) {
    fprintf(stderr, "Starting %s in batch mode.\n",
            globals->options.GetProductName().c_str());
  }
  string command = globals->option_processor.GetCommand();
  vector<string> command_arguments;
  globals->option_processor.GetCommandArguments(&command_arguments);

  if (!command_arguments.empty() && command == "shutdown") {
    string product = globals->options.GetProductName();
    blaze_util::ToLower(&product);
    fprintf(stderr,
            "WARNING: Running command \"shutdown\" in batch mode.  Batch mode "
            "is triggered\nwhen not running %s within a workspace. If you "
            "intend to shutdown an\nexisting %s server, run \"%s "
            "shutdown\" from the directory where\nit was started.\n",
            globals->options.GetProductName().c_str(),
            globals->options.GetProductName().c_str(), product.c_str());
  }
  vector<string> jvm_args_vector = GetArgumentArray();
  if (command != "") {
    jvm_args_vector.push_back(command);
    AddLoggingArgs(&jvm_args_vector);
  }

  jvm_args_vector.insert(jvm_args_vector.end(),
                         command_arguments.begin(),
                         command_arguments.end());

  GoToWorkspace();

  string exe = VerifyJavaVersionAndGetJvm();
  ExecuteProgram(exe, jvm_args_vector);
  pdie(blaze_exit_code::INTERNAL_ERROR, "execv of '%s' failed", exe.c_str());
}

// Like connect(2), but uses the AF_UNIX address denoted by socket_file,
// resolving symbolic links.  (The server may make "socket_file" a
// symlink, to avoid ENAMETOOLONG, in which case the client must
// resolve it in userspace before connecting.)
static int Connect(int socket, const string &socket_file) {
  struct sockaddr_un addr;
  addr.sun_family = AF_UNIX;

  char *resolved_path = realpath(socket_file.c_str(), NULL);
  if (resolved_path != NULL) {
    strncpy(addr.sun_path, resolved_path, sizeof addr.sun_path);
    addr.sun_path[sizeof addr.sun_path - 1] = '\0';
    free(resolved_path);
    sockaddr *paddr = reinterpret_cast<sockaddr *>(&addr);
    return connect(socket, paddr, sizeof addr);
  } else if (errno == ENOENT) {  // No socket means no server to connect to
    errno = ECONNREFUSED;
    return -1;
  } else {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "realpath('%s') failed", socket_file.c_str());
  }
}

// Write the contents of file_name to stream.
static void WriteFileToStreamOrDie(FILE *stream, const char *file_name) {
  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "opening %s failed", file_name);
  }
  char buffer[255];
  int num_read;
  while ((num_read = fread(buffer, 1, sizeof buffer, fp)) > 0) {
    if (ferror(fp)) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to read from '%s'", file_name);
    }
    fwrite(buffer, 1, num_read, stream);
  }
  fclose(fp);
}

// Connects to the Blaze server, returning the socket, or -1 if no
// server is running and !start.  If start, attempts to start a new
// server, and exits on failure.
static int ConnectToServer(bool start) {
  int s = socket(PF_UNIX, SOCK_STREAM, 0);
  if (s == -1)  {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "can't create AF_UNIX socket");
  }

  string server_dir = globals->options.output_base + "/server";

  // The server dir has the socket, so we don't allow access by other
  // users.
  if (MakeDirectories(server_dir, 0700) == -1) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "server directory '%s' could not be created", server_dir.c_str());
  }

  string socket_file = server_dir + "/server.socket";

  if (Connect(s, socket_file) == 0) {
    return s;
  }
  if (!start) {
    return -1;
  } else {
    SetScheduling(
        globals->options.batch_cpu_scheduling,
        globals->options.io_nice_level);

    int fd = StartServer(s);
    if (fcntl(fd, F_SETFL, O_NONBLOCK | fcntl(fd, F_GETFL))) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "Failed: fcntl to enable O_NONBLOCK on pipe");
    }
    // Give the server one minute to start up.
    for (int ii = 0; ii < 600; ++ii) {  // 60s; enough time to connect
                                        // with debugger
      if (Connect(s, socket_file) == 0) {
        if (ii) {
          fputc('\n', stderr);
          fflush(stderr);
        }
        return s;
      }
      fputc('.', stderr);
      fflush(stderr);
      poll(NULL, 0, 100);  // sleep 100ms.  (usleep(3) is obsolete.)
      char c;
      if (read(fd, &c, 1) != -1 || errno != EAGAIN) {
        fprintf(stderr, "\nunexpected pipe read status: %s\n"
            "Server presumed dead. Now printing '%s':\n",
            strerror(errno), globals->jvm_log_file.c_str());
        WriteFileToStreamOrDie(stderr, globals->jvm_log_file.c_str());
        exit(blaze_exit_code::INTERNAL_ERROR);
      }
    }
    die(blaze_exit_code::INTERNAL_ERROR,
        "\nError: couldn't connect to server at '%s' after 60 seconds.",
        socket_file.c_str());
  }
}


// Kills the specified running Blaze server.
static void KillRunningServer(pid_t server_pid) {
  fprintf(stderr, "Sending SIGTERM to previous %s server (pid=%d)... ",
          globals->options.GetProductName().c_str(), server_pid);
  fflush(stderr);
  for (int ii = 0; ii < 100; ++ii) {  // wait up to 10s
    if (kill(server_pid, SIGTERM) == -1) {
      fprintf(stderr, "done.\n");
      return;  // Ding! Dong! The witch is dead!
    }
    poll(NULL, 0, 100);  // sleep 100ms.  (usleep(3) is obsolete.)
  }

  // If the previous attempt did not suceeded, kill the whole group.
  fprintf(stderr,
          "Sending SIGKILL to previous %s server process group (pid=%d)... ",
          globals->options.GetProductName().c_str(), server_pid);
  fflush(stderr);
  killpg(server_pid, SIGKILL);
  if (kill(server_pid, 0) == -1) {  // (probe)
    fprintf(stderr, "could not be killed.\n");  // task state 'Z' or 'D'?
    exit(1);  // TODO(bazel-team): confirm whether this is an internal error.
  } else {
    fprintf(stderr, "killed.\n");
  }
}


// Kills the running Blaze server, if any.  Finds the pid from the socket.
static bool KillRunningServerIfAny() {
  int socket = ConnectToServer(false);
  if (socket != -1) {
    KillRunningServer(GetPeerProcessId(socket));
    return true;
  }
  return false;
}


// Calls fsync() on the file (or directory) specified in 'file_path'.
// pdie()'s if syncing fails.
static void SyncFile(const char *file_path) {
  // fsync always fails on Cygwin with "Permission denied" for some reason.
#ifndef __CYGWIN__
  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "failed to open '%s' for syncing", file_path);
  }
  if (fsync(fd) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "failed to sync '%s'", file_path);
  }
  close(fd);
#endif
}

// Walks the temporary directory recursively and collects full file paths.
static void CollectExtractedFiles(const string &dir_path, vector<string> &files) {
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(dir_path.c_str())) == NULL) {
    die(blaze_exit_code::INTERNAL_ERROR, "opendir failed");
  }

  while ((ent = readdir(dir)) != NULL) {
    if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) {
      continue;
    }

    string filename(blaze_util::JoinPath(dir_path, ent->d_name));
    bool is_directory;
    if (ent->d_type == DT_UNKNOWN) {
      struct stat buf;
      if (lstat(filename.c_str(), &buf) == -1) {
        die(blaze_exit_code::INTERNAL_ERROR, "stat failed");
      }
      is_directory = S_ISDIR(buf.st_mode);
    } else {
      is_directory = (ent->d_type == DT_DIR);
    }

    if (is_directory) {
      CollectExtractedFiles(filename, files);
    } else {
      files.push_back(filename);
    }
  }

  closedir(dir);
}

// Actually extracts the embedded data files into the tree whose root
// is 'embedded_binaries'.
static void ActuallyExtractData(const string &argv0,
                                const string &embedded_binaries) {
  if (MakeDirectories(embedded_binaries, 0777) == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR,
         "couldn't create '%s'", embedded_binaries.c_str());
  }

  fprintf(stderr, "Extracting %s installation...\n",
          globals->options.GetProductName().c_str());

  struct archive *blaze_zip = archive_read_new();
  archive_read_support_format_zip(blaze_zip);
  int retval = archive_read_open_filename(blaze_zip, argv0.c_str(), 10240);
  if (retval != ARCHIVE_OK) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to open %s as a zip file",
        globals->options.GetProductName().c_str());
  }

  struct archive_entry *entry;
  string install_base_key;
  while (archive_read_next_header(blaze_zip, &entry) == ARCHIVE_OK) {
    string path = blaze_util::JoinPath(
        embedded_binaries, archive_entry_pathname(entry));
    if (MakeDirectories(blaze_util::Dirname(path), 0777) == -1) {
      pdie(blaze_exit_code::INTERNAL_ERROR,
           "couldn't create '%s'", path.c_str());
    }
    int fd = open(path.c_str(), O_CREAT | O_WRONLY, 0755);
    if (fd < 0) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "\nFailed to open extraction file: %s", strerror(errno));
    }

    const void *buf;
    size_t size;
    int64_t offset;
    while (true) {
      retval = archive_read_data_block(blaze_zip, &buf, &size, &offset);
      if (retval == ARCHIVE_EOF) {
        break;
      } else if (retval != ARCHIVE_OK) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "\nFailed to extract data from %s zip: (%d) %s",
            globals->options.GetProductName().c_str(), archive_errno(blaze_zip),
            archive_error_string(blaze_zip));
      }
      if (write(fd, buf, size) != size) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "\nError writing zipped file to %s", path.c_str());
      }
    }
    if (close(fd) != 0) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "\nCould not close file %s", path.c_str());
    }
  }
  retval = archive_read_free(blaze_zip);
  if (retval != ARCHIVE_OK) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "\nFailed to close %s zip", globals->options.GetProductName().c_str());
  }

  const time_t TEN_YEARS_IN_SEC = 3600 * 24 * 365 * 10;
  time_t future_time = time(NULL) + TEN_YEARS_IN_SEC;

  // Set the timestamps of the extracted files to the future and make sure (or
  // at least as sure as we can...) that the files we have written are actually
  // on the disk.

  vector<string> extracted_files;
  CollectExtractedFiles(embedded_binaries, extracted_files);

  set<string> synced_directories;
  for (vector<string>::iterator it = extracted_files.begin(); it != extracted_files.end(); it++) {

    const char *extracted_path = it->c_str();

    // Set the time to a distantly futuristic value so we can observe tampering.
    // Note that keeping the default timestamp set by unzip (1970-01-01) and using
    // that to detect tampering is not enough, because we also need the timestamp
    // to change between Blaze releases so that the metadata cache knows that
    // the files may have changed. This is important for actions that use
    // embedded binaries as artifacts.
    struct utimbuf times = { future_time, future_time };
    if (utime(extracted_path, &times) == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to set timestamp on '%s'", extracted_path);
    }

    SyncFile(extracted_path);

    string directory = blaze_util::Dirname(extracted_path);

    // Now walk up until embedded_binaries and sync every directory in between.
    // synced_directories is used to avoid syncing the same directory twice.
    // The !directory.empty() and directory != "/" conditions are not strictly
    // needed, but it makes this loop more robust, because otherwise, if due to
    // some glitch, directory was not under embedded_binaries, it would get
    // into an infinite loop.
    while (directory != embedded_binaries &&
           synced_directories.count(directory) == 0 &&
           !directory.empty() &&
           directory != "/") {
      SyncFile(directory.c_str());
      synced_directories.insert(directory);
      directory = blaze_util::Dirname(directory);
    }
  }

  SyncFile(embedded_binaries.c_str());
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
  struct stat buf;
  if (stat(globals->options.install_base.c_str(), &buf) == -1) {
    uint64 st = MonotonicClock();
    // Work in a temp dir to avoid races.
    string tmp_install = globals->options.install_base + ".tmp." +
        std::to_string(getpid());
    string tmp_binaries = tmp_install + "/_embedded_binaries";
    ActuallyExtractData(self_path, tmp_binaries);

    uint64 et = MonotonicClock();
    globals->extract_data_time = (et - st) / 1000000LL;

    // Now rename the completed installation to its final name. If this
    // fails due to an ENOTEMPTY then we assume another good
    // installation snuck in before us.
    if (rename(tmp_install.c_str(), globals->options.install_base.c_str()) == -1
        && errno != ENOTEMPTY) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "install base directory '%s' could not be renamed into place",
           tmp_install.c_str());
    }
  } else {
    if (!S_ISDIR(buf.st_mode)) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Error: Install base directory '%s' could not be created. "
          "It exists but is not a directory.",
          globals->options.install_base.c_str());
    }

    const time_t time_now = time(NULL);
    string real_install_dir = blaze_util::JoinPath(
        globals->options.install_base,
        "_embedded_binaries");
    for (const auto& it : globals->extracted_binaries) {
      string path = blaze_util::JoinPath(real_install_dir, it);
      // Check that the file exists and is readable.
      if (stat(path.c_str(), &buf) == -1) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "Error: corrupt installation: file '%s' missing."
            " Please remove '%s' and try again.",
            path.c_str(), globals->options.install_base.c_str());
      }
      // Check that the timestamp is in the future. A past timestamp would indicate
      // that the file has been tampered with. See ActuallyExtractData().
      if (buf.st_mtime <= time_now) {
        die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
            "Error: corrupt installation: file '%s' "
            "modified.  Please remove '%s' and try again.",
            path.c_str(), globals->options.install_base.c_str());
      }
    }
  }
}

// Returns true if the server needs to be restarted to accommodate changes
// between the two argument lists.
static bool ServerNeedsToBeKilled(const vector<string>& args1,
                                  const vector<string>& args2) {
  // We need not worry about one side missing an argument and the other side
  // having the default value, since this command line is already the
  // canonicalized one that always contains every switch (with default values
  // if it was not present on the real command line). Same applies for argument
  // ordering.
  if (args1.size() != args2.size()) {
    return true;
  }

  for (int i = 0; i < args1.size(); i++) {
    string option_sources = "--option_sources=";
    if (args1[i].substr(0, option_sources.size()) == option_sources &&
        args2[i].substr(0, option_sources.size()) == option_sources) {
      continue;
    }

    if (args1[i] !=args2[i]) {
      return true;
    }

    if (args1[i] == "--max_idle_secs") {
      // Skip the argument of --max_idle_secs.
      i++;
    }
  }

  return false;
}

// Kills the running Blaze server, if any, if the startup options do not match.
static void KillRunningServerIfDifferentStartupOptions() {
  int socket = ConnectToServer(false);

  if (socket == -1) {
    return;
  }

  pid_t server_pid = GetPeerProcessId(socket);
  close(socket);
  string cmdline_path = globals->options.output_base + "/server/cmdline";
  string joined_arguments;

  // No, /proc/$PID/cmdline does not work, because it is limited to 4K. Even
  // worse, its behavior differs slightly between kernels (in some, when longer
  // command lines are truncated, the last 4 bytes are replaced with
  // "..." + NUL.
  ReadFile(cmdline_path, &joined_arguments);
  vector<string> arguments = blaze_util::Split(joined_arguments, '\0');

  // These strings contain null-separated command line arguments. If they are
  // the same, the server can stay alive, otherwise, it needs shuffle off this
  // mortal coil.
  if (ServerNeedsToBeKilled(arguments, GetArgumentArray())) {
    globals->restart_reason = NEW_OPTIONS;
    fprintf(stderr,
            "WARNING: Running %s server needs to be killed, because the "
            "startup options are different.\n",
            globals->options.GetProductName().c_str());
    KillRunningServer(server_pid);
  }
}


// Kills the old running server if it is not the same version as us,
// dealing with various combinations of installation scheme
// (installation symlink and older MD5_MANIFEST contents).
// This function requires that the installation be complete, and the
// server lock acquired.
static void EnsureCorrectRunningVersion() {
  // Read the previous installation's semaphore symlink in output_base. If the
  // target dirs don't match, or if the symlink was not present, then kill any
  // running servers. Lastly, symlink to our installation so others know which
  // installation is running.
  string installation_path = globals->options.output_base + "/install";
  char prev_installation[PATH_MAX + 1] = "";  // NULs the whole array
  if (readlink(installation_path.c_str(),
               prev_installation, PATH_MAX) == -1 ||
      prev_installation != globals->options.install_base) {
    if (KillRunningServerIfAny()) {
      globals->restart_reason = NEW_VERSION;
    }
    unlink(installation_path.c_str());
    if (symlink(globals->options.install_base.c_str(),
                installation_path.c_str())) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to create installation symlink '%s'",
           installation_path.c_str());
    }
    const time_t time_now = time(NULL);
    struct utimbuf times = { time_now, time_now };
    if (utime(globals->options.install_base.c_str(), &times) == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "failed to set timestamp on '%s'",
           globals->options.install_base.c_str());
    }
  }
}


// A signal-safe version of fprintf(stderr, ...).
//
// WARNING: any output from the blaze client may be interleaved
// with output from the blaze server.  In --curses mode,
// the Blaze server often erases the previous line of output.
// So, be sure to end each such message with TWO newlines,
// otherwise it may be erased by the next message from the
// Blaze server.
// Also, it's a good idea to start each message with a newline,
// in case the Blaze server has written a partial line.
static void sigprintf(const char *format, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, format);
  int r = vsnprintf(buf, sizeof buf, format, ap);
  va_end(ap);
  write(STDERR_FILENO, buf, r);
}


// Signal handler.
static void handler(int signum) {
  // A defensive measure:
  if (kill(globals->server_pid, 0) == -1 && errno == ESRCH) {
    sigprintf("\n%s server has died; client exiting.\n\n",
              globals->options.GetProductName().c_str());
    _exit(1);
  }

  switch (signum) {
    case SIGINT:
      if (++globals->sigint_count >= 3)  {
        sigprintf("\n%s caught third interrupt signal; killed.\n\n",
                  globals->options.GetProductName().c_str());
        kill(globals->server_pid, SIGKILL);
        _exit(1);
      }
      sigprintf("\n%s caught interrupt signal; shutting down.\n\n",
                globals->options.GetProductName().c_str());
      kill(globals->server_pid, SIGINT);
      break;
    case SIGTERM:
      sigprintf("\n%s caught terminate signal; shutting down.\n\n",
                globals->options.GetProductName().c_str());
      kill(globals->server_pid, SIGINT);
      break;
    case SIGPIPE:
      // Don't bother the user with a message in this case; they're
      // probably using head(1) or more(1).
      kill(globals->server_pid, SIGINT);
      signal(SIGPIPE, SIG_IGN);  // ignore subsequent SIGPIPE signals
      globals->received_signal = SIGPIPE;
      break;
    case SIGQUIT:
      sigprintf("\nSending SIGQUIT to JVM process %d (see %s).\n\n",
                globals->server_pid,
                globals->jvm_log_file.c_str());
      kill(globals->server_pid, SIGQUIT);
      break;
  }
}


// Reads a single char from the specified stream.
static char read_server_char(FILE *fp) {
  int c = getc(fp);
  if (c == EOF) {
    // e.g. external SIGKILL of server, misplaced System.exit() in the server,
    // or a JVM crash. Print out the jvm.out file in case there's something
    // useful.
    fprintf(stderr, "Error: unexpected EOF from %s server.\n"
            "Contents of '%s':\n", globals->options.GetProductName().c_str(),
            globals->jvm_log_file.c_str());
    WriteFileToStreamOrDie(stderr, globals->jvm_log_file.c_str());
    exit(blaze_exit_code::INTERNAL_ERROR);
  }
  return static_cast<char>(c);
}

// Constructs the command line for a server request,
static string BuildServerRequest() {
  vector<string> arg_vector;
  string command = globals->option_processor.GetCommand();
  if (command != "") {
    arg_vector.push_back(command);
    AddLoggingArgs(&arg_vector);
  }

  globals->option_processor.GetCommandArguments(&arg_vector);

  string request("blaze");
  for (vector<string>::iterator it = arg_vector.begin();
       it != arg_vector.end(); it++) {
    request.push_back('\0');
    request.append(*it);
  }
  return request;
}

// Performs all I/O for a single client request to the server, and
// shuts down the client (by exit or signal).
static void SendServerRequest(void) ATTRIBUTE_NORETURN;
static void SendServerRequest(void) {
  int socket = -1;
  while (true) {
    socket = ConnectToServer(true);
    globals->server_pid = GetPeerProcessId(socket);

    // Check for deleted server cwd:
    string server_cwd = GetProcessCWD(globals->server_pid);
    if (server_cwd.empty() ||  // GetProcessCWD failed
        server_cwd != globals->workspace ||  // changed
        server_cwd.find(" (deleted)") != string::npos) {  // deleted.
      // There's a distant possibility that the two paths look the same yet are
      // actually different because the two processes have different mount
      // tables.
      if (VerboseLogging()) {
        fprintf(stderr, "Server's cwd moved or deleted (%s).\n",
                server_cwd.c_str());
      }
      close(socket);
      KillRunningServer(globals->server_pid);
    } else {
      break;
    }
  }

  FILE *fp = fdopen(socket, "r");  // use buffering for reads--it's faster

  if (VerboseLogging()) {
    fprintf(stderr, "Connected (server pid=%d).\n", globals->server_pid);
  }

  // Wall clock time since process startup.
  globals->startup_time = ProcessClock() / 1000000LL;
  const string request = BuildServerRequest();

  // Unblock all signals.
  sigset_t sigset;
  sigemptyset(&sigset);
  sigprocmask(SIG_SETMASK, &sigset, NULL);

  signal(SIGINT,  handler);
  signal(SIGTERM, handler);
  signal(SIGPIPE, handler);
  signal(SIGQUIT, handler);

  // Send request and shutdown the write half of the connection:
  // (Request is written in a single chunk.)
  if (write(socket, request.data(), request.size()) != request.size()) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "write() to server failed");
  }
  // In this (totally bizarre) protocol, this is the
  // client's way of saying "um, that's the end of the request".
  if (shutdown(socket, SHUT_WR) == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "shutdown(WR) failed");
  }

  // Wait until we receive some response from the server.
  // (We do this by calling select() with a timeout.)
  // If we don't receive a response within 3 seconds, print a message,
  // so that the user has some idea what is going on.
  while (true) {
    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(socket, &fdset);
    struct timeval timeout;
    timeout.tv_sec = 3;
    timeout.tv_usec = 0;
    int result = select(socket + 1, &fdset, NULL, &fdset, &timeout);
    if (result > 0) {
      // Data is ready on socket.  Go ahead and read it.
      break;
    } else if (result == 0) {
      // Timeout.  Print a message, then go ahead and read from
      // the socket (the read will usually block).
      fprintf(stderr,
              "INFO: Waiting for response from %s server (pid %d)...\n",
              globals->options.GetProductName().c_str(), globals->server_pid);
      break;
    } else {  // result < 0
      // Error.  For EINTR we try again, all other errors are fatal.
      if (errno != EINTR) {
        pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
             "select() on server socket failed");
      }
    }
  }

  // Read and demux the response. This protocol is awful.
  for (;;) {
    // Read one line:
    char at = read_server_char(fp);
    assert(at == '@');
    (void) at;  // avoid warning about unused variable
    char tag = read_server_char(fp);
    assert(tag == '1' || tag == '2' || tag == '3');
    char at_or_newline = read_server_char(fp);
    bool second_at = at_or_newline == '@';
    if (second_at) {
      at_or_newline = read_server_char(fp);
    }
    assert(at_or_newline == '\n');

    if (tag == '3') {
      // In this (totally bizarre) protocol, this is the
      // server's way of saying "um, that's the end of the response".
      break;
    }
    FILE *stream = tag == '1' ? stdout : stderr;
    for (;;) {
      char c = read_server_char(fp);
      if (c == '\n') {
        if (!second_at) fputc(c, stream);
        fflush(stream);
        break;
      } else {
        fputc(c, stream);
      }
    }
  }

  char line[255];
  if (fgets(line, sizeof line, fp) == NULL ||
      !isdigit(line[0])) {
    die(blaze_exit_code::INTERNAL_ERROR,
        "Error: can't read exit code from server.");
  }
  int exit_code;
  blaze_util::safe_strto32(line, &exit_code);

  close(socket);  // might fail EINTR, just ignore.

  if (globals->received_signal) {  // Kill ourselves with the same signal, so
                                  // that callers see the right WTERMSIG value.
    signal(globals->received_signal, SIG_DFL);
    raise(globals->received_signal);
    exit(1);  // (in case raise didn't kill us for some reason)
  }

  exit(exit_code);
}

// Parse the options, storing parsed values in globals.
// Returns the index of the first non-option argument.
static void ParseOptions(int argc, const char *argv[]) {
  string error;
  blaze_exit_code::ExitCode parse_exit_code =
      globals->option_processor.ParseOptions(argc, argv, globals->workspace,
                                             globals->cwd, &error);
  if (parse_exit_code != blaze_exit_code::SUCCESS) {
    die(parse_exit_code, "%s", error.c_str());
  }
  globals->options = globals->option_processor.GetParsedStartupOptions();
}

// Returns the canonical form of a path.
static string MakeCanonical(const char *path) {
  char *resolved_path = realpath(path, NULL);
  if (resolved_path == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "realpath('%s') failed", path);
  }

  string ret = resolved_path;
  free(resolved_path);
  return ret;
}

// Compute the globals globals->cwd and globals->workspace.
static void ComputeWorkspace() {
  char cwdbuf[PATH_MAX];
  if (getcwd(cwdbuf, sizeof cwdbuf) == NULL) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "getcwd() failed");
  }
  globals->cwd = MakeCanonical(cwdbuf);
  globals->workspace = BlazeStartupOptions::GetWorkspace(globals->cwd);
}

// Figure out the base directories based on embedded data, username, cwd, etc.
// Sets globals->options.install_base, globals->options.output_base,
// globals->lock_file, globals->jvm_log_file.
static void ComputeBaseDirectories(const string self_path) {
  // Only start a server when in a workspace because otherwise we won't do more
  // than emit a help message.
  if (!BlazeStartupOptions::InWorkspace(globals->workspace)) {
    globals->options.batch = true;
  }

  // The default install_base is <output_user_root>/install/<md5(blaze)>
  // but if an install_base is specified on the command line, we use that as
  // the base instead.
  if (globals->options.install_base.empty()) {
    string install_user_root = globals->options.output_user_root + "/install";
    globals->options.install_base =
        GetInstallBase(install_user_root, self_path);
  } else {
    // We call GetInstallBase anyway to populate extracted_binaries.
    GetInstallBase("", self_path);
  }

  if (globals->options.output_base.empty()) {
    globals->options.output_base = GetHashedBaseDir(
        globals->options.output_user_root, globals->workspace);
  }

  struct stat buf;
  if (stat(globals->options.output_base.c_str(), &buf) == -1) {
    if (MakeDirectories(globals->options.output_base, 0777) == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "Output base directory '%s' could not be created",
           globals->options.output_base.c_str());
    }
  } else {
    if (!S_ISDIR(buf.st_mode)) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Error: Output base directory '%s' could not be created. "
          "It exists but is not a directory.",
          globals->options.output_base.c_str());
    }
  }
  if (access(globals->options.output_base.c_str(), R_OK | W_OK | X_OK) != 0) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Error: Output base directory '%s' must be readable and writable.",
        globals->options.output_base.c_str());
  }

  globals->options.output_base =
      MakeCanonical(globals->options.output_base.c_str());
  globals->lockfile = globals->options.output_base + "/lock";
  globals->jvm_log_file = globals->options.output_base + "/server/jvm.out";
}

static void CheckEnvironment() {
  if (getenv("LD_ASSUME_KERNEL") != NULL) {
    // Fix for bug: if ulimit -s and LD_ASSUME_KERNEL are both
    // specified, the JVM fails to create threads.  See thread_stack_regtest.
    // This is also provoked by LD_LIBRARY_PATH=/usr/lib/debug,
    // or anything else that causes the JVM to use LinuxThreads.
    fprintf(stderr, "Warning: ignoring LD_ASSUME_KERNEL in environment.\n");
    unsetenv("LD_ASSUME_KERNEL");
  }

  if (getenv("LD_PRELOAD") != NULL) {
    fprintf(stderr, "Warning: ignoring LD_PRELOAD in environment.\n");
    unsetenv("LD_PRELOAD");
  }

  if (getenv("_JAVA_OPTIONS") != NULL) {
    // This would override --host_jvm_args
    fprintf(stderr, "Warning: ignoring _JAVA_OPTIONS in environment.\n");
    unsetenv("_JAVA_OPTIONS");
  }

  if (getenv("TEST_TMPDIR") != NULL) {
    fprintf(stderr, "INFO: $TEST_TMPDIR defined: output root default is "
                    "'%s'.\n", globals->options.output_root.c_str());
  }

  // TODO(bazel-team):  We've also seen a failure during loading (creating
  // threads?) when ulimit -Hs 8192.  Characterize that and check for it here.

  // Make the JVM use ISO-8859-1 for parsing its command line because "blaze
  // run" doesn't handle non-ASCII command line arguments. This is apparently
  // the most reliable way to select the platform default encoding.
  setenv("LANG", "en_US.ISO-8859-1", 1);
  setenv("LANGUAGE", "en_US.ISO-8859-1", 1);
  setenv("LC_ALL", "en_US.ISO-8859-1", 1);
  setenv("LC_CTYPE", "en_US.ISO-8859-1", 1);
}

// Create the lockfile and take an exclusive lock on a region within it.  This
// lock is inherited with the file descriptor across execve(), but not fork().
// So in the batch case, the JVM holds the lock until exit; otherwise, this
// program holds it until exit.
static void AcquireLock() {
  globals->lockfd = open(globals->lockfile.c_str(), O_CREAT|O_RDWR, 0644);
  if (globals->lockfd < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "cannot open lockfile '%s' for writing", globals->lockfile.c_str());
  }

  struct flock lock;
  lock.l_type = F_WRLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start = 0;
  // This doesn't really matter now, but allows us to subdivide the lock
  // later if that becomes meaningful.  (Ranges beyond EOF can be locked.)
  lock.l_len = 4096;

  // Try to take the lock, without blocking.
  if (fcntl(globals->lockfd, F_SETLK, &lock) == -1) {
    if (errno != EACCES && errno != EAGAIN) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "unexpected result from F_SETLK");
    }

    // We didn't get the lock.  Find out who has it.
    struct flock probe = lock;
    probe.l_pid = 0;
    if (fcntl(globals->lockfd, F_GETLK, &probe) == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "unexpected result from F_GETLK");
    }
    if (!globals->options.block_for_lock) {
      die(blaze_exit_code::BAD_ARGV,
          "Another %s command is running (pid=%d). Exiting immediately.",
          globals->options.GetProductName().c_str(), probe.l_pid);
    }
    fprintf(stderr, "Another %s command is running (pid = %d).  "
            "Waiting for it to complete...",
            globals->options.GetProductName().c_str(), probe.l_pid);
    fflush(stderr);

    // Take a clock sample for that start of the waiting time
    uint64 st = MonotonicClock();
    // Try to take the lock again (blocking).
    int r;
    do {
      r = fcntl(globals->lockfd, F_SETLKW, &lock);
    } while (r == -1 && errno == EINTR);
    fprintf(stderr, "\n");
    if (r == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "couldn't acquire file lock");
    }
    // Take another clock sample, calculate elapsed
    uint64 et = MonotonicClock();
    globals->command_wait_time = (et - st) / 1000000LL;
  }

  // Identify ourselves in the lockfile.
  ftruncate(globals->lockfd, 0);
  const char *tty = ttyname(STDIN_FILENO);  // NOLINT (single-threaded)
  string msg = "owner=" + globals->options.GetProductName() + " launcher\npid="
      + std::to_string(getpid()) + "\ntty=" + (tty ? tty : "") + "\n";
  // Don't bother checking for error, since it's unlikely and unimportant.
  // The contents are currently meant only for debugging.
  write(globals->lockfd, msg.data(), msg.size());
}

// Returns the mountpoint containing the specified directory, which
// must exist.  Fails if any parent path could not be statted or
// canonicalised.
static string GetMountpoint(string dir) {
  dev_t initial_device = -1;
  ino_t prev_inode = -1;
  string prev_dir = dir;
  for (;;) {
    struct stat buf;
    if (stat(dir.c_str(), &buf) == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "stat('%s') failed", dir.c_str());
    } else if (initial_device == -1 && prev_inode == -1) {  // first time
      initial_device = buf.st_dev;
    } else if (initial_device != buf.st_dev) {  // we crossed file systems
      char *resolved_path = realpath(prev_dir.c_str(), NULL);
      if (resolved_path == NULL) {
        pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
             "realpath('%s') failed", prev_dir.c_str());
      }
      dir = resolved_path;
      free(resolved_path);
      return dir;
    } else if (prev_inode == buf.st_ino) {  // ".." had no effect => root.
      return "/";
    }

    prev_inode = buf.st_ino;
    prev_dir = dir;
    dir +=  "/..";
  }

  return "/";
}

void SetupStreams() {
  // Line-buffer stderr, since we always flush at the end of a server
  // message.  This saves lots of single-char calls to write(2).
  // This doesn't work if any writes to stderr have already occurred!
  setlinebuf(stderr);

  // Ensure we have three open fds.  Otherwise we can end up with
  // bizarre things like stdout going to the lock file, etc.
  if (fcntl(0, F_GETFL) == -1) open("/dev/null", O_RDONLY);
  if (fcntl(1, F_GETFL) == -1) open("/dev/null", O_WRONLY);
  if (fcntl(2, F_GETFL) == -1) open("/dev/null", O_WRONLY);
}

// Set an 8MB stack for Blaze. When the stack max is unbounded, it changes the
// layout in the JVM's address space, and we are unable to instantiate the
// default 3000MB heap.
static void EnsureFiniteStackLimit() {
  struct rlimit limit;
  const int default_stack = 8 * 1024 * 1024;  // 8MB.
  if (getrlimit(RLIMIT_STACK, &limit)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "getrlimit() failed");
  }

  if (default_stack < limit.rlim_cur) {
    limit.rlim_cur = default_stack;
    if (setrlimit(RLIMIT_STACK, &limit)) {
      perror("setrlimit() failed: If the stack limit is too high, "
             "this can cause the JVM to be unable to allocate enough "
             "contiguous address space for its heap");
    }
  }
}

static void CheckBinaryPath(const string& argv0) {
  if (argv0[0] == '/') {
    globals->binary_path = argv0;
  } else {
    string abs_path = globals->cwd + '/' + argv0;
    char *resolved_path = realpath(abs_path.c_str(), NULL);
    if (resolved_path) {
      globals->binary_path = resolved_path;
      free(resolved_path);
    } else {
      // This happens during our integration tests, but thats okay, as we won't
      // log the invocation anyway.
      globals->binary_path = abs_path;
    }
  }
}

// Create the user's directory where we keep state, installations etc.
// Typically, this happens inside a temp directory, so we have to be
// careful about symlink attacks.
static void CreateSecureOutputRoot() {
  const char* root = globals->options.output_user_root.c_str();
  struct stat fileinfo = {};

  if (MakeDirectories(root, 0755) == -1) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "mkdir('%s')", root);
  }

  // The path already exists.
  // Check ownership and mode, and verify that it is a directory.

  if (lstat(root, &fileinfo) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "lstat('%s')", root);
  }

  if (fileinfo.st_uid != geteuid()) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "'%s' is not owned by me",
        root);
  }

  if ((fileinfo.st_mode & 022) != 0) {
    int new_mode = fileinfo.st_mode & (~022);
    if (chmod(root, new_mode) < 0) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "'%s' has mode %o, chmod to %o failed", root,
          fileinfo.st_mode & 07777, new_mode);
    }
  }

  if (stat(root, &fileinfo) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "stat('%s')", root);
  }

  if (!S_ISDIR(fileinfo.st_mode)) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "'%s' is not a directory",
        root);
  }
}

// TODO(bazel-team): Execute the server as a child process and write its exit
// code to a file. In case the server becomes unresonsive or terminates
// unexpectedly (in a way that isn't already handled), we can observe the file,
// if it exists. (If it doesn't, then we know something went horribly wrong.)
int main(int argc, const char *argv[]) {
  InitGlobals();
  SetupStreams();

  // Must be done before command line parsing.
  ComputeWorkspace();
  CheckBinaryPath(argv[0]);
  ParseOptions(argc, argv);
  string error;
  blaze_exit_code::ExitCode reexec_options_exit_code =
      globals->options.CheckForReExecuteOptions(argc, argv, &error);
  if (reexec_options_exit_code != blaze_exit_code::SUCCESS) {
    die(reexec_options_exit_code, "%s", error.c_str());
  }
  CheckEnvironment();
  CreateSecureOutputRoot();

  const string self_path = GetSelfPath();
  ComputeBaseDirectories(self_path);

  AcquireLock();

  WarnFilesystemType(globals->options.output_base);
  EnsureFiniteStackLimit();

  ExtractData(self_path);
  EnsureCorrectRunningVersion();
  KillRunningServerIfDifferentStartupOptions();

  if (globals->options.batch) {
    SetScheduling(globals->options.batch_cpu_scheduling,
                  globals->options.io_nice_level);
    StartStandalone();
  } else {
    SendServerRequest();
  }
  return 0;
}
}  // namespace blaze

int main(int argc, const char *argv[]) {
  return blaze::main(argc, argv);
}
