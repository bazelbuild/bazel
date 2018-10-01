// Copyright 2016 The Bazel Authors. All rights reserved.
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
#include "src/tools/singlejar/test_util.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/types.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <string>

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/strings.h"

#include "googletest/include/gtest/gtest.h"

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace singlejar_test_util {

bool AllocateFile(const string &name, size_t size) {
#ifdef _WIN32
  int fd = _sopen(name.c_str(), _O_RDWR | _O_CREAT | _O_BINARY, _SH_DENYNO,
                  _S_IREAD | _S_IWRITE);
  int success = _chsize_s(fd, size);
  _close(fd);
  if (success < 0) {
    perror(strerror(errno));
    return false;
  }
  return true;
#else
  int fd = open(name.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0777);
  if (fd < 0) {
    perror(name.c_str());
    return false;
  }
  if (size) {
    if (ftruncate(fd, size) == 0) {
      return close(fd) == 0;
    } else {
      auto last_error = errno;
      close(fd);
      errno = last_error;
      return false;
    }
  } else {
    return close(fd) == 0;
  }
#endif
}

int RunCommand(const char *cmd, ...) {
  string args_string(cmd);
  va_list ap;
  va_start(ap, cmd);
  for (const char *arg = va_arg(ap, const char *); arg;
       arg = va_arg(ap, const char *)) {
    args_string += ' ';
    args_string += arg;
  }
  va_end(ap);
  fprintf(stderr, "Arguments: %s\n", args_string.c_str());
  return system(args_string.c_str());
}

// List zip file contents.
void LsZip(const char *zip_name) {
#if !defined(__APPLE__)
  RunCommand("unzip", "-v", zip_name, nullptr);
#endif
}

string OutputFilePath(const string &relpath) {
  const char *out_dir = getenv("TEST_TMPDIR");
  return blaze_util::JoinPath((nullptr == out_dir) ? "." : out_dir,
                              relpath.c_str());
}

int VerifyZip(const string &zip_path) {
  string verify_command;
  blaze_util::StringPrintf(&verify_command, "zip -Tv %s", zip_path.c_str());
  return system(verify_command.c_str());
}

string GetEntryContents(const string &zip_path, const string &entry_name) {
  string contents;
  string command;
  blaze_util::StringPrintf(&command, "unzip -p %s %s", zip_path.c_str(),
                           entry_name.c_str());
#ifdef _WIN32
  FILE *fp = popen(command.c_str(), "rb");
#else
  FILE *fp = popen(command.c_str(), "r");
#endif
  if (!fp) {
    ADD_FAILURE() << "Command " << command << " failed.";
    return string("");
  }

  char buf[1024];
  while (fgets(buf, sizeof(buf), fp)) {
    contents.append(buf);
  }
  if (feof(fp) && !ferror(fp) && !pclose(fp)) {
    return contents;
  }
  ADD_FAILURE() << "Command " << command << " failed on close.";
  return string("");
}

string CreateTextFile(const string& relpath, const char *contents) {
  string out_path = OutputFilePath(relpath);
  blaze_util::MakeDirectories(blaze_util::Dirname(out_path), 0777);
  if (blaze_util::WriteFile(contents, out_path)) {
    return out_path;
  }
  ADD_FAILURE() << "Cannot write " << out_path;
  return string("");
}

}  // namespace singlejar_test_util
