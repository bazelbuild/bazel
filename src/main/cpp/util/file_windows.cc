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
#include "src/main/cpp/util/file_platform.h"

#include <windows.h>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/file.h"

namespace blaze_util {

using std::string;

#ifdef COMPILER_MSVC
class WindowsPipe : public IPipe {
 public:
  bool Send(void* buffer, int size) override {
    // TODO(bazel-team): implement this.
    pdie(255, "blaze_util::WindowsPipe::Send is not yet implemented");
    return false;
  }

  int Receive(void* buffer, int size) override {
    // TODO(bazel-team): implement this.
    pdie(255, "blaze_util::WindowsPipe::Receive is not yet implemented");
    return 0;
  }
};
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
IPipe* CreatePipe() {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::CreatePipe is not implemented on Windows");
  return nullptr;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool ReadFile(const string& filename, string* content, int max_size) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ReadFile is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool WriteFile(const void* data, size_t size, const string& filename) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::WriteFile is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool UnlinkPath(const string& file_path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::UnlinkPath is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
string Which(const string &executable) {
  pdie(255, "blaze_util::Which is not implemented on Windows");
  return "";
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool PathExists(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::PathExists is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
string MakeCanonical(const char *path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::MakeCanonical is not implemented on Windows");
  return "";
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool CanAccess(const string& path, bool read, bool write, bool exec) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::CanAccess is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool IsDirectory(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::IsDirectory is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
void SyncFile(const string& path) {
  // No-op on Windows native; unsupported by Cygwin.
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
time_t GetMtimeMillisec(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::GetMtimeMillisec is not implemented on Windows");
  return -1;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool SetMtimeMillisec(const string& path, time_t mtime) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::SetMtimeMillisec is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool MakeDirectories(const string& path, unsigned int mode) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze::MakeDirectories is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
string GetCwd() {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::GetCwd is not implemented on Windows");
  return "";
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool ChangeDirectory(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ChangeDirectory is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
void ForEachDirectoryEntry(const string &path,
                           DirectoryEntryConsumer *consume) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ForEachDirectoryEntry is not implemented on Windows");
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

}  // namespace blaze_util
