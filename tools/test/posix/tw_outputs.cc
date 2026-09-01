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

#include "tools/test/posix/tw_outputs.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "third_party/ijar/platform_utils.h"
#include "third_party/ijar/zip.h"
#include "tools/test/test_wrapper_common.h"

namespace bazel {
namespace tools {
namespace test_wrapper {
namespace {

constexpr size_t kMimeSampleSize = 8192;

struct OutputEntry {
  std::string relative_path;
  uint64_t size;
  mode_t mode;
  bool is_directory;
};

using DirectoryIdentity = std::pair<dev_t, ino_t>;

bool Fail(const std::string& message, std::string* error) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool FailWithErrno(const std::string& operation, const std::string& path,
                   int error_number, std::string* error) {
  return Fail(operation + " '" + path +
                  "': " + std::string(std::strerror(error_number)),
              error);
}

std::string JoinPath(const std::string& directory, const std::string& name) {
  if (directory.empty() || directory.back() == '/') {
    return directory + name;
  }
  return directory + "/" + name;
}

std::string ParentDirectory(const std::string& path) {
  const std::string::size_type separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return ".";
  }
  return separator == 0 ? "/" : path.substr(0, separator);
}

bool EnsureParentDirectory(const std::string& path, std::string* error) {
  const std::string parent = ParentDirectory(path);
  std::string current = !parent.empty() && parent[0] == '/' ? "/" : "";
  size_t position = current.size();
  while (position < parent.size()) {
    const size_t separator = parent.find('/', position);
    const size_t length = separator == std::string::npos
                              ? parent.size() - position
                              : separator - position;
    if (length != 0) {
      current = JoinPath(current, parent.substr(position, length));
      struct stat existing;
      if (stat(current.c_str(), &existing) == 0) {
        if (!S_ISDIR(existing.st_mode)) {
          return FailWithErrno("Parent directory is not a directory", current,
                               ENOTDIR, error);
        }
      } else if (errno != ENOENT) {
        return FailWithErrno("Cannot stat parent directory", current, errno,
                             error);
      } else if (mkdir(current.c_str(), 0777) != 0) {
        const int mkdir_error = errno;
        if (mkdir_error != EEXIST || stat(current.c_str(), &existing) != 0 ||
            !S_ISDIR(existing.st_mode)) {
          return FailWithErrno("Cannot create parent directory", current,
                               mkdir_error, error);
        }
      }
    }
    if (separator == std::string::npos) {
      break;
    }
    position = separator + 1;
  }
  return true;
}

bool ListOutputs(const std::string& root, const std::string& relative_dir,
                 std::set<DirectoryIdentity>* ancestors,
                 std::vector<OutputEntry>* entries,
                 std::vector<std::string>* root_children, std::string* error) {
  const std::string directory =
      relative_dir.empty() ? root : JoinPath(root, relative_dir);
  std::unique_ptr<DIR, decltype(&closedir)> stream(opendir(directory.c_str()),
                                                   &closedir);
  if (stream == nullptr) {
    return FailWithErrno("Cannot open undeclared-output directory", directory,
                         errno, error);
  }

  std::vector<std::pair<std::string, DirectoryIdentity>> subdirectories;
  for (;;) {
    errno = 0;
    dirent* const entry = readdir(stream.get());
    if (entry == nullptr) {
      if (errno != 0) {
        return FailWithErrno("Cannot read undeclared-output directory",
                             directory, errno, error);
      }
      break;
    }

    const std::string name(entry->d_name);
    if (name == "." || name == "..") {
      continue;
    }

    if (relative_dir.empty()) {
      root_children->push_back(name);
    }

    const std::string relative_path =
        relative_dir.empty() ? name : JoinPath(relative_dir, name);
    const std::string absolute_path = JoinPath(root, relative_path);
    struct stat file_stat;
    if (stat(absolute_path.c_str(), &file_stat) != 0) {
      const int stat_error = errno;
      if (stat_error == ENOENT || stat_error == ENOTDIR) {
        continue;
      }
      return FailWithErrno("Cannot stat undeclared output", absolute_path,
                           stat_error, error);
    }

    if (S_ISDIR(file_stat.st_mode)) {
      entries->push_back(
          OutputEntry{relative_path, 0, file_stat.st_mode, true});
      const DirectoryIdentity identity(file_stat.st_dev, file_stat.st_ino);
      if (ancestors->find(identity) == ancestors->end()) {
        subdirectories.emplace_back(relative_path, identity);
      }
    } else if (S_ISREG(file_stat.st_mode)) {
      if (file_stat.st_size < 0) {
        return Fail(
            "Undeclared output has an invalid size: '" + absolute_path + "'",
            error);
      }
      entries->push_back(OutputEntry{relative_path,
                                     static_cast<uint64_t>(file_stat.st_size),
                                     file_stat.st_mode, false});
    }
  }

  DIR* const opened_directory = stream.release();
  if (closedir(opened_directory) != 0) {
    return FailWithErrno("Cannot close undeclared-output directory", directory,
                         errno, error);
  }

  for (const auto& subdirectory : subdirectories) {
    ancestors->insert(subdirectory.second);
    if (!ListOutputs(root, subdirectory.first, ancestors, entries,
                     root_children, error)) {
      return false;
    }
    ancestors->erase(subdirectory.second);
  }
  return true;
}

bool ReadPrefix(const std::string& path, std::string* content,
                std::string* error) {
  const int descriptor = open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    return FailWithErrno("Cannot read undeclared output", path, errno, error);
  }

  std::array<char, kMimeSampleSize> buffer;
  ssize_t bytes_read;
  do {
    bytes_read = read(descriptor, buffer.data(), buffer.size());
  } while (bytes_read < 0 && errno == EINTR);
  const int read_error = errno;
  if (close(descriptor) != 0 && bytes_read >= 0) {
    return FailWithErrno("Cannot close undeclared output", path, errno, error);
  }
  if (bytes_read < 0) {
    return FailWithErrno("Cannot read undeclared output", path, read_error,
                         error);
  }
  content->assign(buffer.data(), static_cast<size_t>(bytes_read));
  return true;
}

bool StartsWithBytes(const std::string& content,
                     std::initializer_list<unsigned char> signature) {
  if (content.size() < signature.size()) {
    return false;
  }
  size_t position = 0;
  for (const unsigned char byte : signature) {
    if (static_cast<unsigned char>(content[position++]) != byte) {
      return false;
    }
  }
  return true;
}

bool IsUtf8Text(const std::string& content) {
  for (size_t position = 0; position < content.size();) {
    const unsigned char first = static_cast<unsigned char>(content[position++]);
    if (first < 0x80) {
      if ((first < 0x20 && first != '\t' && first != '\n' && first != '\r' &&
           first != '\f') ||
          first == 0x7f) {
        return false;
      }
      continue;
    }

    size_t remaining;
    uint32_t code_point;
    uint32_t minimum;
    if (first >= 0xc2 && first <= 0xdf) {
      remaining = 1;
      code_point = first & 0x1f;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      remaining = 2;
      code_point = first & 0x0f;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      remaining = 3;
      code_point = first & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }

    if (remaining > content.size() - position) {
      return content.size() == kMimeSampleSize;
    }
    for (size_t index = 0; index < remaining; ++index) {
      const unsigned char continuation =
          static_cast<unsigned char>(content[position++]);
      if ((continuation & 0xc0) != 0x80) {
        return false;
      }
      code_point = (code_point << 6) | (continuation & 0x3f);
    }
    if (code_point < minimum || code_point > 0x10ffff ||
        (code_point >= 0xd800 && code_point <= 0xdfff)) {
      return false;
    }
  }
  return true;
}

std::string LowercaseTextPrefix(const std::string& content) {
  size_t position = StartsWithBytes(content, {0xef, 0xbb, 0xbf}) ? 3 : 0;
  while (position < content.size() &&
         (content[position] == ' ' || content[position] == '\t' ||
          content[position] == '\n' || content[position] == '\r')) {
    ++position;
  }

  std::string prefix = content.substr(position, 128);
  std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                 [](unsigned char value) {
                   return value >= 'A' && value <= 'Z'
                              ? static_cast<char>(value + ('a' - 'A'))
                              : static_cast<char>(value);
                 });
  return prefix;
}

std::string ClassifyContent(const std::string& content) {
  if (content.empty()) {
    return "inode/x-empty";
  }
  if (StartsWithBytes(content, {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})) {
    return "image/png";
  }
  if (StartsWithBytes(content, {0xff, 0xd8, 0xff})) {
    return "image/jpeg";
  }
  if (content.compare(0, 6, "GIF87a") == 0 ||
      content.compare(0, 6, "GIF89a") == 0) {
    return "image/gif";
  }
  if (StartsWithBytes(content, {0x00, 0x00, 0x01, 0x00}) ||
      StartsWithBytes(content, {0x00, 0x00, 0x02, 0x00})) {
    return "image/vnd.microsoft.icon";
  }
  if (content.compare(0, 5, "%PDF-") == 0) {
    return "application/pdf";
  }
  if (StartsWithBytes(content, {'P', 'K', 0x03, 0x04}) ||
      StartsWithBytes(content, {'P', 'K', 0x05, 0x06}) ||
      StartsWithBytes(content, {'P', 'K', 0x07, 0x08})) {
    return "application/zip";
  }
  if (StartsWithBytes(content, {0x1f, 0x8b})) {
    return "application/gzip";
  }
  if (!IsUtf8Text(content)) {
    return "application/octet-stream";
  }

  const std::string prefix = LowercaseTextPrefix(content);
  if (prefix.compare(0, 14, "<!doctype html") == 0 ||
      prefix.compare(0, 5, "<html") == 0 ||
      prefix.compare(0, 5, "<head") == 0 ||
      prefix.compare(0, 6, "<title") == 0 ||
      prefix.compare(0, 5, "<body") == 0) {
    return "text/html";
  }
  if (prefix.compare(0, 4, "<svg") == 0 ||
      (prefix.compare(0, 5, "<?xml") == 0 &&
       prefix.find("<svg") != std::string::npos)) {
    return "image/svg+xml";
  }
  if (prefix.compare(0, 5, "<?xml") == 0) {
    return "text/xml";
  }
  if (!prefix.empty() && (prefix.front() == '{' || prefix.front() == '[')) {
    return "application/json";
  }
  if (prefix.compare(0, 2, "#!") == 0) {
    return "text/x-shellscript";
  }
  return "text/plain";
}

bool WriteManifest(const std::string& root, const std::string& manifest,
                   const std::vector<OutputEntry>& entries,
                   std::string* error) {
  if (manifest.empty()) {
    return true;
  }

  std::string content;
  for (const OutputEntry& entry : entries) {
    if (entry.is_directory) {
      continue;
    }
    std::string sample;
    if (!ReadPrefix(JoinPath(root, entry.relative_path), &sample, error)) {
      return false;
    }
    content += FormatUndeclaredOutputManifestEntry(
        entry.relative_path, entry.size, ClassifyContent(sample));
  }
  if (content.empty()) {
    return true;
  }
  if (!EnsureParentDirectory(manifest, error)) {
    return false;
  }
  std::ofstream output(manifest,
                       std::ios::out | std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return FailWithErrno("Cannot open undeclared-output manifest", manifest,
                         errno, error);
  }
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  output.close();
  if (output.fail()) {
    return Fail("Cannot write undeclared-output manifest '" + manifest + "'",
                error);
  }
  return true;
}

bool AppendFile(const std::string& input, std::ofstream* output,
                std::string* error) {
  std::ifstream stream(input, std::ios::in | std::ios::binary);
  if (!stream.is_open()) {
    return FailWithErrno("Cannot open undeclared-output annotation", input,
                         errno, error);
  }

  std::array<char, 65536> buffer;
  while (stream.good()) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize bytes_read = stream.gcount();
    if (bytes_read > 0) {
      output->write(buffer.data(), bytes_read);
      if (output->fail()) {
        return Fail(
            "Cannot append undeclared-output annotation '" + input + "'",
            error);
      }
    }
  }
  if (!stream.eof()) {
    return Fail("Cannot read undeclared-output annotation '" + input + "'",
                error);
  }
  return true;
}

bool HasSuffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

bool ConcatenateAnnotations(const std::string& directory,
                            const std::vector<std::string>& entries,
                            const std::string& suffix,
                            const std::string& destination,
                            std::string* error) {
  std::vector<std::string> matching;
  for (const std::string& entry : entries) {
    if (!entry.empty() && entry.front() != '.' && HasSuffix(entry, suffix)) {
      const std::string path = JoinPath(directory, entry);
      struct stat file_stat;
      if (stat(path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
        matching.push_back(entry);
      }
    }
  }
  if (matching.empty()) {
    return true;
  }
  if (!EnsureParentDirectory(destination, error)) {
    return false;
  }

  std::ofstream output(destination,
                       std::ios::out | std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return FailWithErrno("Cannot open undeclared-output annotations",
                         destination, errno, error);
  }
  for (const std::string& entry : matching) {
    if (!AppendFile(JoinPath(directory, entry), &output, error)) {
      return false;
    }
  }
  output.close();
  if (output.fail()) {
    return Fail(
        "Cannot write undeclared-output annotations '" + destination + "'",
        error);
  }
  return true;
}

bool ProcessAnnotations(const UndeclaredOutputs& outputs, std::string* error) {
  if (outputs.annotations.empty() || outputs.annotations_dir.empty()) {
    return true;
  }

  std::unique_ptr<DIR, decltype(&closedir)> directory(
      opendir(outputs.annotations_dir.c_str()), &closedir);
  if (directory == nullptr) {
    if (errno == ENOENT || errno == ENOTDIR) {
      return true;
    }
    return FailWithErrno("Cannot open undeclared-output annotations directory",
                         outputs.annotations_dir, errno, error);
  }

  std::vector<std::string> entries;
  for (;;) {
    errno = 0;
    dirent* const entry = readdir(directory.get());
    if (entry == nullptr) {
      if (errno != 0) {
        return FailWithErrno(
            "Cannot read undeclared-output annotations directory",
            outputs.annotations_dir, errno, error);
      }
      break;
    }
    if (std::strcmp(entry->d_name, ".") != 0 &&
        std::strcmp(entry->d_name, "..") != 0) {
      entries.emplace_back(entry->d_name);
    }
  }
  std::sort(entries.begin(), entries.end());
  return ConcatenateAnnotations(outputs.annotations_dir, entries, ".part",
                                outputs.annotations, error) &&
         ConcatenateAnnotations(outputs.annotations_dir, entries, ".pb",
                                outputs.annotations + ".pb", error);
}

bool ReadZipEntry(const std::string& path, uint64_t size,
                  devtools_ijar::u1* destination, std::string* error) {
  const int descriptor = open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    return FailWithErrno("Cannot open undeclared output for ZIP", path, errno,
                         error);
  }

  size_t remaining = static_cast<size_t>(size);
  while (remaining > 0) {
    const ssize_t bytes_read = read(descriptor, destination, remaining);
    if (bytes_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int read_error = errno;
      close(descriptor);
      return FailWithErrno("Cannot read undeclared output for ZIP", path,
                           read_error, error);
    }
    if (bytes_read == 0) {
      close(descriptor);
      return Fail(
          "Undeclared output changed while creating ZIP: '" + path + "'",
          error);
    }
    destination += bytes_read;
    remaining -= static_cast<size_t>(bytes_read);
  }
  if (close(descriptor) != 0) {
    return FailWithErrno("Cannot close undeclared output for ZIP", path, errno,
                         error);
  }
  return true;
}

std::string ZipError(devtools_ijar::ZipBuilder* builder) {
  const char* const message = builder->GetError();
  return message == nullptr ? "unknown ZIP error" : std::string(message);
}

bool CreateZip(const UndeclaredOutputs& outputs,
               const std::vector<OutputEntry>& entries, std::string* error) {
  if (outputs.zip.empty() || entries.empty()) {
    return true;
  }
  if (entries.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return Fail("Too many undeclared outputs to archive", error);
  }
  if (!EnsureParentDirectory(outputs.zip, error)) {
    return false;
  }

  std::vector<std::string> relative_paths;
  relative_paths.reserve(entries.size());
  for (const OutputEntry& entry : entries) {
    if (entry.size >
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return Fail("Undeclared output exceeds the ZIP file-size limit: '" +
                      JoinPath(outputs.root, entry.relative_path) + "'",
                  error);
    }
    relative_paths.push_back(entry.relative_path +
                             (entry.is_directory ? "/" : ""));
  }

  ZipEntryPaths paths;
  paths.Create(outputs.root, relative_paths);
  const devtools_ijar::u8 estimated_size =
      devtools_ijar::ZipBuilder::EstimateSize(paths.AbsPathPtrs(),
                                              paths.EntryPathPtrs(),
                                              static_cast<int>(paths.Size()));
  if (estimated_size == 0) {
    return Fail(
        "Cannot estimate undeclared-output ZIP size for '" + outputs.zip + "'",
        error);
  }
  if (estimated_size > std::numeric_limits<uint32_t>::max()) {
    return Fail("Undeclared outputs exceed the maximum ZIP archive size: '" +
                    outputs.zip + "'",
                error);
  }

  std::unique_ptr<devtools_ijar::ZipBuilder> builder(
      devtools_ijar::ZipBuilder::Create(outputs.zip.c_str(), estimated_size));
  if (builder == nullptr) {
    return FailWithErrno("Cannot create undeclared-output ZIP", outputs.zip,
                         errno, error);
  }

  for (size_t index = 0; index < entries.size(); ++index) {
    const OutputEntry& entry = entries[index];
    const devtools_ijar::Stat file_stat = {entry.size, entry.mode,
                                           entry.is_directory};
    devtools_ijar::u1* const destination =
        builder->NewFile(paths.EntryPathPtrs()[index],
                         devtools_ijar::stat_to_zipattr(file_stat));
    if (destination == nullptr) {
      return Fail("Cannot add undeclared output '" + entry.relative_path +
                      "' to ZIP: " + ZipError(builder.get()),
                  error);
    }
    if (!entry.is_directory &&
        !ReadZipEntry(JoinPath(outputs.root, entry.relative_path), entry.size,
                      destination, error)) {
      return false;
    }
    if (builder->FinishFile(static_cast<size_t>(entry.size),
                            /*compress=*/true, /*compute_crc=*/true) == -1) {
      return Fail("Cannot finish undeclared-output ZIP entry '" +
                      entry.relative_path + "': " + ZipError(builder.get()),
                  error);
    }
  }
  if (builder->Finish() == -1) {
    return Fail("Cannot finish undeclared-output ZIP '" + outputs.zip +
                    "': " + ZipError(builder.get()),
                error);
  }
  return true;
}

bool DeleteArchivedOutputs(const UndeclaredOutputs& outputs,
                           const std::vector<std::string>& root_children,
                           std::string* error) {
  if (outputs.zip.empty()) {
    return true;
  }
  for (const std::string& child : root_children) {
    const std::string path = JoinPath(outputs.root, child);
    if (path == outputs.zip) {
      continue;
    }
    if (!blaze_util::RemoveRecursively(blaze_util::Path(path))) {
      return FailWithErrno("Cannot remove archived undeclared output", path,
                           errno, error);
    }
  }
  return true;
}

}  // namespace

bool ProcessUndeclaredOutputs(const UndeclaredOutputs& outputs,
                              std::string* error) {
  if (error != nullptr) {
    error->clear();
  }

  std::vector<OutputEntry> entries;
  std::vector<std::string> root_children;
  if (!outputs.root.empty() &&
      (!outputs.manifest.empty() || !outputs.zip.empty())) {
    struct stat root_stat;
    if (stat(outputs.root.c_str(), &root_stat) != 0) {
      const int stat_error = errno;
      if (stat_error != ENOENT && stat_error != ENOTDIR) {
        return FailWithErrno("Cannot stat undeclared-output directory",
                             outputs.root, stat_error, error);
      }
    } else if (!S_ISDIR(root_stat.st_mode)) {
      return Fail(
          "Undeclared-output root is not a directory: '" + outputs.root + "'",
          error);
    } else {
      std::set<DirectoryIdentity> ancestors;
      ancestors.emplace(root_stat.st_dev, root_stat.st_ino);
      if (!ListOutputs(outputs.root, "", &ancestors, &entries, &root_children,
                       error)) {
        return false;
      }
      std::sort(entries.begin(), entries.end(),
                [](const OutputEntry& first, const OutputEntry& second) {
                  return first.relative_path < second.relative_path;
                });
      std::sort(root_children.begin(), root_children.end());
      if (!WriteManifest(outputs.root, outputs.manifest, entries, error)) {
        return false;
      }
    }
  }

  if (!ProcessAnnotations(outputs, error)) {
    return false;
  }
  if (!CreateZip(outputs, entries, error)) {
    return false;
  }
  if (!entries.empty() &&
      !DeleteArchivedOutputs(outputs, root_children, error)) {
    return false;
  }
  return true;
}

}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel
