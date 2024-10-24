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
// This program creates a "runfiles tree" from a "runfiles manifest".
//
// The command line arguments are an input manifest INPUT and an output
// directory RUNFILES. First, the files in the RUNFILES directory are scanned
// and any extraneous ones are removed. Second, any missing files are created.
// Finally, a copy of the input manifest is written to RUNFILES/MANIFEST.
//
// The input manifest consists of lines, each containing a relative path within
// the runfiles, a space, and an optional absolute path.  If this second path
// is present, a symlink is created pointing to it; otherwise an empty file is
// created.
//
// Given the line
//   <workspace root>/output/path /real/path
// we will create directories
//   RUNFILES/<workspace root>
//   RUNFILES/<workspace root>/output
// a symlink
//   RUNFILES/<workspace root>/output/path -> /real/path
// and the output manifest will contain a line
//   <workspace root>/output/path /real/path
//
// If --use_metadata is supplied, every other line is treated as opaque
// metadata, and is ignored here.
//
// All output paths must be relative and generally (but not always) begin with
// <workspace root>. No output path may be equal to another.  No output path may
// be a path prefix of another.

#define _FILE_OFFSET_BITS 64

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <map>
#include <string>

// program_invocation_short_name is not portable.
static const char *argv0;

const char *input_filename;
const char *output_base_dir;

#define LOG() { \
  fprintf(stderr, "%s (args %s %s): ", \
          argv0, input_filename, output_base_dir); \
}

#define DIE(args...) { \
  LOG(); \
  fprintf(stderr, args); \
  fprintf(stderr, "\n"); \
  exit(1); \
}

#define PDIE(args...) { \
  int saved_errno = errno; \
  LOG(); \
  fprintf(stderr, args); \
  fprintf(stderr, ": %s [%d]\n", strerror(saved_errno), saved_errno); \
  exit(1); \
}

enum FileType {
  FILE_TYPE_REGULAR,
  FILE_TYPE_DIRECTORY,
  FILE_TYPE_SYMLINK
};

struct FileInfo {
  FileType type;
  std::string symlink_target;

  bool operator==(const FileInfo &other) const {
    return type == other.type && symlink_target == other.symlink_target;
  }

  bool operator!=(const FileInfo &other) const {
    return !(*this == other);
  }
};

typedef std::map<std::string, FileInfo> FileInfoMap;

// Replaces \s, \n, and \b with their respective characters.
std::string Unescape(const std::string &path) {
  std::string result;
  result.reserve(path.size());
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '\\' && i + 1 < path.size()) {
      switch (path[i + 1]) {
        case 's': {
          result.push_back(' ');
          break;
        }
        case 'n': {
          result.push_back('\n');
          break;
        }
        case 'b': {
          result.push_back('\\');
          break;
        }
        default: {
          result.push_back(path[i]);
          result.push_back(path[i + 1]);
          break;
        }
      }
      ++i;
    } else {
      result.push_back(path[i]);
    }
  }
  return result;
}

class RunfilesCreator {
 public:
  explicit RunfilesCreator(const std::string &output_base)
      : output_base_(output_base),
        output_filename_("MANIFEST"),
        temp_filename_(output_filename_ + ".tmp") {
    SetupOutputBase();
    if (chdir(output_base_.c_str()) != 0) {
      PDIE("chdir '%s'", output_base_.c_str());
    }
  }

  void ReadManifest(const std::string &manifest_file, bool allow_relative,
                    bool use_metadata) {
    // Remove file left over from previous invocation. This ensures that
    // opening succeeds if the existing file is read-only.
    if (unlink(temp_filename_.c_str()) != 0 && errno != ENOENT) {
      PDIE("removing temporary file at '%s/%s'", output_base_.c_str(),
           temp_filename_.c_str());
    }
    FILE *outfile = fopen(temp_filename_.c_str(), "w");
    if (!outfile) {
      PDIE("opening '%s/%s' for writing", output_base_.c_str(),
           temp_filename_.c_str());
    }
    FILE *infile = fopen(manifest_file.c_str(), "r");
    if (!infile) {
      PDIE("opening '%s' for reading", manifest_file.c_str());
    }

    // read input manifest
    int lineno = 0;
    char buf[3 * PATH_MAX];
    while (fgets(buf, sizeof buf, infile)) {
      // copy line to output manifest
      if (fputs(buf, outfile) == EOF) {
        PDIE("writing to '%s/%s'", output_base_.c_str(),
             temp_filename_.c_str());
      }

      // parse line
      ++lineno;
      // Skip metadata lines. They are used solely for
      // dependency checking.
      if (use_metadata && lineno % 2 == 0) continue;

      int n = strlen(buf)-1;
      if (!n || buf[n] != '\n') {
        DIE("missing terminator at line %d: '%s'\n", lineno, buf);
      }
      buf[n] = '\0';
      if (buf[0] ==  '/') {
        DIE("paths must not be absolute: line %d: '%s'\n", lineno, buf);
      }
      std::string link;
      std::string target;
      if (buf[0] == ' ') {
        // The link path contains escape sequences for spaces and backslashes.
        char *s = strchr(buf + 1, ' ');
        if (!s) {
          DIE("missing field delimiter at line %d: '%s'\n", lineno, buf);
        }
        link = Unescape(std::string(buf + 1, s));
        target = Unescape(s + 1);
      } else {
        // The line is of the form "foo /target/path", with only a single space
        // in the link path.
        const char *s = strchr(buf, ' ');
        if (!s) {
          DIE("missing field delimiter at line %d: '%s'\n", lineno, buf);
        }
        link = std::string(buf, s - buf);
        target = s + 1;
      }
      if (!allow_relative && target[0] != '\0' && target[0] != '/'
          && target[1] != ':') {  // Match Windows paths, e.g. C:\foo or C:/foo.
        DIE("expected absolute path at line %d: '%s'\n", lineno, buf);
      }

      FileInfo *info = &manifest_[link];
      if (target[0] == '\0') {
        // No target means an empty file.
        info->type = FILE_TYPE_REGULAR;
      } else {
        info->type = FILE_TYPE_SYMLINK;
        info->symlink_target = target;
      }

      FileInfo parent_info;
      parent_info.type = FILE_TYPE_DIRECTORY;

      while (true) {
        int k = link.rfind('/');
        if (k < 0) break;
        link.erase(k, std::string::npos);
        if (!manifest_.insert(std::make_pair(link, parent_info)).second) break;
      }
    }
    if (fclose(outfile) != 0) {
      PDIE("writing to '%s/%s'", output_base_.c_str(),
           temp_filename_.c_str());
    }
    fclose(infile);

    // Don't delete the temp manifest file.
    manifest_[temp_filename_].type = FILE_TYPE_REGULAR;
  }

  void CreateRunfiles() {
    if (unlink(output_filename_.c_str()) != 0 && errno != ENOENT) {
      PDIE("removing previous file at '%s/%s'", output_base_.c_str(),
           output_filename_.c_str());
    }

    ScanTreeAndPrune(".");
    CreateFiles();

    // rename output file into place
    if (rename(temp_filename_.c_str(), output_filename_.c_str()) != 0) {
      PDIE("renaming '%s/%s' to '%s/%s'",
           output_base_.c_str(), temp_filename_.c_str(),
           output_base_.c_str(), output_filename_.c_str());
    }
  }

 private:
  void SetupOutputBase() {
    struct stat st;
    if (stat(output_base_.c_str(), &st) != 0) {
      // Technically, this will cause problems if the user's umask contains
      // 0200, but we don't care. Anyone who does that deserves what's coming.
      if (mkdir(output_base_.c_str(), 0777) != 0) {
        PDIE("creating directory '%s'", output_base_.c_str());
      }
    } else {
      EnsureDirReadAndWritePerms(output_base_);
    }
  }

  void ScanTreeAndPrune(const std::string &path) {
    // A note on non-empty files:
    // We don't distinguish between empty and non-empty files. That is, if
    // there's a file that has contents, we don't truncate it here, even though
    // the manifest supports creation of empty files, only. Given that
    // .runfiles are *supposed* to be immutable, this shouldn't be a problem.
    EnsureDirReadAndWritePerms(path);

    struct dirent *entry;
    DIR *dh = opendir(path.c_str());
    if (!dh) {
      PDIE("opendir '%s'", path.c_str());
    }

    errno = 0;
    const std::string prefix = (path == "." ? "" : path + "/");
    while ((entry = readdir(dh)) != nullptr) {
      if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;

      std::string entry_path = prefix + entry->d_name;
      FileInfo actual_info;
      actual_info.type = DentryToFileType(entry_path, entry);

      if (actual_info.type == FILE_TYPE_SYMLINK) {
        ReadLinkOrDie(entry_path, &actual_info.symlink_target);
      }

      FileInfoMap::iterator expected_it = manifest_.find(entry_path);
      if (expected_it == manifest_.end() ||
          expected_it->second != actual_info) {
#if !defined(__CYGWIN__)
        DelTree(entry_path, actual_info.type);
#else
        // On Windows, if deleting failed, lamely assume that
        // the link points to the right place.
        if (!DelTree(entry_path, actual_info.type)) {
          manifest_.erase(expected_it);
        }
#endif
      } else {
        manifest_.erase(expected_it);
        if (actual_info.type == FILE_TYPE_DIRECTORY) {
          ScanTreeAndPrune(entry_path);
        }
      }

      errno = 0;
    }
    if (errno != 0) {
      PDIE("reading directory '%s'", path.c_str());
    }
    closedir(dh);
  }

  void CreateFiles() {
    for (FileInfoMap::const_iterator it = manifest_.begin();
         it != manifest_.end(); ++it) {
      const std::string &path = it->first;
      switch (it->second.type) {
        case FILE_TYPE_DIRECTORY:
          if (mkdir(path.c_str(), 0777) != 0) {
            PDIE("mkdir '%s'", path.c_str());
          }
          break;
        case FILE_TYPE_REGULAR:
          {
            int fd = open(path.c_str(), O_CREAT|O_EXCL|O_WRONLY, 0555);
            if (fd < 0) {
              PDIE("creating empty file '%s'", path.c_str());
            }
            close(fd);
          }
          break;
        case FILE_TYPE_SYMLINK:
          {
            const std::string& target = it->second.symlink_target;
            if (symlink(target.c_str(), path.c_str()) != 0) {
              PDIE("symlinking '%s' -> '%s'", path.c_str(), target.c_str());
            }
          }
          break;
      }
    }
  }

  FileType DentryToFileType(const std::string &path, struct dirent *ent) {
#ifdef _DIRENT_HAVE_D_TYPE
    if (ent->d_type != DT_UNKNOWN) {
      if (ent->d_type == DT_DIR) {
        return FILE_TYPE_DIRECTORY;
      } else if (ent->d_type == DT_LNK) {
        return FILE_TYPE_SYMLINK;
      } else {
        return FILE_TYPE_REGULAR;
      }
    } else  // NOLINT (the brace is in the next line)
#endif
    {
      struct stat st;
      LStatOrDie(path, &st);
      if (S_ISDIR(st.st_mode)) {
        return FILE_TYPE_DIRECTORY;
      } else if (S_ISLNK(st.st_mode)) {
        return FILE_TYPE_SYMLINK;
      } else {
        return FILE_TYPE_REGULAR;
      }
    }
  }

  void LStatOrDie(const std::string &path, struct stat *st) {
    if (lstat(path.c_str(), st) != 0) {
      PDIE("lstating file '%s'", path.c_str());
    }
  }

  void StatOrDie(const std::string &path, struct stat *st) {
    if (stat(path.c_str(), st) != 0) {
      PDIE("stating file '%s'", path.c_str());
    }
  }

  void ReadLinkOrDie(const std::string &path, std::string *output) {
    char readlink_buffer[PATH_MAX];
    int sz = readlink(path.c_str(), readlink_buffer, sizeof(readlink_buffer));
    if (sz < 0) {
      PDIE("reading symlink '%s'", path.c_str());
    }
    // readlink returns a non-null terminated string.
    std::string(readlink_buffer, sz).swap(*output);
  }

  void EnsureDirReadAndWritePerms(const std::string &path) {
    const int kMode = 0700;
    struct stat st;
    LStatOrDie(path, &st);
    if ((st.st_mode & kMode) != kMode) {
      int new_mode = st.st_mode | kMode;
      if (chmod(path.c_str(), new_mode) != 0) {
        PDIE("chmod '%s'", path.c_str());
      }
    }
  }

  bool DelTree(const std::string &path, FileType file_type) {
    if (file_type != FILE_TYPE_DIRECTORY) {
      if (unlink(path.c_str()) != 0) {
#if !defined(__CYGWIN__)
        PDIE("unlinking '%s'", path.c_str());
#endif
        return false;
      }
      return true;
    }

    EnsureDirReadAndWritePerms(path);

    struct dirent *entry;
    DIR *dh = opendir(path.c_str());
    if (!dh) {
      PDIE("opendir '%s'", path.c_str());
    }
    errno = 0;
    while ((entry = readdir(dh)) != nullptr) {
      if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
      const std::string entry_path = path + '/' + entry->d_name;
      FileType entry_file_type = DentryToFileType(entry_path, entry);
      DelTree(entry_path, entry_file_type);
      errno = 0;
    }
    if (errno != 0) {
      PDIE("readdir '%s'", path.c_str());
    }
    closedir(dh);
    if (rmdir(path.c_str()) != 0) {
      PDIE("rmdir '%s'", path.c_str());
    }
    return true;
  }

 private:
  std::string output_base_;
  std::string output_filename_;
  std::string temp_filename_;

  FileInfoMap manifest_;
};

int main(int argc, char **argv) {
  argv0 = argv[0];

  argc--; argv++;
  bool allow_relative = false;
  bool use_metadata = false;

  while (argc >= 1) {
    if (strcmp(argv[0], "--allow_relative") == 0) {
      allow_relative = true;
      argc--; argv++;
    } else if (strcmp(argv[0], "--use_metadata") == 0) {
      use_metadata = true;
      argc--; argv++;
    } else {
      break;
    }
  }

  if (argc != 2) {
    fprintf(stderr, "usage: %s "
            "[--allow_relative] [--use_metadata] "
            "INPUT RUNFILES\n",
            argv0);
    return 1;
  }

  input_filename = argv[0];
  output_base_dir = argv[1];

  std::string manifest_file = input_filename;
  if (input_filename[0] != '/') {
    char cwd_buf[PATH_MAX];
    if (getcwd(cwd_buf, sizeof(cwd_buf)) == nullptr) {
      PDIE("getcwd failed");
    }
    manifest_file = std::string(cwd_buf) + '/' + manifest_file;
  }

  RunfilesCreator runfiles_creator(output_base_dir);
  runfiles_creator.ReadManifest(manifest_file, allow_relative, use_metadata);
  runfiles_creator.CreateRunfiles();

  return 0;
}
