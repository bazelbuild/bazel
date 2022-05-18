#include <sys/stat.h>
#include <string.h>

#include <fs_attr.h>
#include <TypeConstants.h>

#include "src/main/native/unix_jni.h"

namespace blaze_jni {

using std::string;

string ErrorMessage(int error_number) {
  return string(strerror(error_number));
}

int portable_fstatat(int dirfd, char *name, portable_stat_struct *statbuf,
                     int flags) {
  return fstatat(dirfd, name, statbuf, flags);
}

int StatSeconds(const portable_stat_struct &statbuf, StatTimes t) {
  switch (t) {
    case STAT_ATIME:
      return statbuf.st_atim.tv_sec;
    case STAT_CTIME:
      return statbuf.st_ctim.tv_sec;
    case STAT_MTIME:
      return statbuf.st_mtim.tv_sec;
  }
  return 0;
}

int StatNanoSeconds(const portable_stat_struct &statbuf, StatTimes t) {
  switch (t) {
    case STAT_ATIME:
      return statbuf.st_atim.tv_nsec;
    case STAT_CTIME:
      return statbuf.st_ctim.tv_nsec;
    case STAT_MTIME:
      return statbuf.st_mtim.tv_nsec;
  }
  return 0;
}

static ssize_t read_attr(const char *path, const char *name, void *value,
                         size_t size, bool *attr_not_found, bool traverse) {
  attr_info info;
  ssize_t read;
  int fd;

  fd = open(path, O_RDONLY | (traverse ? 0 : O_NOTRAVERSE));
  if(fd < 0)
    return -1;

  if(fs_stat_attr(fd, name, &info) == B_OK)
    read = fs_read_attr(fd, name, info.type, 0, value, size);

  *attr_not_found = errno == B_ENTRY_NOT_FOUND;

  close(fd);
  return read;
}

ssize_t portable_getxattr(const char *path, const char *name, void *value,
                          size_t size, bool *attr_not_found) {
  return read_attr(path, name, value, size, attr_not_found, true);
}

ssize_t portable_lgetxattr(const char *path, const char *name, void *value,
                           size_t size, bool *attr_not_found) {
  return read_attr(path, name, value, size, attr_not_found, false);
}

int portable_sysctlbyname(const char *name_chars, void *mibp, size_t *sizep) {
  // I don't think Haiku has sysctl.
  errno = ENOSYS;
  return -1;
}

int portable_push_disable_sleep() {
  // Currently not implemented.
  return -1;
}

int portable_pop_disable_sleep() {
  // Currently not implemented.
  return -1;
}

void portable_start_suspend_monitoring() {
  // Currently not implemented.
}

void portable_start_thermal_monitoring() {
  // Currently not implemented.
}

int portable_thermal_load() {
  // Currently not implemented.
  return 0;
}

void portable_start_system_load_advisory_monitoring() {
  // Currently not implemented.
}

int portable_system_load_advisory() {
  // Currently not implemented.
  return 0;
}

void portable_start_memory_pressure_monitoring() {
  // Currently not implemented.
}

MemoryPressureLevel portable_memory_pressure() {
  // Currently not implemented.
  return MemoryPressureLevelNormal;
}

void portable_start_disk_space_monitoring() {
  // Currently not implemented.
}

void portable_start_cpu_speed_monitoring() {
  // Currently not implemented.
}

int portable_cpu_speed() {
  // Currently not implemented.
  return -1;
}

}  // namespace blaze_jni
