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
// INTERNAL header file for use by C++ code in this package.

#ifndef BAZEL_SRC_MAIN_NATIVE_UNIX_JNI_H__
#define BAZEL_SRC_MAIN_NATIVE_UNIX_JNI_H__

#include <errno.h>
#include <jni.h>
#include <stdint.h>
#include <sys/stat.h>

#include <cstddef>
#include <string>

namespace blaze_jni {

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
// stat64 is deprecated on OS X/BSD.
typedef struct stat portable_stat_struct;
#define portable_stat ::stat
#define portable_lstat ::lstat
#else
typedef struct stat64 portable_stat_struct;
#define portable_stat ::stat64
#define portable_lstat ::lstat64
#endif

#if !defined(ENODATA)
# if defined(ENOATTR)
#  define ENODATA ENOATTR
# else
#  error We do not know how to handle missing ENODATA
# endif
#endif

// Posts a NativePosixFilesException to the current thread with the given
// POSIX error number and associated standard error message.
void PostNativePosixFilesException(JNIEnv* env, int saved_errno,
                                   const std::string& message);

// Returns the standard error message for a given UNIX error number.
std::string ErrorMessage(int error_number);

// Encoding for different timestamps in a struct stat.
enum StatTimes {
  STAT_ATIME,  // access
  STAT_MTIME,  // modification
  STAT_CTIME,  // status change
};

// Returns nanoseconds since Unix epoch from the given struct stat field.
int64_t StatEpochNanoseconds(const portable_stat_struct& statbuf, StatTimes t);

// Runs getxattr(2). If the attribute is not found or extended attributes are
// not supported by the filesystem, returns -1 and sets attr_not_found to true.
// For all other errors, returns -1, sets attr_not_found to false and leaves
// errno set to the error code returned by the system call.
ssize_t portable_getxattr(const char *path, const char *name, void *value,
                          size_t size, bool *attr_not_found);

// Runs lgetxattr(2). If the attribute is not found or extended attributes are
// not supported by the filesystem, returns -1 and sets attr_not_found to true.
// For all other errors, returns -1, sets attr_not_found to false and leaves
// errno set to the error code returned by the system call.
ssize_t portable_lgetxattr(const char *path, const char *name, void *value,
                           size_t size, bool *attr_not_found);

// Throws an exception of the given Java class name with the given message.
void PostException(JNIEnv* env, const char* exception_classname,
                   const std::string& message);

// Throws a java.lang.AssertionError with the given message.
void PostAssertionError(JNIEnv* env, const std::string& message);

// Registers a singleton IntConsumer JNI callback as a global reference.
// Throws AssertionError and returns false if already registered or if
// registration fails.
bool RegisterIntConsumerCallback(JNIEnv* env, jobject callback,
                                 jobject* callback_global_ref,
                                 const char* name);

// Invokes accept(int) on the given IntConsumer global reference if non-null.
void PerformIntegerValueCallback(jobject int_consumer_object, int value);

}  // namespace blaze_jni

#endif  // BAZEL_SRC_MAIN_NATIVE_UNIX_JNI_H__
