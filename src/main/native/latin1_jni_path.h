// Copyright 2019 The Bazel Authors. All rights reserved.
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

#ifndef THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_LATIN1_JNI_PATH_H_
#define THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_LATIN1_JNI_PATH_H_

#include <jni.h>

// Latin1 <--> java.lang.String conversion functions.
// Derived from similar routines in Sun JDK.  See:
// j2se/src/solaris/native/java/io/UnixFileSystem_md.c
// j2se/src/share/native/common/jni_util.c
//
// Like the Sun JDK in its usual configuration, we assume all UNIX
// filenames are Latin1 encoded.

/**
 * Returns a new Java String for the specified Latin1 characters.
 */
jstring NewStringLatin1(JNIEnv *env, const char *str);

/**
 * Returns a nul-terminated Latin1-encoded byte array for the
 * specified Java string, or null on failure.  Unencodable characters
 * are replaced by '?'.  Must be followed by a call to
 * ReleaseStringLatin1Chars.
 */
char *GetStringLatin1Chars(JNIEnv *env, jstring jstr);

/**
 * Release the Latin1 chars returned by a prior call to
 * GetStringLatin1Chars.
 */
void ReleaseStringLatin1Chars(const char *s);

#endif  // THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_LATIN1_JNI_PATH_H_
