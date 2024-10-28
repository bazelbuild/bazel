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

#include "src/main/native/latin1_jni_path.h"

#include <string.h>

namespace blaze_jni {

jstring NewStringLatin1(JNIEnv *env, const char *str) {
  int len = strlen(str);
  jchar buf[512];
  jchar *str1;

  if (len > 512) {
    str1 = new jchar[len];
  } else {
    str1 = buf;
  }

  for (int i = 0; i < len; i++) {
    str1[i] = (unsigned char)str[i];
  }
  jstring result = env->NewString(str1, len);
  if (str1 != buf) {
    delete[] str1;
  }
  return result;
}

char *GetStringLatin1Chars(JNIEnv *env, jstring jstr) {
  static jclass String_class = env->FindClass("java/lang/String");
  static jfieldID String_coder_field = env->GetFieldID(String_class, "coder", "B");
  static jfieldID String_value_field = env->GetFieldID(String_class, "value", "[B");

  // All path strings used in Bazel are encoded as raw bytes with a Latin1
  // coder.
  if (env->GetByteField(jstr, String_coder_field) != 0) {
    return nullptr;
  }
  jint len = env->GetStringLength(jstr);
  char *result = new char[len + 1];
  if (jobject jvalue = env->GetObjectField(jstr, String_value_field)) {
    env->GetByteArrayRegion((jbyteArray)jvalue, 0, len, (jbyte *)result);
  } else {
    return nullptr;
  }
  result[len] = 0;
  return result;
}

/**
 * Release the Latin1 chars returned by a prior call to
 * GetStringLatin1Chars.
 */
void ReleaseStringLatin1Chars(const char *s) { delete[] s; }

}  // namespace blaze_jni
