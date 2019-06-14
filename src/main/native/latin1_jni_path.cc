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

static jfieldID String_coder_field;
static jfieldID String_value_field;

static bool CompactStringsEnabled(JNIEnv *env) {
  if (jclass klass = env->FindClass("java/lang/String")) {
    if (jfieldID csf = env->GetStaticFieldID(klass, "COMPACT_STRINGS", "Z")) {
      if (env->GetStaticBooleanField(klass, csf)) {
        if ((String_coder_field = env->GetFieldID(klass, "coder", "B"))) {
          if ((String_value_field = env->GetFieldID(klass, "value", "[B"))) {
            return true;
          }
        }
      }
    }
  }
  env->ExceptionClear();
  return false;
}

char *GetStringLatin1Chars(JNIEnv *env, jstring jstr) {
  jint len = env->GetStringLength(jstr);

  // Fast path for latin1 strings.
  static bool cs_enabled = CompactStringsEnabled(env);
  const int LATIN1 = 0;
  if (cs_enabled && env->GetByteField(jstr, String_coder_field) == LATIN1) {
    char *result = new char[len + 1];
    if (jobject jvalue = env->GetObjectField(jstr, String_value_field)) {
      env->GetByteArrayRegion((jbyteArray)jvalue, 0, len, (jbyte *)result);
    }
    result[len] = 0;
    return result;
  }

  const jchar *str = env->GetStringCritical(jstr, NULL);
  if (str == NULL) {
    return NULL;
  }

  char *result = new char[len + 1];
  for (int i = 0; i < len; i++) {
    jchar unicode = str[i];  // (unsigned)
    result[i] = unicode <= 0x00ff ? unicode : '?';
  }

  result[len] = 0;
  env->ReleaseStringCritical(jstr, str);
  return result;
}

/**
 * Release the Latin1 chars returned by a prior call to
 * GetStringLatin1Chars.
 */
void ReleaseStringLatin1Chars(const char *s) { delete[] s; }
