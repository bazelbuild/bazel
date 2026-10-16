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

#include <jni.h>
#include <string.h>

#include <string_view>

#include "src/main/cpp/util/logging.h"

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

namespace {

jclass GetClass(JNIEnv* env, const char* name) {
  jclass clazz = env->FindClass(name);
  BAZEL_CHECK_NE(clazz, nullptr);
  return static_cast<jclass>(env->NewGlobalRef(clazz));
}

jmethodID GetStaticMethodId(JNIEnv* env, jclass clazz, const char* name,
                            const char* sig) {
  jmethodID method = env->GetStaticMethodID(clazz, name, sig);
  BAZEL_CHECK_NE(method, nullptr);
  return method;
}

jfieldID GetFieldId(JNIEnv* env, jclass clazz, const char* name,
                    const char* sig) {
  jfieldID field = env->GetFieldID(clazz, name, sig);
  BAZEL_CHECK_NE(field, nullptr);
  return field;
}

void LogBadPath(JNIEnv* env, jstring jstr) {
  static const jclass cls = GetClass(
      env, "com/google/devtools/build/lib/unix/NativePosixFilesServiceImpl");
  static const jmethodID method =
      GetStaticMethodId(env, cls, "logBadPath", "(Ljava/lang/String;)V");
  env->CallStaticVoidMethod(cls, method, jstr);
}

char* GetStringLatin1Chars(JNIEnv* env, jstring jstr) {
  static const jclass String_class = GetClass(env, "java/lang/String");
  static const jfieldID String_coder_field =
      GetFieldId(env, String_class, "coder", "B");
  static const jfieldID String_value_field =
      GetFieldId(env, String_class, "value", "[B");
  static const jclass NullPointerException_class =
      GetClass(env, "java/lang/NullPointerException");

  if (jstr == nullptr) {
    env->ThrowNew(NullPointerException_class,
                  "JStringLatin1Holder: String should not be null");
    return nullptr;
  }

  jint len = env->GetStringLength(jstr);

  // Fast path for strings with a Latin1 coder, which all well-formed path
  // strings in Bazel ought to be.
  if (env->GetByteField(jstr, String_coder_field) == 0) {
    jobject jvalue = env->GetObjectField(jstr, String_value_field);
    if (jvalue == nullptr) {
      // Memory corruption?
      env->ThrowNew(NullPointerException_class,
                    "JStringLatin1Holder: String.value should not be null");
      return nullptr;
    }
    char* result = new char[len + 1];
    env->GetByteArrayRegion((jbyteArray)jvalue, 0, len, (jbyte*)result);
    result[len] = 0;
    return result;
  }

  // Fallback path for strings with a UTF-16 coder, which are not well-formed
  // but must be tolerated until all occurrences at Google have been fixed.
  // TODO(tjgq): Delete this fallback.
  LogBadPath(env, jstr);
  const jchar* str = env->GetStringCritical(jstr, nullptr);
  if (str == nullptr) {
    // OutOfMemoryError already set by GetStringCritical.
    BAZEL_CHECK_NE(env->ExceptionOccurred(), nullptr);
    return nullptr;
  }
  char* result = new char[len + 1];
  for (int i = 0; i < len; i++) {
    jchar unicode = str[i];  // (unsigned)
    result[i] = unicode <= 0x00ff ? unicode : '?';
  }
  env->ReleaseStringCritical(jstr, str);
  result[len] = 0;
  return result;
}

}  // namespace

JStringLatin1Holder::JStringLatin1Holder(JNIEnv* env, jstring string)
    : chars(GetStringLatin1Chars(env, string)) {}

JStringLatin1Holder::~JStringLatin1Holder() {
  if (chars != nullptr) {
    delete[] chars;
  }
}

JStringLatin1Holder::operator const char*() const { return chars; }

JStringLatin1Holder::operator std::string_view() const {
  return std::string_view(chars);
}

}  // namespace blaze_jni
