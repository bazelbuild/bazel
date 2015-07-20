#include <jni.h>
#include <stdlib.h>
#include <string.h>

const char* hello = "Hello JNI";

static jstring NewStringLatin1(JNIEnv *env, const char *str) {
  int len = strlen(str);
  jchar *str1;
  str1 = reinterpret_cast<jchar *>(malloc(len * sizeof(jchar)));

  for (int i = 0; i < len ; i++) {
    str1[i] = (unsigned char) str[i];
  }
  jstring result = env->NewString(str1, len);
  free(str1);
  return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_bazel_Jni_hello(JNIEnv *env, jclass clazz) {
  return NewStringLatin1(env, hello);
}
