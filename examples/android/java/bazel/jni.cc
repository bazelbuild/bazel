#include <jni.h>

#include "examples/android/java/bazel/jni_dep.h"

const char* hello = "Hello JNI";

extern "C" JNIEXPORT jstring JNICALL
Java_bazel_Jni_hello(JNIEnv *env, jclass clazz) {
  return NewStringLatin1(env, hello);
}
