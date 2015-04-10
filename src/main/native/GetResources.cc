#include <jni.h>
#include <stdio.h>
#include <unistd.h>
#include "GetResources.h"

JNIEXPORT jint JNICALL Java_GetResources_getNumProcessors (JNIEnv *, jobject)
{
  return sysconf(_SC_NPROCESSORS_ONLN);
}

JNIEXPORT jlong JNICALL Java_GetResources_getMemoryAvailable (JNIEnv *, jobject)
{
  // sysconf(_SC_PHYS_PAGES) apparently does not work on Mac? So returning 3000.
  return 3000;
}
