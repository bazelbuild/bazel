// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <CoreServices/CoreServices.h>
#include <jni.h>
#include <pthread.h>
#include <stdlib.h>
#include <list>
#include <string>

// A structure to pass around the FSEvents info and the list of paths.
struct JNIEventsDiffAwareness {
  // FSEvents run loop (thread)
  CFRunLoopRef runLoop;
  // FSEvents stream reference (reference to the listened stream)
  FSEventStreamRef stream;
  // List of paths that have been changed since last polling
  std::list<std::string> paths;
  // Mutex to protect concurrent access of paths.
  // FsEventsDiffAwarenessCallback fill that list which is emptied
  // by the MacOSXEventsDiffAwareness#poll() method.
  // The former is called inside the FsEvents run loop and the latter
  // from Java threads.
  pthread_mutex_t mutex;

  JNIEventsDiffAwareness() { pthread_mutex_init(&mutex, nullptr); }

  ~JNIEventsDiffAwareness() { pthread_mutex_destroy(&mutex); }
};

// Callback called when an event is reported by the FSEvents API
void FsEventsDiffAwarenessCallback(ConstFSEventStreamRef streamRef,
                                   void *clientCallBackInfo, size_t numEvents,
                                   void *eventPaths,
                                   const FSEventStreamEventFlags eventFlags[],
                                   const FSEventStreamEventId eventIds[]) {
  /* We are just returning the list of modified path but we could return a bit
   * more information,
   * see
   * https://developer.apple.com/library/mac/documentation/Darwin/Reference/FSEvents_Ref/#//apple_ref/doc/c_ref/FSEventStreamCallback
   * If we ever do more, we should be careful because creation and deletion
   * event get coaslesced
   * into one single entry.
   */
  char **paths = static_cast<char **>(eventPaths);

  JNIEventsDiffAwareness *info =
      static_cast<JNIEventsDiffAwareness *>(clientCallBackInfo);
  pthread_mutex_lock(&(info->mutex));
  for (int i = 0; i < numEvents; i++) {
    info->paths.push_back(std::string(paths[i]));
  }
  pthread_mutex_unlock(&(info->mutex));
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_skyframe_MacOSXFsEventsDiffAwareness_create(
    JNIEnv *env, jobject fsEventsDiffAwareness, jobjectArray paths,
    jdouble latency) {
  // Create a FSEventStreamContext to pass around (env, fsEventsDiffAwareness)
  JNIEventsDiffAwareness *info = new JNIEventsDiffAwareness();

  FSEventStreamContext context;
  context.version = 0;
  context.info = static_cast<void *>(info);
  context.retain = NULL;
  context.release = NULL;
  context.copyDescription = NULL;

  // Create an CFArrayRef of CFStringRef from the Java array of String
  jsize length = env->GetArrayLength(paths);
  CFStringRef *pathsArray = new CFStringRef[length];
  for (int i = 0; i < length; i++) {
    jstring path = (jstring)env->GetObjectArrayElement(paths, i);
    const char *pathCStr = env->GetStringUTFChars(path, NULL);
    pathsArray[i] =
        CFStringCreateWithCString(NULL, pathCStr, kCFStringEncodingUTF8);
    env->ReleaseStringUTFChars(path, pathCStr);
  }
  CFArrayRef pathsToWatch =
      CFArrayCreate(NULL, (const void **)pathsArray, 1, NULL);
  delete[] pathsArray;
  info->stream = FSEventStreamCreate(
      NULL, &FsEventsDiffAwarenessCallback, &context, pathsToWatch,
      kFSEventStreamEventIdSinceNow, static_cast<CFAbsoluteTime>(latency),
      kFSEventStreamCreateFlagNoDefer | kFSEventStreamCreateFlagFileEvents);

  // Save the info pointer to FSEventsDiffAwareness#nativePointer
  jbyteArray array = env->NewByteArray(sizeof(info));
  env->SetByteArrayRegion(array, 0, sizeof(info),
                          reinterpret_cast<const jbyte *>(&info));
  jclass clazz = env->GetObjectClass(fsEventsDiffAwareness);
  jfieldID fid = env->GetFieldID(clazz, "nativePointer", "J");
  env->SetLongField(fsEventsDiffAwareness, fid, reinterpret_cast<jlong>(info));
}

JNIEventsDiffAwareness *GetInfo(JNIEnv *env, jobject fsEventsDiffAwareness) {
  jclass clazz = env->GetObjectClass(fsEventsDiffAwareness);
  jfieldID fid = env->GetFieldID(clazz, "nativePointer", "J");
  jlong field = env->GetLongField(fsEventsDiffAwareness, fid);
  return reinterpret_cast<JNIEventsDiffAwareness *>(field);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_skyframe_MacOSXFsEventsDiffAwareness_run(
    JNIEnv *env, jobject fsEventsDiffAwareness) {
  JNIEventsDiffAwareness *info = GetInfo(env, fsEventsDiffAwareness);
  info->runLoop = CFRunLoopGetCurrent();
  FSEventStreamScheduleWithRunLoop(info->stream, info->runLoop,
                                   kCFRunLoopDefaultMode);
  FSEventStreamStart(info->stream);
  CFRunLoopRun();
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_google_devtools_build_lib_skyframe_MacOSXFsEventsDiffAwareness_poll(
    JNIEnv *env, jobject fsEventsDiffAwareness) {
  JNIEventsDiffAwareness *info = GetInfo(env, fsEventsDiffAwareness);
  pthread_mutex_lock(&(info->mutex));

  jclass classString = env->FindClass("java/lang/String");
  jobjectArray result =
      env->NewObjectArray(info->paths.size(), classString, NULL);
  int i = 0;
  for (auto it = info->paths.begin(); it != info->paths.end(); it++, i++) {
    env->SetObjectArrayElement(result, i, env->NewStringUTF(it->c_str()));
  }
  info->paths.clear();
  pthread_mutex_unlock(&(info->mutex));
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_skyframe_MacOSXFsEventsDiffAwareness_doClose(
    JNIEnv *env, jobject fsEventsDiffAwareness) {
  JNIEventsDiffAwareness *info = GetInfo(env, fsEventsDiffAwareness);
  CFRunLoopStop(info->runLoop);
  FSEventStreamStop(info->stream);
  FSEventStreamUnscheduleFromRunLoop(info->stream, info->runLoop,
                                     kCFRunLoopDefaultMode);
  FSEventStreamInvalidate(info->stream);
  FSEventStreamRelease(info->stream);
  delete info;
}
