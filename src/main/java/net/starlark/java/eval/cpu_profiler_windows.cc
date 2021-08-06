// Copyright 2020 The Bazel Authors. All rights reserved.
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

// Starlark CPU profiler stubs for unsupported platforms.

#include <fcntl.h>
#include <io.h>
#include <jni.h>
#include <stdlib.h>
#include <windows.h>

// We need this in order to use SetTimer and KillTimer.
#pragma comment (lib, "User32.lib")

// We need this to use htonl.
#pragma comment(lib, "Ws2_32.lib")

namespace cpu_profiler {

    static int fd;  // the write end of the profile event pipe

    static UINT_PTR timer;

    extern "C" JNIEXPORT jboolean JNICALL
    Java_net_starlark_java_eval_CpuProfiler_supported(JNIEnv *env, jclass clazz) {
      return true;
    }

    int getId() {
      return GetCurrentProcessId();
    }

    extern "C" JNIEXPORT jint JNICALL
    Java_net_starlark_java_eval_CpuProfiler_gettid(JNIEnv *env, jclass clazz) {
      return getId();
    }

    // makeFD: return new FileDescriptor(fd)
    //
    // This would be easy to do in Java, but for the field being private.
    // Java really does everything it can to make system programming hateful.
    static jobject makeFD(JNIEnv *env, int fd) {
      jclass fdclass = env->FindClass("java/io/FileDescriptor");
      if (fdclass == nullptr) return nullptr;  // exception

      jmethodID init = env->GetMethodID(fdclass, "<init>", "()V");
      if (init == nullptr) return nullptr;  // exception
      jobject fdobj = env->NewObject(fdclass, init);

      jfieldID fd_field = env->GetFieldID(fdclass, "fd", "I");
      if (fd_field == nullptr) return nullptr;  // exception
      env->SetIntField(fdobj, fd_field, fd);

      return fdobj;
    }

    extern "C" JNIEXPORT jobject JNICALL
    Java_net_starlark_java_eval_CpuProfiler_createPipe(JNIEnv *env, jclass clazz) {

      /* set up security attributes to allow pipes to be inherited */
      SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};

      HANDLE readPipe = CreateNamedPipeA("readPipe",
              /* read */ 0x00000001 | /* allow overlapping */ 0x40000000,
              /* write as bytes */ 0x00000000,
              /* instances */ 1,
              /* system default buffer size for out */ 0,
              /* system default buffer size for in */ 0,
              /* 50 milliseconds */ 0,
              &sa);
      HANDLE writePipe = CreateNamedPipeA("writePipe",
              /* write */ 0x00000002 | /* allow overlapping */ 0x40000000,
              /* write as bytes */ 0x00000000,
              /* instances */ 1,
              /* system default buffer size for out */ 0,
              /* system default buffer size for in */ 0,
              /* 50 milliseconds */ 0,
              &sa);

      fd = _open_osfhandle((long)writePipe, _O_RDWR);

      return makeFD(env, _open_osfhandle((long)readPipe, _O_RDWR));
    }

    // SIGPROF handler.
    // Warning: asynchronous! See signal-safety(7) for the programming discipline.
    void onsigprof(int sig) {
      int old_errno = errno;

      if (fd == 0) {
        const char *msg = "startTimer called before createPipe\n";
        write(2, msg, strlen(msg));
        abort();
      }

      // Send an event containing the int32be-encoded OS thread ID.
      int tid = getId();
      int tid_be = htonl(tid);
      int r = write(fd, (void *)&tid_be, sizeof tid_be);
      if (r < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          // The Java router thread cannot keep up.
          //
          // A busy 12-core machine receives 12 * 100Hz = 1200 signals per second,
          // and thus writes 4.8KB/s to the pipe. The default pipe buffer
          // size on Linux is 64KiB, sufficient to buffer ~14s of data.
          // (It is a quarter of that on Mac OS X.)
          //
          // Rather than block in write(2), causing the JVM to deadlock,
          // we print an error and discard the event.
          const char *msg =
                  "Starlark profile event router thread cannot keep up; discarding "
                  "events\n";
          write(2, msg, strlen(msg));
        } else {
          // We shouldn't use perror in a signal handler.
          // Strictly, we shouldn't use strerror either,
          // but for all errors returned by write it merely
          // returns a constant.
          char buf[1024] = "write: ";
          strncat(buf, strerror(errno), sizeof buf - strlen(buf) - 1);
          strncat(buf, "\n", sizeof buf - strlen(buf) - 1);
          write(2, buf, strlen(buf));
          abort();
        }
      }

      errno = old_errno;
    }


    extern "C" JNIEXPORT jboolean JNICALL
    Java_net_starlark_java_eval_CpuProfiler_startTimer(JNIEnv *env, jclass clazz,
                                                       jlong period_micros) {

      // Start the CPU interval timer.
      timer = SetTimer(nullptr, 0, period_micros, nullptr);
      if (timer == 0) {
        perror("setitimer");
        abort();
      }

      return true;
    }

    extern "C" JNIEXPORT void JNICALL
    Java_net_starlark_java_eval_CpuProfiler_stopTimer(JNIEnv *env, jclass clazz) {
      // Disarm the CPU interval timer.
      if (KillTimer(nullptr, timer)) {
          perror("killtimer");
          abort();
      }
    }

}  // namespace cpu_profiler