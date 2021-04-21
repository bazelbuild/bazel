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

// POSIX support for Starlark CPU profiler.

#include <arpa/inet.h>  // for htonl
#include <errno.h>
#include <fcntl.h>
#include <jni.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace cpu_profiler {

// static native boolean supported();
extern "C" JNIEXPORT jboolean JNICALL
Java_net_starlark_java_eval_CpuProfiler_supported(JNIEnv *env, jclass clazz) {
  return true;
}

static int fd;  // the write end of the profile event pipe

pid_t gettid(void) {
#ifdef __linux__
  return (pid_t)syscall(SYS_gettid);
#else  // darwin
  return (pid_t)syscall(SYS_thread_selfid);
#endif
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
  pid_t tid = gettid();
  uint32_t tid_be = htonl(tid);
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

// static native jint gettid();
extern "C" JNIEXPORT jint JNICALL
Java_net_starlark_java_eval_CpuProfiler_gettid(JNIEnv *env, jclass clazz) {
  return gettid();
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

// static native FileDescriptor createPipe();
extern "C" JNIEXPORT jobject JNICALL
Java_net_starlark_java_eval_CpuProfiler_createPipe(JNIEnv *env, jclass clazz) {
  // Create a pipe for profile events from the handler to Java.
  // The default pipe size is 64KiB on Linux and 16KiB on Mac OS X.
  int pipefds[2];
  if (pipe(pipefds) < 0) {
    perror("pipe");
    abort();
  }
  fd = pipefds[1];

  // Make the write end non-blocking so that the signal
  // handler can detect overflow (rather than deadlock).
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

  // Return the read end of the event pipe,
  // wrapped by a java FileDescriptor.
  return makeFD(env, pipefds[0]);
}

// static native boolean startTimer(long period_micros);
extern "C" JNIEXPORT jboolean JNICALL
Java_net_starlark_java_eval_CpuProfiler_startTimer(JNIEnv *env, jclass clazz,
                                                   jlong period_micros) {
  // Install the signal handler.
  // Use sigaction(2) not signal(2) so that we can correctly
  // restore the previous handler if necessary.
  struct sigaction oldact = {}, act = {};
  act.sa_handler = onsigprof;
  act.sa_flags = SA_RESTART;  // the JVM doesn't expect EINTR
  if (sigaction(SIGPROF, &act, &oldact) < 0) {
    perror("sigaction");
    abort();
  }

  // Is a handler already in effect?
  // Check for 3-arg and 1-arg forms.
  typedef void (*sighandler_t)(int);  // don't rely on this GNU extension
  sighandler_t prev = (oldact.sa_flags & SA_SIGINFO) != 0
                          ? reinterpret_cast<sighandler_t>(oldact.sa_sigaction)
                          : oldact.sa_handler;
  // The initial handler (DFL or IGN) may vary by thread package.
  if (prev != SIG_DFL && prev != SIG_IGN) {
    // Someone else is profiling this JVM.
    // Restore their handler and fail.
    (void)sigaction(SIGPROF, &oldact, nullptr);
    return false;
  }

  // Start the CPU interval timer.
  struct timeval period = {
      .tv_sec = 0,
      .tv_usec = static_cast<suseconds_t>(period_micros),
  };
  struct itimerval timer = {.it_interval = period, .it_value = period};
  if (setitimer(ITIMER_PROF, &timer, nullptr) < 0) {
    perror("setitimer");
    abort();
  }

  return true;
}

// static native void stopTimer();
extern "C" JNIEXPORT void JNICALL
Java_net_starlark_java_eval_CpuProfiler_stopTimer(JNIEnv *env, jclass clazz) {
  // Disarm the CPU interval timer.
  struct itimerval timer = {};
  if (setitimer(ITIMER_PROF, &timer, nullptr) < 0) {
    perror("setitimer");
    abort();
  }

  // Uninstall signal handler.
  signal(SIGPROF, SIG_IGN);
}

}  // namespace cpu_profiler
