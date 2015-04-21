// Copyright 2014 Google Inc. All rights reserved.
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

#include <jni.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <string>

#include "src/main/native/unix_jni.h"

// Returns the field ID for FileDescriptor.fd.
static jfieldID GetFileDescriptorField(JNIEnv *env) {
  // See http://java.sun.com/docs/books/jni/html/fldmeth.html#26855
  static jclass fd_class = NULL;
  if (fd_class == NULL) { /* note: harmless race condition */
    jclass local = env->FindClass("java/io/FileDescriptor");
    CHECK(local != NULL);
    fd_class = static_cast<jclass>(env->NewGlobalRef(local));
  }
  static jfieldID fieldId = NULL;
  if (fieldId == NULL) { /* note: harmless race condition */
    fieldId = env->GetFieldID(fd_class, "fd", "I");
    CHECK(fieldId != NULL);
  }
  return fieldId;
}

// Returns the UNIX filedescriptor from a java.io.FileDescriptor instance.
static jint GetUnixFileDescriptor(JNIEnv *env, jobject fd_obj) {
  return env->GetIntField(fd_obj, GetFileDescriptorField(env));
}

// Sets the UNIX filedescriptor of a java.io.FileDescriptor instance.
static void SetUnixFileDescriptor(JNIEnv *env, jobject fd_obj, jint fd) {
  env->SetIntField(fd_obj, GetFileDescriptorField(env), fd);
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    socket
 * Signature: (Ljava/io/FileDescriptor;)V
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_socket(JNIEnv *env,
                                               jclass clazz,
                                               jobject fd_sock) {
  int sock;
  if ((sock = ::socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
    ::PostException(env, errno, ::ErrorMessage(errno));
    return;
  }
  SetUnixFileDescriptor(env, fd_sock, sock);
}

// Initialize "addr" from "name_chars", reporting error and returning
// false on failure.
static bool InitializeSockaddr(JNIEnv *env,
                               struct sockaddr_un *addr,
                               const char* name_chars) {
  memset(addr, 0, sizeof *addr);
  addr->sun_family = AF_UNIX;
  // Note: UNIX_PATH_MAX is quite small!
  if (strlen(name_chars) >= sizeof(addr->sun_path)) {
    ::PostFileException(env, ENAMETOOLONG, name_chars);
    return false;
  }
  strcpy((char*) &addr->sun_path, name_chars);
  return true;
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    bind
 * Signature: (Ljava/io/FileDescriptor;Ljava/lang/String;)V
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_bind(JNIEnv *env,
                                             jclass clazz,
                                             jobject fd_svr,
                                             jstring name) {
  int svr_sock = GetUnixFileDescriptor(env, fd_svr);
  const char* name_chars = env->GetStringUTFChars(name, NULL);
  struct sockaddr_un addr;
  if (InitializeSockaddr(env, &addr, name_chars) &&
      ::bind(svr_sock, (struct sockaddr *) &addr, sizeof addr) < 0) {
    ::PostException(env, errno, ::ErrorMessage(errno));
  }
  env->ReleaseStringUTFChars(name, name_chars);
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    listen
 * Signature: (Ljava/io/FileDescriptor;I)V
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_listen(JNIEnv *env,
                                               jclass clazz,
                                               jobject fd_svr,
                                               jint backlog) {
  int svr_sock = GetUnixFileDescriptor(env, fd_svr);
  if (::listen(svr_sock, backlog) < 0) {
    ::PostException(env, errno, ::ErrorMessage(errno));
  }
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    select
 * Signature: (L[java/io/FileDescriptor;[java/io/FileDescriptor;[java/io/FileDescriptor;J)Ljava/io/FileDescriptor
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_poll(JNIEnv *env,
                                               jclass clazz,
                                               jobject rfds_svr,
                                               jlong timeoutMillis) {
  // TODO(bazel-team): Handle Unix signals, namely SIGTERM.

  // Copy Java FD into pollfd
  pollfd pollfd;
  pollfd.fd = GetUnixFileDescriptor(env, rfds_svr);
  pollfd.events = POLLIN;
  pollfd.revents = 0;

  int count = poll(&pollfd, 1, timeoutMillis);
  if (count == 0) {
    // throws a timeout exception.
    ::PostException(env, ETIMEDOUT, ::ErrorMessage(ETIMEDOUT));
  } else if (count < 0) {
    ::PostException(env, errno, ::ErrorMessage(errno));
  }
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    accept
 * Signature: (Ljava/io/FileDescriptor;Ljava/io/FileDescriptor;)V
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_accept(JNIEnv *env,
                                               jclass clazz,
                                               jobject fd_svr,
                                               jobject fd_cli) {
  int svr_sock = GetUnixFileDescriptor(env, fd_svr);
  int cli_sock;
  if ((cli_sock = ::accept(svr_sock, NULL, NULL)) < 0) {
    ::PostException(env, errno, ::ErrorMessage(errno));
    return;
  }
  SetUnixFileDescriptor(env, fd_cli, cli_sock);
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    close
 * Signature: (Ljava/io/FileDescriptor;)V
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_close(JNIEnv *env,
                                              jclass clazz,
                                              jobject fd_svr) {
  int svr_sock = GetUnixFileDescriptor(env, fd_svr);
  if (::close(svr_sock) < 0) {
    ::PostException(env, errno, ::ErrorMessage(errno));
  }
  SetUnixFileDescriptor(env, fd_svr, -1);
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    connect
 * Signature: (Ljava/io/FileDescriptor;Ljava/lang/String;)V
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_connect(JNIEnv *env,
                                                jclass clazz,
                                                jobject fd_cli,
                                                jstring name) {
  const char* name_chars = env->GetStringUTFChars(name, NULL);
  jint cli_sock = GetUnixFileDescriptor(env, fd_cli);
  if (cli_sock == -1) {
    ::PostFileException(env, ENOTSOCK, name_chars);
  } else {
    struct sockaddr_un addr;
    if (InitializeSockaddr(env, &addr, name_chars)) {
      if (::connect(cli_sock, (struct sockaddr *) &addr, sizeof addr) < 0) {
        ::PostException(env, errno, ::ErrorMessage(errno));
      }
    }
  }
  env->ReleaseStringUTFChars(name, name_chars);
}

/*
 * Class:     com.google.devtools.build.lib.unix.LocalSocket
 * Method:    shutdown()
 * Signature: (Ljava/io/FileDescriptor;I)V
 * Parameters: code: 0 to shutdown input and 1 to shutdown output.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocket_shutdown(JNIEnv *env,
                                                 jclass clazz,
                                                 jobject fd,
                                                 jint code) {
  int action;
  if (code == 0) {
    action = SHUT_RD;
  } else {
    CHECK(code == 1);
    action = SHUT_WR;
  }

  int sock = GetUnixFileDescriptor(env, fd);
  if (shutdown(sock, action) < 0) {
    ::PostException(env, errno, ::ErrorMessage(errno));
  }
}

// TODO(bazel-team): These methods were removed in JDK8, so they
// can be removed when we are no longer using JDK7.  See note in
// LocalSocketImpl.
static jmethodID increment_use_count_;
static jmethodID decrement_use_count_;

// >=JDK8
static jmethodID fd_attach_;
static jmethodID fd_close_all_;

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocketImpl_init(JNIEnv *env, jclass ignored) {
  jclass cls = env->FindClass("java/io/FileDescriptor");
  if (cls == NULL) {
    cls = env->FindClass("java/lang/NoClassDefFoundError");
    env->ThrowNew(cls, "FileDescriptor class not found");
    return;
  }

  // JDK7
  increment_use_count_ =
      env->GetMethodID(cls, "incrementAndGetUseCount", "()I");
  if (increment_use_count_ != NULL) {
    decrement_use_count_ =
        env->GetMethodID(cls, "decrementAndGetUseCount", "()I");
  } else {
    // JDK8
    env->ExceptionClear();  // The pending exception from increment_use_count_

    fd_attach_ = env->GetMethodID(cls, "attach", "(Ljava/io/Closeable;)V");
    fd_close_all_ = env->GetMethodID(cls, "closeAll", "(Ljava/io/Closeable;)V");

    if (fd_attach_ == NULL || fd_close_all_ == NULL) {
      cls = env->FindClass("java/lang/NoSuchMethodError");
      env->ThrowNew(cls, "FileDescriptor methods not found");
      return;
    }
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocketImpl_ref(JNIEnv *env, jclass clazz,
                                                jobject fd, jobject closer) {
  if (increment_use_count_ != NULL) {
    env->CallIntMethod(fd, increment_use_count_);
  }

  if (fd_attach_ != NULL) {
    env->CallVoidMethod(fd, fd_attach_, closer);
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocketImpl_unref(JNIEnv *env, jclass clazz,
                                                  jobject fd) {
  if (decrement_use_count_ != NULL) {
    env->CallIntMethod(fd, decrement_use_count_);
    return true;
  }
  return false;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_unix_LocalSocketImpl_close0(JNIEnv *env, jclass clazz,
                                                   jobject fd,
                                                   jobject closeable) {
  if (fd_close_all_ != NULL) {
    env->CallVoidMethod(fd, fd_close_all_, closeable);
    return true;
  }
  // This will happen if fd_close_all_ is NULL, which means we are running in
  // <=JDK7, which means that the caller needs to invoke close() explicitly.
  return false;
}
