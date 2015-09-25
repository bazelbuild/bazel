// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.unix;

import com.google.devtools.build.lib.UnixJniLoader;

import java.io.Closeable;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.SocketAddress;
import java.net.SocketImpl;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A simple implementation of SocketImpl for sockets that wrap a UNIX
 * file-descriptor.  This SocketImpl assumes that the socket is already
 * created, bound, connected and supports no socket options or out-of-band
 * features.  This is used to implement server-side accepted client sockets
 * (i.e. those returned by {@link LocalServerSocket#accept}).
 */
class LocalSocketImpl extends SocketImpl {
  private static final Logger logger =
      Logger.getLogger(LocalSocketImpl.class.getName());

  static {
    UnixJniLoader.loadJni();
    init();
  }

  // The logic here is a little twisted, to support JDK7 and JDK8.

  // 1) In JDK7, the FileDescriptor class keeps a reference count of
  //    instances using the fd, and closes it when it goes to 0.  The
  //    reference count is only decremented by the finalizer for a
  //    given class.  When the call to close() happens, the fd is
  //    closed regardless of the current state of the refcount.
  //
  // 2) In JDK8, every instance that uses the fd registers a Closeable
  //    with the FileDescriptor.  Since the FileDescriptor has a
  //    reference to every user, only when all of the users and the
  //    FileDescriptor get GC'd does the finalizer run.  An explicit
  //    call to close() calls FileDescriptor.closeAll(), which
  //    force-closes all of the users.

  // So, in our case:

  // 1) ref() increments the refcount in JDK7, and registers with the
  //    FD in JDK8.

  // 2) unref() decrements the refcount in JDK7, and does nothing in
  //    JDK8.

  // 3) The finalizer decrements the refcount in JDK7, and simply
  //    calls close() in JDK8 (where we don't have to worry about
  //    multiple live users of the FD).  The close() method itself is
  //    idempotent.

  // 4) close() calls fd.closeAll in JDK8, which, in turn, calls
  //    closer.close().  In JDK7, close() calls closer.close()
  //    explicitly.
  private static native void init();
  private static native void ref(FileDescriptor fd, Closeable closeable);
  private static native boolean unref(FileDescriptor fd);
  private static native boolean close0(FileDescriptor fd, Closeable closeable);

  private final boolean isInitialized;
  private final Closeable closer = new Closeable() {
      AtomicBoolean isClosed = new AtomicBoolean(false);
      @Override public void close() throws IOException {
        if (isClosed.compareAndSet(false, true)) {
          LocalSocket.close(fd);
        }
      }
    };

  // Note to callers: if you pass a FD into this constructor, this
  // instance is now responsible for closing it (in the sense of
  // LocalSocket.close()).  If some other instance tries to close it,
  // then terrible things will happen.
  LocalSocketImpl(FileDescriptor fd) {
    this.fd = fd; // (inherited field)
    ref(fd, closer);
    isInitialized = true;
  }

  @Override protected void finalize() {
    try {
      if (isInitialized) {
        if (!unref(fd)) {
          // JDK8 codepath
          close0(fd, closer);
        }
      }
    } catch (Exception e) {
      logger.log(Level.WARNING, "Unable to access FileDescriptor class - " +
          "may cause a file descriptor leak", e);
    }
  }
  @Override protected InputStream getInputStream() {
    return new FileInputStream(getFileDescriptor());
  }
  @Override protected OutputStream getOutputStream() {
    return new FileOutputStream(getFileDescriptor());
  }
  @Override protected void close() throws IOException {
    if (fd.valid()) {
      if (!close0(fd, closer)) {
        // JDK7 codepath
        closer.close();
      }
    }
  }

  // Unused:
  @Override
  public void setOption(int optID, Object value)  {
    throw new UnsupportedOperationException("setOption");
  }
  @Override
  public Object getOption(int optID) {
    throw new UnsupportedOperationException("getOption");
  }
  @Override protected void create(boolean stream) {
    throw new UnsupportedOperationException("create");
  }
  @Override protected void connect(String host, int port) {
    throw new UnsupportedOperationException("connect");
  }
  @Override protected void connect(InetAddress address, int port) {
    throw new UnsupportedOperationException("connect2");
  }
  @Override protected void connect(SocketAddress address, int timeout) {
    throw new UnsupportedOperationException("connect3");
  }
  @Override protected void bind(InetAddress host, int port) {
    throw new UnsupportedOperationException("bind");
  }
  @Override protected void listen(int backlog) {
    throw new UnsupportedOperationException("listen");
  }
  @Override protected void accept(SocketImpl s) {
    throw new UnsupportedOperationException("accept");
  }
  @Override protected int available() {
    throw new UnsupportedOperationException("available");
  }
  @Override protected void sendUrgentData(int i) {
    throw new UnsupportedOperationException("sendUrgentData");
  }
}
