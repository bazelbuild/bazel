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
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.SocketException;
import java.net.SocketTimeoutException;

/**
 * Abstract superclass for client and server local sockets.
 */
abstract class LocalSocket implements Closeable {

  protected enum State {
    NEW,
    BOUND, // server only
    LISTENING, // server only
    CONNECTED, // client only
    CLOSED,
  }

  protected LocalSocketAddress address = null;
  protected FileDescriptor fd = new FileDescriptor();
  protected State state;
  protected boolean inputShutdown = false;
  protected boolean outputShutdown = false;

  /**
   * Constructs an unconnected local socket.
   */
  protected LocalSocket() throws IOException {
    socket(fd);
    if (!fd.valid()) {
      throw new IOException("Couldn't create socket!");
    }
    this.state = State.NEW;
  }

  /**
   * Returns the address of the endpoint this socket is bound to.
   *
   * @return a <code>SocketAddress</code> representing the local endpoint of
   *   this socket.
   */
  public LocalSocketAddress getLocalSocketAddress() {
    return address;
  }

  /**
   * Closes this socket. This operation is idempotent.
   *
   * To be consistent with Java Socket, the shutdown states of the socket are
   * not changed. This makes it easier to port applications between Socket and
   * LocalSocket.
   *
   * @throws IOException if an I/O error occurred when closing the socket.
   */
  @Override
  public synchronized void close() throws IOException {
    if (state == State.CLOSED) {
      return;
    }
    // Closes the file descriptor if it has not been closed by the
    // input/output streams.
    if (!fd.valid()) {
      throw new IllegalStateException("LocalSocket.close(-1)");
    }
    close(fd);
    if (fd.valid()) {
      throw new IllegalStateException("LocalSocket.close() did not set fd to -1");
    }
    this.state = State.CLOSED;
  }

  /**
   * Returns the closed state of the ServerSocket.
   *
   * @return true if the socket has been closed
   */
  public synchronized boolean isClosed() {
    // If the file descriptor has been closed by the input/output
    // streams, marks the socket as closed too.
    return state == State.CLOSED;
  }

  /**
   * Returns the connected state of the ClientSocket.
   *
   * @return true if the socket is currently connected.
   */
  public synchronized boolean isConnected() {
    return state == State.CONNECTED;
  }

  protected synchronized void checkConnected() throws SocketException {
    if (!isConnected()) {
      throw new SocketException("Transport endpoint is not connected");
    }
  }

  protected synchronized void checkNotClosed() throws SocketException {
    if (isClosed()) {
      throw new SocketException("socket is closed");
    }
  }

  /**
   * Returns the shutdown state of the input channel.
   *
   * @return true is the input channel of the socket is shutdown.
   */
  public synchronized boolean isInputShutdown() {
    return inputShutdown;
  }

  /**
   * Returns the shutdown state of the output channel.
   *
   * @return true is the input channel of the socket is shutdown.
   */
  public synchronized boolean isOutputShutdown() {
    return outputShutdown;
  }

  protected synchronized void checkInputNotShutdown() throws SocketException {
    if (isInputShutdown()) {
      throw new SocketException("Socket input is shutdown");
    }
  }

  protected synchronized void checkOutputNotShutdown() throws SocketException {
    if (isOutputShutdown()) {
      throw new SocketException("Socket output is shutdown");
    }
  }

  static final int SHUT_RD = 0;         // Mapped to BSD SHUT_RD in JNI.
  static final int SHUT_WR = 1;         // Mapped to BSD SHUT_WR in JNI.

  public synchronized void shutdownInput() throws IOException {
    checkNotClosed();
    checkConnected();
    checkInputNotShutdown();
    inputShutdown = true;
    shutdown(fd, SHUT_RD);
  }

  public synchronized void shutdownOutput() throws IOException {
    checkNotClosed();
    checkConnected();
    checkOutputNotShutdown();
    outputShutdown = true;
    shutdown(fd, SHUT_WR);
  }

  ////////////////////////////////////////////////////////////////////////
  // JNI:

  static {
    UnixJniLoader.loadJni();
  }

  // The native calls below are thin wrappers around linux system calls. The
  // semantics remains the same except for poll(). See the comments for the
  // method.
  //
  // Note: FileDescriptor is a box for a mutable integer that is visible only
  // to native code.

  // Generic operations:
  protected static native void socket(FileDescriptor server)
      throws IOException;
  static native void close(FileDescriptor server)
      throws IOException;
  /**
   * Shut down part of a full-duplex connection
   * @param code Must be either SHUT_RD or SHUT_WR
   */
  static native void shutdown(FileDescriptor fd, int code)
      throws IOException;

  /**
   * This method checks waits for the given file descriptor to become available for read.
   * If timeoutMillis passed and there is no activity, a SocketTimeoutException will be thrown.
   */
  protected static native void poll(FileDescriptor read, long timeoutMillis)
      throws IOException, SocketTimeoutException, InterruptedIOException;

  // Server operations:
  protected static native void bind(FileDescriptor server, String filename)
      throws IOException;
  protected static native void listen(FileDescriptor server, int backlog)
      throws IOException;
  protected static native void accept(FileDescriptor server,
                                      FileDescriptor client)
      throws IOException;

  // Client operations:
  protected static native void connect(FileDescriptor client, String filename)
      throws IOException;
}
