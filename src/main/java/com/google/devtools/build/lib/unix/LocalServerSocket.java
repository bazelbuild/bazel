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
package com.google.devtools.build.lib.unix;

import java.io.FileDescriptor;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.Socket;
import java.net.SocketException;
import java.net.SocketTimeoutException;

/**
 * <p>An implementation of ServerSocket for local (AF_UNIX) sockets.
 *
 * <p>This class intentionally doesn't extend java.net.ServerSocket although it
 * has some similarity to it.  The java.net class hierarchy is a terrible mess
 * and is inextricably coupled to the Internet Protocol.
 *
 * <p>This code is not intended to be portable to non-UNIX platforms.
 */
public class LocalServerSocket extends LocalSocket {

  // Socket timeout in milliseconds. No timeout by default.
  private long soTimeoutMillis = 0;

  /**
   * Constructs an unbound local server socket.
   */
  public LocalServerSocket() throws IOException {
    super();
  }

  /**
   * Constructs a server socket, binds it to the specified address, and
   * listens for incoming connections with the specified backlog.
   *
   * @throws IOException if any of the socket/bind/listen operations failed.
   */
  public LocalServerSocket(LocalSocketAddress address, int backlog)
      throws IOException {
    this();
    bind(address);
    listen(backlog);
  }

  /**
   * Constructs a server socket, binds it to the specified address, and begin
   * listening for incoming connections using the default backlog.
   *
   * @throws IOException if any of the socket/bind/listen operations failed.
   */
  public LocalServerSocket(LocalSocketAddress address) throws IOException {
    this(address, 50);
  }

  /**
   * Specifies the timeout in milliseconds for accept(). Setting it to
   * zero means an indefinite timeout.
   */
  public void setSoTimeout(long timeoutMillis) {
    soTimeoutMillis = timeoutMillis;
  }

  /**
   * Returns the current timeout in milliseconds.
   */
  public long getSoTimeout() {
    return soTimeoutMillis;
  }

  /**
   * Binds the specified address to this socket.  The socket must be unbound.
   * This causes the filesystem entry to appear.
   *
   * @throws IOException if the bind failed.
   */
  public synchronized void bind(LocalSocketAddress address)
      throws IOException {
    if (address == null) {
      throw new NullPointerException("address");
    }
    checkNotClosed();
    if (state != State.NEW) {
      throw new SocketException("socket is already bound to an address");
    }
    bind(fd, address.getName().toString()); // JNI
    this.address = address;
    this.state = State.BOUND;
  }

  /**
   * Listen for incoming connections on a socket using the specfied backlog.
   * The socket must be bound but not already listening.
   *
   * @throws IOException if the listen failed.
   */
  public synchronized void listen(int backlog) throws IOException {
    if (backlog < 1) {
      throw new IllegalArgumentException("backlog=" + backlog);
    }
    checkNotClosed();
    if (address == null) {
      throw new SocketException("socket has no address bound");
    }
    if (state == State.LISTENING) {
      throw new SocketException("socket is already listening");
    }
    listen(fd, backlog); // JNI
    this.state = State.LISTENING;
  }

  /**
   * Blocks until a connection is made to this socket and accepts it, returning
   * a new socket connected to the client.
   *
   * @return the new socket connected to the client.
   * @throws IOException if an error occurs when waiting for a connection.
   * @throws SocketTimeoutException if a timeout was previously set with
   *         setSoTimeout and the timeout has been reached.
   * @throws InterruptedIOException if the thread is interrupted when the
   *         method is blocked.
   */
  public synchronized Socket accept()
      throws IOException, SocketTimeoutException, InterruptedIOException {
    if (state != State.LISTENING) {
      throw new SocketException("socket is not in listening state");
    }

    // Throws a SocketTimeoutException if timeout.
    if (soTimeoutMillis != 0) {
      poll(fd, soTimeoutMillis); // JNI
    }

    FileDescriptor clientFd = new FileDescriptor();
    accept(fd, clientFd); // JNI
    final LocalSocketImpl impl = new LocalSocketImpl(clientFd);
    return new Socket(impl) {
        @Override
        public boolean isConnected() {
          return true;
        }
        @Override
        public synchronized void close() throws IOException {
          if (isClosed()) {
            return;
          } else {
            super.close();
            // Workaround for the fact that super.created==false because we
            // created the impl ourselves.  As a result, super.close() doesn't
            // call impl.close().   *Sigh*, java.net is horrendous.
            // (Perhaps we should dispense with Socket/SocketImpl altogether?)
            impl.close();
          }
        }
      };
  }

  @Override
  public String toString() {
    return "LocalServerSocket(" + address + ")";
  }
}
