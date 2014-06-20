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

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.SocketException;

/**
 * <p>An implementation of client Socket for local (AF_UNIX) sockets.
 *
 * <p>This class intentionally doesn't extend java.net.Socket although it
 * has some similarity to it.  The java.net class hierarchy is a terrible mess
 * and is inextricably coupled to the Internet Protocol.
 *
 * <p>This code is not intended to be portable to non-UNIX platforms.
 */
public class LocalClientSocket extends LocalSocket {

  /**
   * Constructs an unconnected local client socket.
   *
   * @throws IOException if the socket could not be created.
   */
  public LocalClientSocket() throws IOException {
    super();
  }

  /**
   * Constructs a client socket and connects it to the specified address.
   *
   * @throws IOException if either of the socket/connect operations failed.
   */
  public LocalClientSocket(LocalSocketAddress address) throws IOException {
    super();
    connect(address);
  }

  /**
   * Connect to the specified server.  Blocks until the server accepts the
   * connection.
   *
   * @throws IOException if the connection failed.
   */
  public synchronized void connect(LocalSocketAddress address)
      throws IOException {
    checkNotClosed();
    if (state == State.CONNECTED) {
      throw new SocketException("socket is already connected");
    }
    connect(fd, address.getName().toString()); // JNI
    this.address = address;
    this.state = State.CONNECTED;
  }

  /**
   * Returns the input stream for reading from the server.
   *
   * @param closeSocket close the socket when this input stream is closed.
   * @throws IOException if there was a problem.
   */
  public synchronized InputStream getInputStream(final boolean closeSocket) throws IOException {
    checkConnected();
    checkInputNotShutdown();
    return new FileInputStream(fd) {
      @Override
      public void close() throws IOException {
        if (closeSocket) {
          LocalClientSocket.this.close();
        }
      }
    };
  }

  /**
   * Returns the input stream for reading from the server.
   *
   * @throws IOException if there was a problem.
   */
  public synchronized InputStream getInputStream() throws IOException {
    return getInputStream(false);
  }

  /**
   * Returns the output stream for writing to the server.
   *
   * @throws IOException if there was a problem.
   */
  public synchronized OutputStream getOutputStream() throws IOException {
    checkConnected();
    checkOutputNotShutdown();
    return new FileOutputStream(fd) {
        @Override public void close() {
          // Don't close the file descriptor.
        }
      };
  }

  @Override
  public String toString() {
    return "LocalClientSocket(" + address + ")";
  }
}
