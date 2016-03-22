// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.server;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.unix.LocalClientSocket;
import com.google.devtools.build.lib.unix.LocalSocketAddress;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.util.io.StreamDemultiplexer;
import com.google.devtools.build.lib.vfs.Path;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;

/**
 * A client to test RPCServer.
 */
public class RPCTestingClient {

  private final RecordingOutErr outErr;
  private final Path socketFile;

  /**
   * Create a client to RPCServer. {@code socketFile} must be a file
   * on disk; this will not work with the in-memory file system.
   */
  public RPCTestingClient(RecordingOutErr outErr, Path socketFile) {
    this.socketFile = socketFile;
    this.outErr = outErr;
  }

  public int sendRequest(String command, String... params)
      throws Exception {
    String request = command;
    for (String param : params) {
      request += "\0" + param;
    }
    return sendRequest(request);
  }

  public int sendRequest(String request) throws Exception {
    LocalClientSocket connection = new LocalClientSocket();
    connection.connect(new LocalSocketAddress(socketFile.getPathFile()));
    try {
      OutputStream out = connection.getOutputStream();
      byte[] requestBytes = request.getBytes(UTF_8);
      byte[] requestLength = new byte[4];
      requestLength[0] = (byte) (requestBytes.length << 24);
      requestLength[1] = (byte) ((requestBytes.length << 16) & 0xff);
      requestLength[2] = (byte) ((requestBytes.length << 8) & 0xff);
      requestLength[3] = (byte) (requestBytes.length & 0xff);
      out.write(requestLength);
      out.write(requestBytes);
      out.flush();
      connection.shutdownOutput();

      OutputStream stdout = outErr.getOutputStream();
      OutputStream stderr = outErr.getErrorStream();
      ByteArrayOutputStream control = new ByteArrayOutputStream();
      StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1,
          stdout, stderr, control);
      ByteStreams.copy(connection.getInputStream(), demux);
      demux.flush();

      byte[] controlBytes = control.toByteArray();
      return (((int) controlBytes[0]) << 24)
          + (((int) controlBytes[1]) << 16)
          + (((int) controlBytes[2]) << 8)
          + ((int) controlBytes[3]);
    } finally {
      connection.close();
    }
  }

}
