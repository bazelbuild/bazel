// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.List;

/**
 * A sandboxfs implementation that uses an external sandboxfs binary to manage the mount point.
 *
 * <p>This implementation provides support for the reconfiguration protocol introduced in 0.2.0.
 */
final class RealSandboxfs02Process extends RealSandboxfsProcess {

  /**
   * Writer with which to send data to the sandboxfs instance. Null only after {@link #destroy()}
   * has been invoked.
   */
  private final JsonWriter processStdIn;

  /**
   * Reader with which to receive data from the sandboxfs instance. Null only after {@link
   * #destroy()} has been invoked.
   */
  private final JsonReader processStdOut;

  /**
   * Initializes a new sandboxfs process instance.
   *
   * @param process process handle for the already-running sandboxfs instance
   */
  RealSandboxfs02Process(Path mountPoint, Subprocess process) {
    super(mountPoint, process);

    this.processStdIn =
        new JsonWriter(
            new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), UTF_8)));
    this.processStdOut =
        new JsonReader(new BufferedReader(new InputStreamReader(process.getInputStream(), UTF_8)));

    // Must use lenient writing and parsing to accept a stream of separate top-level JSON objects.
    this.processStdIn.setLenient(true);
    this.processStdOut.setLenient(true);
  }

  /**
   * Waits for sandboxfs to accept the reconfiguration request.
   *
   * @param wantId name of the sandbox for which we are waiting a response
   * @throws IOException if sandboxfs fails to accept the request for any reason
   */
  private synchronized void waitAck(String wantId) throws IOException {
    processStdOut.beginObject();
    String id = null;
    String error = null;
    while (processStdOut.hasNext()) {
      String name = processStdOut.nextName();
      switch (name) {
        case "error":
          if (processStdOut.peek() == JsonToken.NULL) {
            processStdOut.nextNull();
          } else {
            checkState(error == null);
            error = processStdOut.nextString();
          }
          break;

        case "id":
          if (processStdOut.peek() == JsonToken.NULL) {
            processStdOut.nextNull();
          } else {
            checkState(id == null);
            id = processStdOut.nextString();
          }
          break;

        default:
          throw new IOException("Invalid field name in response: " + name);
      }
    }
    processStdOut.endObject();
    if (id == null || !id.equals(wantId)) {
      throw new IOException("Got response with ID " + id + " but expected " + wantId);
    }
    if (error != null) {
      throw new IOException(error);
    }
  }

  /** Encodes a mapping into JSON. */
  @SuppressWarnings("UnnecessaryParentheses")
  private static void writeMapping(JsonWriter writer, Mapping mapping) throws IOException {
    writer.beginObject();
    {
      writer.name("path");
      writer.value(mapping.path().getPathString());
      writer.name("underlying_path");
      writer.value(mapping.target().getPathString());
      writer.name("writable");
      writer.value(mapping.writable());
    }
    writer.endObject();
  }

  @Override
  @SuppressWarnings("UnnecessaryParentheses")
  public synchronized void createSandbox(String id, List<Mapping> mappings) throws IOException {
    checkArgument(!PathFragment.containsSeparator(id));

    processStdIn.beginObject();
    {
      processStdIn.name("CreateSandbox");
      processStdIn.beginObject();
      {
        processStdIn.name("id");
        processStdIn.value(id);
        processStdIn.name("mappings");
        processStdIn.beginArray();
        for (Mapping mapping : mappings) {
          writeMapping(processStdIn, mapping);
        }
        processStdIn.endArray();
      }
      processStdIn.endObject();
    }
    processStdIn.endObject();

    processStdIn.flush();
    waitAck(id);
  }

  @Override
  @SuppressWarnings("UnnecessaryParentheses")
  public synchronized void destroySandbox(String id) throws IOException {
    checkArgument(!PathFragment.containsSeparator(id));

    processStdIn.beginObject();
    {
      processStdIn.name("DestroySandbox");
      processStdIn.value(id);
    }
    processStdIn.endObject();

    processStdIn.flush();
    waitAck(id);
  }
}
