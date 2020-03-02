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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.StringWriter;

/**
 * A sandboxfs implementation that uses an external sandboxfs binary to manage the mount point.
 *
 * <p>This implementation provides support for the reconfiguration protocol available in the 0.1.x
 * series.
 */
final class RealSandboxfs01Process extends RealSandboxfsProcess {

  /**
   * Writer with which to send data to the sandboxfs instance. Null only after {@link #destroy()}
   * has been invoked.
   */
  private final BufferedWriter processStdIn;

  /**
   * Reader with which to receive data from the sandboxfs instance. Null only after {@link
   * #destroy()} has been invoked.
   */
  private final BufferedReader processStdOut;

  /**
   * Initializes a new sandboxfs process instance.
   *
   * @param process process handle for the already-running sandboxfs instance
   */
  RealSandboxfs01Process(Path mountPoint, Subprocess process) {
    super(mountPoint, process);

    this.processStdIn =
        new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), UTF_8));
    this.processStdOut = new BufferedReader(new InputStreamReader(process.getInputStream(), UTF_8));
  }

  /**
   * Pushes a new configuration to sandboxfs and waits for acceptance.
   *
   * @param config the configuration chunk to push to sandboxfs
   * @throws IOException if sandboxfs cannot be reconfigured either because of an error in the
   *     configuration or because we failed to communicate with the subprocess
   */
  private synchronized void reconfigure(String config) throws IOException {
    processStdIn.write(config);
    processStdIn.write("\n\n");
    processStdIn.flush();

    String done = processStdOut.readLine();
    if (done == null) {
      throw new IOException("premature end of output from sandboxfs");
    }
    if (!done.equals("Done")) {
      throw new IOException("received unknown string from sandboxfs: " + done + "; expected Done");
    }
  }

  @Override
  @SuppressWarnings("UnnecessaryParentheses")
  public void createSandbox(String name, SandboxCreator creator) throws IOException {
    checkArgument(!PathFragment.containsSeparator(name));
    PathFragment root = PathFragment.create("/").getRelative(name);

    StringWriter stringWriter = new StringWriter();
    try (JsonWriter writer = new JsonWriter(stringWriter)) {
      writer.beginArray();
      creator.create(
          ((path, underlyingPath, writable) -> {
            writer.beginObject();
            {
              writer.name("Map");
              writer.beginObject();
              {
                writer.name("Mapping");
                writer.value((root.getRelative(path.toRelative())).getPathString());
                writer.name("Target");
                writer.value(underlyingPath.getPathString());
                writer.name("Writable");
                writer.value(writable);
              }
              writer.endObject();
            }
            writer.endObject();
          }));
      writer.endArray();
    }
    reconfigure(stringWriter.toString());
  }

  @Override
  @SuppressWarnings("UnnecessaryParentheses")
  public void destroySandbox(String name) throws IOException {
    checkArgument(!PathFragment.containsSeparator(name));
    PathFragment root = PathFragment.create("/").getRelative(name);

    StringWriter stringWriter = new StringWriter();
    try (JsonWriter writer = new JsonWriter(stringWriter)) {
      writer.beginArray();
      {
        writer.beginObject();
        {
          writer.name("Unmap");
          writer.value(root.getPathString());
        }
        writer.endObject();
      }
      writer.endArray();
    }
    reconfigure(stringWriter.toString());
  }
}
