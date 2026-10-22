// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox.backend;

import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.common.options.Options;
import io.grpc.ManagedChannel;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Comparator;
import java.util.stream.Stream;

/**
 * The channel over which Bazel talks to a sandbox backend it has launched: a gRPC connection to a
 * unix domain socket in a directory that only the current user can access.
 *
 * <p>This is the only class that knows how Bazel and the backend find each other. The backend is
 * told the location of its socket on its command line and has to create it once it accepts
 * connections.
 *
 * <p>TODO: Windows isn't supported yet. Neither the unix domain sockets of the Netty transport nor
 * named pipes are known to work there with gRPC. A loopback TCP port with a per-session token is
 * the likely fallback, and would only need to change this class.
 */
final class SandboxBackendTransport implements AutoCloseable {
  private final Path directory;
  private final Path socket;

  private SandboxBackendTransport(Path directory) {
    this.directory = directory;
    this.socket = directory.resolve("backend.sock");
  }

  /** Creates the private directory that holds the socket. */
  static SandboxBackendTransport create() throws IOException {
    if (OS.getCurrent() == OS.WINDOWS) {
      throw new IOException("Sandbox backends are not supported on Windows yet");
    }
    // The path of a unix domain socket is limited to roughly a hundred bytes, which rules out the
    // default temporary directory on some platforms.
    Path parent = Path.of("/tmp");
    if (!Files.isDirectory(parent)) {
      parent = Path.of(System.getProperty("java.io.tmpdir"));
    }
    return new SandboxBackendTransport(
        Files.createTempDirectory(
            parent,
            "sbk-",
            PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------"))));
  }

  /** The path at which the backend has to create its socket. */
  String socketPath() {
    return socket.toString();
  }

  /** Whether the backend has created its socket. */
  boolean isSocketCreated() {
    return Files.exists(socket);
  }

  /** Creates a channel to the backend. The connection is established lazily. */
  ManagedChannel connect() throws IOException {
    return GoogleAuthUtils.newChannel(
        /* executor= */ null,
        "unix:" + socket,
        /* proxy= */ null,
        Options.getDefaults(AuthAndTLSOptions.class),
        /* interceptors= */ null,
        /* serviceConfig= */ null);
  }

  @Override
  public void close() {
    try (Stream<Path> files = Files.walk(directory)) {
      files.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
    } catch (IOException e) {
      // Best effort, the directory is in the temporary directory.
    }
  }
}
