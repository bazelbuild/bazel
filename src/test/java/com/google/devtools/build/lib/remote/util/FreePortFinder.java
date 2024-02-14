// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import java.io.IOException;
import java.net.DatagramSocket;
import java.net.ServerSocket;
import java.net.SocketException;
import java.util.Random;

/**
 * Container for a cross-platform routine for finding a free port for a fake server to bind to
 * during testing.
 */
public final class FreePortFinder {

  /**
   * Finds an unused port and returns it, throwing {@link java.io.IOException} if no port can be
   * found.
   */
  public static int pickUnusedRandomPort() throws IOException, InterruptedException {
    Random rand = new Random();
    for (int i = 0; i < 128; ++i) {
      int port = rand.nextInt(64551) + 1024;
      if (isPortAvailable(port)) {
        return port;
      }
      if (Thread.interrupted()) {
        throw new InterruptedException("interrupted");
      }
    }

    throw new IOException("Failed to find available port");
  }

  private static boolean isPortAvailable(int port) {
    if (port < 1024 || port > 65535) {
      return false;
    }

    try (ServerSocket ss = new ServerSocket(port)) {
      ss.setReuseAddress(true);
    } catch (IOException e) {
      return false;
    }

    try (DatagramSocket ds = new DatagramSocket(port)) {
      ds.setReuseAddress(true);
    } catch (SocketException e) {
      return false;
    }

    return true;
  }

  private FreePortFinder() {}
}
