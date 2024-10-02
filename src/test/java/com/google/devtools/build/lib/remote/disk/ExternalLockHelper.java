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
package com.google.devtools.build.lib.remote.disk;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** A helper binary that holds a shared or exclusive lock on a file. */
public final class ExternalLockHelper {
  private ExternalLockHelper() {}

  public static void main(String[] args) throws IOException, InterruptedException {
    if (args.length != 2) {
      throw new IOException("bad arguments");
    }
    Path path = Path.of(args[0]).toAbsolutePath();
    Files.createDirectories(path.getParent());
    boolean shared = args[1].equals("shared");
    try (FileChannel channel =
            FileChannel.open(
                path,
                StandardOpenOption.READ,
                StandardOpenOption.WRITE,
                StandardOpenOption.CREATE);
        FileLock lock = channel.lock(0, Long.MAX_VALUE, shared)) {
      // Signal parent that the lock is held.
      System.out.println("!");
      // Block until killed by parent.
      while (true) {
        Thread.sleep(1000);
      }
    }
  }
}
