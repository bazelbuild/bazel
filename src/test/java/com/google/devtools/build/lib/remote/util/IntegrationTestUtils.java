// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.testutil.TestUtils.tmpDirFile;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.File;
import java.io.IOException;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.SocketException;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/** Integration test utilities. */
public final class IntegrationTestUtils {
  private IntegrationTestUtils() {}

  private static final PathFragment WORKER_PATH =
      PathFragment.create(
          "io_bazel/src/tools/remote/worker"
              + (OS.getCurrent() == OS.WINDOWS ? ".exe" : ""));

  private static final AtomicInteger WORKER_COUNTER = new AtomicInteger(0);

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

  private static void waitForPortOpen(Subprocess process, int port)
      throws IOException, InterruptedException {
    var addr = new InetSocketAddress("localhost", port);
    var timeout = new IOException("Timed out when waiting for port to open");
    for (var i = 0; i < 20; ++i) {
      if (!process.isAlive()) {
        var message = new String(process.getErrorStream().readAllBytes(), UTF_8);
        throw new IOException("Failed to start worker: " + message);
      }

      try {
        try (var socketChannel = SocketChannel.open()) {
          socketChannel.configureBlocking(/* block= */ true);
          socketChannel.connect(addr);
        }
        return;
      } catch (IOException e) {
        timeout.addSuppressed(e);
        Thread.sleep(1000);
      }
    }
    throw timeout;
  }

  public static WorkerInstance startWorker() throws IOException, InterruptedException {
    return startWorker(/* useHttp= */ false);
  }

  public static WorkerInstance startWorker(boolean useHttp)
      throws IOException, InterruptedException {
    PathFragment testTmpDir = PathFragment.create(tmpDirFile().getAbsolutePath());
    PathFragment stdPath = testTmpDir.getRelative("remote.std");
    PathFragment workPath = testTmpDir.getRelative("remote.work_path");
    PathFragment casPath = testTmpDir.getRelative("remote.cas_path");
    int workerPort = pickUnusedRandomPort();
    var worker =
        new WorkerInstance(WORKER_COUNTER, useHttp, workerPort, stdPath, workPath, casPath);
    worker.start();
    return worker;
  }

  private static void ensureTouchFile(PathFragment path) throws IOException {
    File file = new File(path.getSafePathString());
    if (file.exists()) {
      throw new IOException(path + " already exists");
    }
    if (!file.createNewFile()) {
      throw new IOException("Failed to create file " + path);
    }
  }

  private static void ensureMkdir(PathFragment path) throws IOException {
    File dir = new File(path.getSafePathString());
    if (dir.exists()) {
      throw new IOException(path + " already exists");
    }
    if (!dir.mkdirs()) {
      throw new IOException("Failed to create directory " + path);
    }
  }

  public static class WorkerInstance {
    private final AtomicInteger counter;
    private final boolean useHttp;
    private final int port;
    private final PathFragment stdPathPrefix;
    private final PathFragment workPathPrefix;
    private final PathFragment casPathPrefix;

    @Nullable private Subprocess process;
    @Nullable PathFragment stdoutPath;
    @Nullable PathFragment stderrPath;
    @Nullable PathFragment workPath;
    @Nullable PathFragment casPath;

    private WorkerInstance(
        AtomicInteger counter,
        boolean useHttp,
        int port,
        PathFragment stdPathPrefix,
        PathFragment workPathPrefix,
        PathFragment casPathPrefix) {
      this.counter = counter;
      this.useHttp = useHttp;
      this.port = port;
      this.stdPathPrefix = stdPathPrefix;
      this.workPathPrefix = workPathPrefix;
      this.casPathPrefix = casPathPrefix;
    }

    private void start() throws IOException, InterruptedException {
      Preconditions.checkState(process == null);
      Preconditions.checkState(stdoutPath == null);
      Preconditions.checkState(stderrPath == null);
      Preconditions.checkState(workPath == null);
      Preconditions.checkState(casPath == null);

      var suffix = String.valueOf(counter.getAndIncrement());
      var stdPath = stdPathPrefix.getRelative(suffix);
      stdoutPath = stdPath.getRelative("stdoud");
      stderrPath = stdPath.getRelative("stderr");
      workPath = workPathPrefix.getRelative(suffix);
      casPath = casPathPrefix.getRelative(suffix);

      ensureMkdir(workPath);
      ensureMkdir(casPath);
      ensureMkdir(stdPath);
      ensureTouchFile(stdoutPath);
      ensureTouchFile(stderrPath);
      String workerPath = Runfiles.create().rlocation(WORKER_PATH.getSafePathString());
      process =
          new SubprocessBuilder()
              .setStdout(new File(stdoutPath.getSafePathString()))
              .setStderr(new File(stderrPath.getSafePathString()))
              .setArgv(
                  ImmutableList.of(
                      workerPath,
                      "--work_path=" + workPath.getSafePathString(),
                      "--cas_path=" + casPath.getSafePathString(),
                      (useHttp ? "--http_listen_port=" : "--listen_port=") + port))
              .start();
      waitForPortOpen(process, port);
    }

    public void stop() throws IOException {
      Preconditions.checkNotNull(process);
      process.destroyAndWait();
      process = null;

      deleteDir(stdoutPath);
      stdoutPath = null;
      deleteDir(stderrPath);
      stderrPath = null;

      deleteDir(workPath);
      workPath = null;

      deleteDir(casPath);
      casPath = null;
    }

    public void restart() throws IOException, InterruptedException {
      stop();
      start();
    }

    public String getStdout() {
      try {
        var out = Files.readAllBytes(Paths.get(stdoutPath.getSafePathString()));
        return new String(out, UTF_8);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    public String getStderr() {
      try {
        var out = Files.readAllBytes(Paths.get(stderrPath.getSafePathString()));
        return new String(out, UTF_8);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    private static void deleteDir(PathFragment path) throws IOException {
      try (var stream = Files.walk(Paths.get(path.getSafePathString()))) {
        stream.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
      }
    }

    public int getPort() {
      return port;
    }

    public PathFragment getCasPath() {
      return casPath;
    }
  }
}
