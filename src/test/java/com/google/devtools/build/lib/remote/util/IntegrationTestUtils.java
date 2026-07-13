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

import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

/** Integration test utilities. */
public final class IntegrationTestUtils {
  private IntegrationTestUtils() {}

  private static final String WORKER_RLOCATIONPATH =
      "io_bazel/src/tools/remote/worker" + (OS.getCurrent() == OS.WINDOWS ? ".exe" : "");

  /**
   * Manages a remote worker instance as a {@link TestRule}.
   *
   * <p>Should be kept in a static variable annotated with both {@link org.junit.ClassRule} and
   * {@link org.junit.Rule}.
   */
  public static WorkerInstance createWorker(String... extraArgs) {
    return createWorker(/* useHttp= */ false, extraArgs);
  }

  /**
   * Manages a remote worker instance as a {@link TestRule}.
   *
   * <p>Should be kept in a static variable annotated with both {@link org.junit.ClassRule} and
   * {@link org.junit.Rule}.
   */
  public static WorkerInstance createWorker(boolean useHttp, String... extraArgs) {
    // The worker directory must not be a subdirectory of the test temporary directory for two
    // reasons:
    // 1. It should be preserved between individual tests so that the worker can be kept running.
    // 2. Even if that wasn't needed, JUnit runs "after" methods of rules after those of
    //    superclasses, which means that BuildIntegrationtestCase's cleanup method would attempt
    //    to delete the worker directory before the worker is stopped, which fails on Windows.
    Path workerTmpDir;
    try {
      workerTmpDir = Files.createTempDirectory(systemTmpDir(), "remote.");
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return new WorkerInstance(useHttp, workerTmpDir, ImmutableList.copyOf(extraArgs));
  }

  private static Path systemTmpDir() {
    if (OS.getCurrent() == OS.WINDOWS) {
      return Path.of(System.getenv("TEMP"));
    }
    String tmpdir = System.getenv("TMPDIR");
    if (tmpdir == null) {
      tmpdir = "/tmp";
    }
    return Path.of(tmpdir);
  }

  private static void ensureMkdir(Path path) throws IOException {
    if (!Files.notExists(path)) {
      throw new IOException(path + " already exists");
    }
    Files.createDirectories(path);
  }

  public static class WorkerInstance implements TestRule {
    private final boolean useHttp;
    private final Path stdPath;
    private final Path stdoutPath;
    private final Path stderrPath;
    private final Path workPath;
    private final Path casPath;
    private final ImmutableList<String> extraArgs;

    @Nullable private Integer port;
    @Nullable private Subprocess process;

    private WorkerInstance(boolean useHttp, Path dir, ImmutableList<String> extraArgs) {
      this.useHttp = useHttp;
      this.stdPath = dir.resolve("std");
      this.stdoutPath = stdPath.resolve("stdout");
      this.stderrPath = stdPath.resolve("stderr");
      this.workPath = dir.resolve("work_path");
      this.casPath = dir.resolve("cas_path");
      this.extraArgs = extraArgs;
    }

    @Override
    public Statement apply(Statement base, Description description) {
      if (description.isSuite()) {
        return new Statement() {
          @Override
          public void evaluate() throws Throwable {
            try (var ignored = start()) {
              base.evaluate();
            }
          }
        };
      } else if (description.isTest()) {
        return new Statement() {
          @Override
          public void evaluate() throws Throwable {
            try {
              base.evaluate();
            } finally {
              reset();
            }
          }
        };
      } else {
        return base;
      }
    }

    public AutoCloseable start() throws IOException, InterruptedException {
      Preconditions.checkState(process == null);
      Preconditions.checkState(port == null);

      ensureMkdir(workPath);
      ensureMkdir(casPath);
      ensureMkdir(stdPath);
      Files.createFile(stdoutPath);
      Files.createFile(stderrPath);
      Runfiles runfiles = Runfiles.preload().withSourceRepository("");
      String workerPath = runfiles.rlocation(WORKER_RLOCATIONPATH);
      ImmutableMap.Builder<String, String> env = ImmutableMap.builder();
      env.putAll(System.getenv());
      env.putAll(runfiles.getEnvVars());
      port = FreePortFinder.pickUnusedRandomPort();
      process =
          new SubprocessBuilder(System.getenv())
              .setEnv(env.buildKeepingLast())
              .setStdout(stdoutPath.toFile())
              .setStderr(stderrPath.toFile())
              .setArgv(
                  ImmutableList.<String>builder()
                      .add(
                          workerPath,
                          "--work_path=" + workPath,
                          "--cas_path=" + casPath,
                          (useHttp ? "--http_listen_port=" : "--listen_port=") + port)
                      .addAll(extraArgs)
                      .build())
              .start();
      waitForPortOpen(process, port);

      return this::stop;
    }

    private void waitForPortOpen(Subprocess process, int port)
        throws IOException, InterruptedException {
      var addr = new InetSocketAddress("localhost", port);
      var timeout = new IOException("Timed out while trying to connect to worker");
      for (var i = 0; i < 20; ++i) {
        if (!process.isAlive()) {
          throw new IOException(
              String.format(
                  "Worker died while trying to connect\n"
                      + "----- STDOUT -----\n%s\n"
                      + "----- STDERR -----\n%s\n",
                  getStdout(), getStderr()));
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

    private void stop() throws IOException {
      Preconditions.checkNotNull(process);
      process.destroyAndWait();
      process = null;

      deleteTree(stdPath);
      deleteTree(workPath);
      deleteTree(casPath);
    }

    public void reset() throws IOException, InterruptedException {
      // The DiskCacheClient in the worker expects the CAS subdirectories to exist.
      List<Path> toClear;
      try (var stream = Files.list(casPath)) {
        toClear = stream.toList();
      }
      for (var path : toClear) {
        deleteTree(path);
        ensureMkdir(path);
      }
    }

    public String getStdout() {
      try {
        var out = Files.readAllBytes(stdoutPath);
        return new String(out, UTF_8);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    public String getStderr() {
      try {
        var out = Files.readAllBytes(stderrPath);
        return new String(out, UTF_8);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    private static void deleteTree(Path path) throws IOException {
      List<Path> toDelete;
      try (var stream = Files.walk(path)) {
        toDelete = stream.sorted(Comparator.reverseOrder()).toList();
      }
      for (var p : toDelete) {
        Files.delete(p);
      }
    }

    public int getPort() {
      return port;
    }

    public PathFragment getCasPath() {
      return PathFragment.create(casPath.toString());
    }

    /** Returns the path of the blob with the given contents in the worker's CAS. */
    public PathFragment getCasBlobPath(byte[] contents) {
      return getCasBlobPath(
          Digest.newBuilder()
              .setHash(Hashing.sha256().hashBytes(contents).toString())
              .setSizeBytes(contents.length)
              .build());
    }

    /** Returns the path of the blob with the given digest in the worker's CAS. */
    public PathFragment getCasBlobPath(Digest digest) {
      return PathFragment.create(casBlobPath(digest).toString());
    }

    /**
     * Deletes the blob with the given contents from the worker's CAS, leaving all other state (in
     * particular action cache entries referencing the blob) intact.
     */
    public void evictBlob(byte[] contents) throws IOException {
      Files.delete(Path.of(getCasBlobPath(contents).getPathString()));
    }

    // Mirrors the on-disk layout of DiskCacheClient.
    private Path casBlobPath(Digest digest) {
      String hash = digest.getHash();
      return casPath.resolve("cas").resolve(hash.substring(0, 2)).resolve(hash);
    }
  }
}
