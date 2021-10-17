// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.worker;


import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Utilities that come in handy when unit-testing the worker code. */
class TestUtils {

  private TestUtils() {}

  /** A helper method to create a fake Spawn with the given execution info. */
  static Spawn createSpawn(ImmutableMap<String, String> executionInfo) {
    return new SimpleSpawn(
        new ActionsTestUtil.NullAction(),
        /* arguments= */ ImmutableList.of(),
        /* environment= */ ImmutableMap.of(),
        executionInfo,
        /* inputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /* outputs= */ ImmutableSet.of(),
        ResourceSet.ZERO);
  }

  /** A helper method to create a WorkerKey through WorkerParser. */
  static WorkerKey createWorkerKey(
      WorkerProtocolFormat protocolFormat,
      FileSystem fs,
      String mnemonic,
      boolean multiplex,
      boolean sandboxed,
      boolean dynamic,
      String... args) {
    WorkerOptions workerOptions = new WorkerOptions();
    workerOptions.workerMultiplex = multiplex;
    workerOptions.workerSandboxing = sandboxed;

    Spawn spawn =
        createSpawn(
            ImmutableMap.of(
                ExecutionRequirements.WORKER_KEY_MNEMONIC, mnemonic,
                ExecutionRequirements.REQUIRES_WORKER_PROTOCOL, "proto",
                ExecutionRequirements.SUPPORTS_WORKERS, "1",
                ExecutionRequirements.SUPPORTS_MULTIPLEX_WORKERS, "1"));

    return WorkerParser.createWorkerKey(
        spawn,
        /* workerArgs= */ ImmutableList.copyOf(args),
        /* env= */ ImmutableMap.of("env1", "foo", "env2", "bar"),
        /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
        /* workerFilesCombinedHash= */ HashCode.fromInt(0),
        /* workerFiles= */ ImmutableSortedMap.of(),
        workerOptions,
        dynamic,
        protocolFormat);
  }

  static WorkerKey createWorkerKey(
      FileSystem fileSystem, String mnemonic, boolean proxied, String... args) {
    return createWorkerKey(
        WorkerProtocolFormat.PROTO,
        fileSystem,
        mnemonic,
        proxied,
        /* sandboxed= */ false,
        /* dynamic= */ false,
        args);
  }

  static WorkerKey createWorkerKey(WorkerProtocolFormat protocolFormat, FileSystem fs) {
    return createWorkerKey(protocolFormat, fs, false);
  }

  static WorkerKey createWorkerKey(
      WorkerProtocolFormat protocolFormat, FileSystem fs, boolean dynamic) {
    return createWorkerKey(
        protocolFormat,
        fs,
        /* mnemonic= */ "dummy",
        /* proxied= */ true,
        /* sandboxed= */ true,
        dynamic,
        /* args...= */ "arg1",
        "arg2",
        "arg3");
  }

  static WorkerKey createWorkerKey(
      FileSystem fs, boolean multiplex, boolean sandboxed, boolean dynamic) {
    return createWorkerKey(
        WorkerProtocolFormat.PROTO,
        fs,
        /* mnemonic= */ "dummy",
        multiplex,
        sandboxed,
        dynamic,
        /* args...= */ "arg1",
        "arg2",
        "arg3");
  }

  /** A worker that uses a fake subprocess for I/O. */
  static class TestWorker extends SingleplexWorker {
    private final FakeSubprocess fakeSubprocess;

    TestWorker(
        WorkerKey workerKey,
        int workerId,
        final Path workDir,
        Path logFile,
        FakeSubprocess fakeSubprocess) {
      super(workerKey, workerId, workDir, logFile);
      this.fakeSubprocess = fakeSubprocess;
    }

    @Override
    Subprocess createProcess() {
      return fakeSubprocess;
    }

    FakeSubprocess getFakeSubprocess() {
      return fakeSubprocess;
    }
  }

  /**
   * The {@link Worker} object uses a {@link Subprocess} to interact with persistent worker
   * binaries. Since this test is strictly testing {@link Worker} and not any outside persistent
   * worker binaries, a {@link FakeSubprocess} instance is used to fake the {@link InputStream} and
   * {@link OutputStream} that normally write and read from a persistent worker.
   */
  static class FakeSubprocess implements Subprocess {
    private final InputStream inputStream;
    private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    private final ByteArrayInputStream errStream = new ByteArrayInputStream(new byte[0]);
    private boolean wasDestroyed = false;

    /** Creates a fake Subprocess that writes {@code bytes} to its "stdout". */
    FakeSubprocess(byte[] bytes) throws IOException {
      inputStream = new ByteArrayInputStream(bytes);
    }

    FakeSubprocess(InputStream responseStream) throws IOException {
      this.inputStream = responseStream;
    }

    @Override
    public InputStream getInputStream() {
      return inputStream;
    }

    @Override
    public OutputStream getOutputStream() {
      return outputStream;
    }

    @Override
    public InputStream getErrorStream() {
      return errStream;
    }

    @Override
    public boolean destroy() {
      for (Closeable stream : new Closeable[] {inputStream, outputStream, errStream}) {
        try {
          stream.close();
        } catch (IOException e) {
          throw new IllegalStateException(e);
        }
      }

      wasDestroyed = true;
      return true;
    }

    @Override
    public int exitValue() {
      return 0;
    }

    @Override
    public boolean finished() {
      return true;
    }

    @Override
    public boolean timedout() {
      return false;
    }

    @Override
    public void waitFor() throws InterruptedException {
      // Do nothing.
    }

    @Override
    public void close() {
      // Do nothing.
    }

    @Override
    public boolean isAlive() {
      return !wasDestroyed;
    }
  }
}
