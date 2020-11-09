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
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
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

  static WorkerKey createWorkerKey(
      FileSystem fileSystem, String mnemonic, boolean proxied, String... args) {
    return new WorkerKey(
        /* args= */ ImmutableList.copyOf(args),
        /* env= */ ImmutableMap.of("env1", "foo", "env2", "bar"),
        /* execRoot= */ fileSystem.getPath("/fake"),
        /* mnemonic= */ mnemonic,
        /* workerFilesCombinedHash= */ HashCode.fromInt(0),
        /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
        /* mustBeSandboxed= */ false,
        /* proxied= */ proxied,
        WorkerProtocolFormat.PROTO);
  }

  static WorkerKey createWorkerKey(WorkerProtocolFormat protocolFormat, FileSystem fs) {
    return new WorkerKey(
        /* args= */ ImmutableList.of("arg1", "arg2", "arg3"),
        /* env= */ ImmutableMap.of("env1", "foo", "env2", "bar"),
        /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
        /* mnemonic= */ "dummy",
        /* workerFilesCombinedHash= */ HashCode.fromInt(0),
        /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
        /* mustBeSandboxed= */ true,
        /* proxied= */ true,
        protocolFormat);
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
      return wasDestroyed;
    }
  }
}
