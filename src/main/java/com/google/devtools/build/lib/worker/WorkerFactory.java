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
package com.google.devtools.build.lib.worker;

import com.google.common.eventbus.EventBus;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.SandboxedWorker.WorkerSandboxOptions;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.apache.commons.pool2.BaseKeyedPooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

/** Factory used by the pool to create / destroy / validate worker processes. */
public class WorkerFactory extends BaseKeyedPooledObjectFactory<WorkerKey, Worker> {

  // It's fine to use an AtomicInteger here (which is 32-bit), because it is only incremented when
  // spawning a new worker, thus even under worst-case circumstances and buggy workers quitting
  // after each action, this should never overflow.
  // This starts at 1 to avoid hiding latent problems of multiplex workers not returning a
  // request_id (which is indistinguishable from 0 in proto3).
  private static final AtomicInteger pidCounter = new AtomicInteger(1);

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Path workerBaseDir;
  private Reporter reporter;
  private EventBus eventBus;

  /**
   * Options specific to hardened sandbox. Null if {@code --experimental_worker_sandbox_hardening}
   * is not set.
   */
  @Nullable private final WorkerSandboxOptions hardenedSandboxOptions;

  public WorkerFactory(Path workerBaseDir) {
    this(workerBaseDir, null);
  }

  public WorkerFactory(Path workerBaseDir, @Nullable WorkerSandboxOptions hardenedSandboxOptions) {
    this.workerBaseDir = workerBaseDir;
    this.hardenedSandboxOptions = hardenedSandboxOptions;
  }

  public void setReporter(Reporter reporter) {
    this.reporter = reporter;
  }

  public void setEventBus(EventBus eventBus) {
    this.eventBus = eventBus;
  }

  @Override
  public Worker create(WorkerKey key) throws IOException {
    int workerId = pidCounter.getAndIncrement();
    String workTypeName = key.getWorkerTypeName();
    if (!workerBaseDir.isDirectory()) {
      workerBaseDir.createDirectoryAndParents();
    }
    Path logFile =
        workerBaseDir.getRelative(workTypeName + "-" + workerId + "-" + key.getMnemonic() + ".log");

    Worker worker;
    if (key.isSandboxed()) {
      if (key.isMultiplex()) {
        WorkerMultiplexer workerMultiplexer = WorkerMultiplexerManager.getInstance(key, logFile);
        Path workDir = getSandboxedWorkerPath(key);
        worker = new SandboxedWorkerProxy(key, workerId, logFile, workerMultiplexer, workDir);
      } else {
        Path workDir = getSandboxedWorkerPath(key, workerId);
        worker = new SandboxedWorker(key, workerId, workDir, logFile, hardenedSandboxOptions);
      }
    } else if (key.isMultiplex()) {
      WorkerMultiplexer workerMultiplexer = WorkerMultiplexerManager.getInstance(key, logFile);
      worker =
          new WorkerProxy(
              key, workerId, workerMultiplexer.getLogFile(), workerMultiplexer, key.getExecRoot());
    } else {
      worker = new SingleplexWorker(key, workerId, key.getExecRoot(), logFile);
    }

    String msg =
        String.format(
            "Created new %s %s %s (id %d, key hash %d), logging to %s",
            key.isSandboxed() ? "sandboxed" : "non-sandboxed",
            key.getMnemonic(),
            workTypeName,
            workerId,
            key.hashCode(),
            worker.getLogFile());
    WorkerLoggingHelper.logMessage(reporter, WorkerLoggingHelper.LogLevel.INFO, msg);
    if (eventBus != null) {
      eventBus.post(new WorkerCreatedEvent(key.hashCode(), key.getMnemonic()));
    }
    return worker;
  }

  Path getSandboxedWorkerPath(WorkerKey key, int workerId) {
    String workspaceName = key.getExecRoot().getBaseName();
    return workerBaseDir
        .getRelative(key.getWorkerTypeName() + "-" + workerId + "-" + key.getMnemonic())
        .getRelative(workspaceName);
  }

  Path getSandboxedWorkerPath(WorkerKey key) {
    String workspaceName = key.getExecRoot().getBaseName();
    return workerBaseDir
        .getRelative(key.getMnemonic() + "-" + key.getWorkerTypeName() + "-workdir")
        .getRelative(workspaceName);
  }

  /** Use the DefaultPooledObject implementation. */
  @Override
  public PooledObject<Worker> wrap(Worker worker) {
    return new DefaultPooledObject<>(worker);
  }

  /** When a worker process is discarded, destroy its process, too. */
  @Override
  public void destroyObject(WorkerKey key, PooledObject<Worker> p) {
    int workerId = p.getObject().getWorkerId();
    String msg =
        String.format(
            "Destroying %s %s (id %d, key hash %d)",
            key.getMnemonic(), key.getWorkerTypeName(), workerId, key.hashCode());
    WorkerLoggingHelper.logMessage(reporter, WorkerLoggingHelper.LogLevel.INFO, msg);
    p.getObject().destroy();
    if (eventBus != null) {
      eventBus.post(new WorkerDestroyedEvent(key.hashCode(), key.getMnemonic()));
    }
  }

  /**
   * Returns true if this worker is still valid. The worker is considered to be valid as long as its
   * process has not exited and its files have not changed on disk.
   */
  @Override
  public boolean validateObject(WorkerKey key, PooledObject<Worker> p) {
    Worker worker = p.getObject();
    if (worker.isDoomed()) {
      return false;
    }
    Optional<Integer> exitValue = worker.getExitValue();
    if (exitValue.isPresent()) {
      if (worker.diedUnexpectedly()) {
        String msg =
            String.format(
                "%s %s (id %d) has unexpectedly died with exit code %d.",
                key.getMnemonic(), key.getWorkerTypeName(), worker.getWorkerId(), exitValue.get());
        ErrorMessage errorMessage =
            ErrorMessage.builder()
                .message(msg)
                .logFile(worker.getLogFile())
                .logSizeLimit(4096)
                .build();
        WorkerLoggingHelper.logMessage(
            reporter, WorkerLoggingHelper.LogLevel.WARNING, errorMessage.toString());
      }
      return false;
    }
    boolean filesChanged =
        !key.getWorkerFilesCombinedHash().equals(worker.getWorkerFilesCombinedHash());

    if (filesChanged) {
      StringBuilder msg = new StringBuilder();
      msg.append(
          String.format(
              "%s %s (id %d) can no longer be used, because its files have changed on disk:",
              key.getMnemonic(), key.getWorkerTypeName(), worker.getWorkerId()));
      TreeSet<PathFragment> files = new TreeSet<>();
      files.addAll(key.getWorkerFilesWithDigests().keySet());
      files.addAll(worker.getWorkerFilesWithDigests().keySet());
      for (PathFragment file : files) {
        byte[] oldDigest = worker.getWorkerFilesWithDigests().get(file);
        byte[] newDigest = key.getWorkerFilesWithDigests().get(file);
        if (!Arrays.equals(oldDigest, newDigest)) {
          msg.append("\n")
              .append(file.getPathString())
              .append(": ")
              .append(hexStringForDebugging(oldDigest))
              .append(" -> ")
              .append(hexStringForDebugging(newDigest));
        }
      }

      WorkerLoggingHelper.logMessage(
          reporter, WorkerLoggingHelper.LogLevel.WARNING, msg.toString());
    }

    return !filesChanged;
  }

  private static String hexStringForDebugging(@Nullable byte[] bytes) {
    return bytes != null ? BaseEncoding.base16().encode(bytes).toLowerCase(Locale.ROOT) : "<none>";
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof WorkerFactory)) {
      return false;
    }
    WorkerFactory that = (WorkerFactory) o;
    return workerBaseDir.equals(that.workerBaseDir)
        && Objects.equals(this.hardenedSandboxOptions, that.hardenedSandboxOptions);
  }

  @Override
  public int hashCode() {
    return Objects.hash(workerBaseDir, hardenedSandboxOptions);
  }

  /** This class simultaneously sends messages to a logger and an event reporter. */
  private static final class WorkerLoggingHelper {
    private WorkerLoggingHelper() {}

    public static void logMessage(@Nullable Reporter reporter, LogLevel level, String message) {
      switch (level) {
        case INFO:
          logger.atInfo().log("%s", message);
          if (reporter != null) {
            reporter.handle(Event.info(message));
          }
          return;
        case WARNING:
          logger.atWarning().log("%s", message);
          if (reporter != null) {
            reporter.handle(Event.warn(message));
          }
          return;
      }
      throw new IllegalStateException(String.format("illegal logging level %s", level));
    }

    public static enum LogLevel {
      INFO("INFO"),
      WARNING("WARNING");

      private final String level;

      LogLevel(final String level) {
        this.level = level;
      }

      @Override
      public String toString() {
        return level;
      }
    }
  }
}
