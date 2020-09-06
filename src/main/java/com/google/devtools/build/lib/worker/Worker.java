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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.gson.stream.JsonReader;
import com.google.protobuf.util.JsonFormat;
import com.google.protobuf.util.JsonFormat.Printer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.SortedMap;
import javax.annotation.Nullable;

/**
 * Interface to a worker process running as a child process.
 *
 * <p>A worker process must follow this protocol to be usable via this class: The worker process is
 * spawned on demand. The worker process is free to exit whenever necessary, as new instances will
 * be relaunched automatically. Communication happens via the WorkerProtocol protobuf, sent to and
 * received from the worker process via stdin / stdout.
 *
 * <p>Other code in Blaze can talk to the worker process via input / output streams provided by this
 * class.
 */
class Worker {
  /** An unique identifier of the work process. */
  protected final WorkerKey workerKey;
  /** An unique ID of the worker. It will be used in WorkRequest and WorkResponse as well. */
  protected final int workerId;
  /** The execution root of the worker. */
  protected final Path workDir;
  /** The path of the log file for this worker. */
  private final Path logFile;
  /** Stream for reading the protobuf WorkResponse. */
  @Nullable protected RecordingInputStream protoRecordingStream;
  /** Reader for reading the JSON WorkResponse. */
  @Nullable protected JsonReader jsonReader;
  /** Printer for writing the JSON WorkRequest bytes */
  @Nullable protected Printer jsonPrinter;
  /** BufferedWriter for the JSON WorkRequest bytes */
  @Nullable protected BufferedWriter jsonWriter;

  private Subprocess process;
  private Thread shutdownHook;
  /** True if we deliberately destroyed this process. */
  private boolean wasDestroyed;

  Worker(WorkerKey workerKey, int workerId, final Path workDir, Path logFile) {
    this.workerKey = workerKey;
    this.workerId = workerId;
    this.workDir = workDir;
    this.logFile = logFile;

    final Worker self = this;
    this.shutdownHook =
        new Thread(
            () -> {
              try {
                self.shutdownHook = null;
                self.destroy();
              } catch (IOException e) {
                // We can't do anything here.
              }
            });
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  Subprocess createProcess() throws IOException {
    ImmutableList<String> args = workerKey.getArgs();
    File executable = new File(args.get(0));
    if (!executable.isAbsolute() && executable.getParent() != null) {
      List<String> newArgs = new ArrayList<>(args);
      newArgs.set(0, new File(workDir.getPathFile(), newArgs.get(0)).getAbsolutePath());
      args = ImmutableList.copyOf(newArgs);
    }
    SubprocessBuilder processBuilder = new SubprocessBuilder();
    processBuilder.setArgv(args);
    processBuilder.setWorkingDirectory(workDir.getPathFile());
    processBuilder.setStderr(logFile.getPathFile());
    processBuilder.setEnv(workerKey.getEnv());
    return processBuilder.start();
  }

  void destroy() throws IOException {
    if (shutdownHook != null) {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
    }
    if (process != null) {
      if (workerKey.getProtocolFormat() == WorkerProtocolFormat.JSON) {
        jsonReader.close();
        jsonWriter.close();
      }

      wasDestroyed = true;
      process.destroyAndWait();
    }
  }

  /**
   * Returns a unique id for this worker. This is used to distinguish different worker processes in
   * logs and messages.
   */
  int getWorkerId() {
    return this.workerId;
  }

  /** Returns the path of the log file for this worker. */
  public Path getLogFile() {
    return logFile;
  }

  HashCode getWorkerFilesCombinedHash() {
    return workerKey.getWorkerFilesCombinedHash();
  }

  SortedMap<PathFragment, HashCode> getWorkerFilesWithHashes() {
    return workerKey.getWorkerFilesWithHashes();
  }

  boolean isAlive() {
    // This is horrible, but Process.isAlive() is only available from Java 8 on and this is the
    // best we can do prior to that.
    return !process.finished();
  }

  /** Returns true if this process is dead but we didn't deliberately kill it. */
  boolean diedUnexpectedly() {
    return process != null && !wasDestroyed && !process.isAlive();
  }

  /** Returns the exit value of this worker's process, if it has exited. */
  public Optional<Integer> getExitValue() {
    return process != null && !process.isAlive()
        ? Optional.of(process.exitValue())
        : Optional.empty();
  }

  // TODO(karlgray): Create wrapper class that handles writing and reading JSON worker protocol to
  // and from stream.
  void putRequest(WorkRequest request) throws IOException {
    switch (workerKey.getProtocolFormat()) {
      case JSON:
        checkNotNull(jsonWriter, "Did prepareExecution get called before putRequest?");
        checkNotNull(jsonPrinter, "Did prepareExecution get called before putRequest?");

        jsonPrinter.appendTo(request, jsonWriter);
        jsonWriter.flush();
        break;

      case PROTO:
        request.writeDelimitedTo(process.getOutputStream());
        process.getOutputStream().flush();
        break;
    }
  }

  WorkResponse getResponse() throws IOException {
    switch (workerKey.getProtocolFormat()) {
      case JSON:
        checkNotNull(jsonReader, "Did prepareExecution get called before putRequest?");

        return readResponse(jsonReader);

      case PROTO:
        protoRecordingStream = new RecordingInputStream(process.getInputStream());
        protoRecordingStream.startRecording(4096);
        // response can be null when the worker has already closed
        // stdout at this point and thus the InputStream is at EOF.
        return WorkResponse.parseDelimitedFrom(protoRecordingStream);
    }

    throw new IllegalStateException(
        "Invalid protocol format; protocol formats are currently proto or json");
  }

  private static WorkResponse readResponse(JsonReader reader) throws IOException {
    Integer exitCode = null;
    String output = null;
    Integer requestId = null;

    reader.beginObject();
    while (reader.hasNext()) {
      String name = reader.nextName();
      switch (name) {
        case "exitCode":
          if (exitCode != null) {
            throw new IOException("Work response cannot have more than one exit code");
          }
          exitCode = reader.nextInt();
          break;
        case "output":
          if (output != null) {
            throw new IOException("Work response cannot have more than one output");
          }
          output = reader.nextString();
          break;
        case "requestId":
          if (requestId != null) {
            throw new IOException("Work response cannot have more than one requestId");
          }
          requestId = reader.nextInt();
          break;
        default:
          throw new IOException(name + " is an incorrect field in work response");
      }
    }
    reader.endObject();

    WorkResponse.Builder responseBuilder = WorkResponse.newBuilder();

    if (exitCode != null) {
      responseBuilder.setExitCode(exitCode);
    }
    if (output != null) {
      responseBuilder.setOutput(output);
    }
    if (requestId != null) {
      responseBuilder.setRequestId(requestId);
    }

    return responseBuilder.build();
  }

  String getRecordingStreamMessage() {
    protoRecordingStream.readRemaining();
    return protoRecordingStream.getRecordedDataAsString();
  }

  public void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException {
    if (process == null) {
      process = createProcess();

      if (workerKey.getProtocolFormat() == WorkerProtocolFormat.JSON) {
        checkState(jsonReader == null, "JSON streams inconsistent with process status");
        checkState(jsonPrinter == null, "JSON streams inconsistent with process status");
        checkState(jsonWriter == null, "JSON streams inconsistent with process status");

        jsonReader =
            new JsonReader(
                new BufferedReader(new InputStreamReader(process.getInputStream(), UTF_8)));
        jsonReader.setLenient(true);
        jsonPrinter = JsonFormat.printer().omittingInsignificantWhitespace();
        jsonWriter = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), UTF_8));
      }
    }
  }

  public void finishExecution(Path execRoot) throws IOException {}
}
