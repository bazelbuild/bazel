// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.RemoteProtocol.FileEntry;
import com.google.devtools.build.lib.remote.RemoteProtocol.RemoteWorkRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.RemoteWorkResponse;
import com.google.devtools.build.lib.remote.RemoteWorkGrpc.RemoteWorkFutureStub;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Implementation of {@link RemoteWorkExecutor} that uses MemcacheActionCache and gRPC for
 * communicating the work, inputs and outputs.
 */
@ThreadSafe
public class MemcacheWorkExecutor implements RemoteWorkExecutor {
  /**
   * A cache used to store the input and output files as well as the build status
   * of the remote work.
   */
  protected final MemcacheActionCache cache;

  /** Execution root for running this work locally. */
  private final Path execRoot;

  /** Channel over which to send work to run remotely. */
  private final ManagedChannel channel;

  private static final int MAX_WORK_SIZE_BYTES = 1024 * 1024 * 512;

  /**
   * This constructor is used when this class is used in a client.
   * It requires a host address and port to connect to a remote service.
   */
  private MemcacheWorkExecutor(MemcacheActionCache cache, String host, int port) {
    this.cache = cache;
    this.execRoot = null;
    this.channel = NettyChannelBuilder.forAddress(host, port).usePlaintext(true).build();
  }

  /**
   * This constructor is used when this class is used in the remote worker.
   * A path to the execution root is needed for executing work locally.
   */
  private MemcacheWorkExecutor(MemcacheActionCache cache, Path execRoot) {
    this.cache = cache;
    this.execRoot = execRoot;
    this.channel = null;
  }

  /**
   * Create an instance of MemcacheWorkExecutor that talks to a remote server.
   * @param cache An instance of MemcacheActionCache.
   * @param host Hostname of the server to connect to.
   * @param port Port of the server to connect to.
   * @return An instance of MemcacheWorkExecutor that talks to a remote server.
   */
  public static MemcacheWorkExecutor createRemoteWorkExecutor(
      MemcacheActionCache cache, String host, int port) {
    return new MemcacheWorkExecutor(cache, host, port);
  }

  /**
   * Create an instance of MemcacheWorkExecutor that runs locally.
   * @param cache An instance of MemcacheActionCache.
   * @param execRoot Path of the execution root where work is executed.
   * @return An instance of MemcacheWorkExecutor tthat runs locally in the execution root.
   */
  public static MemcacheWorkExecutor createLocalWorkExecutor(
      MemcacheActionCache cache, Path execRoot) {
    return new MemcacheWorkExecutor(cache, execRoot);
  }

  @Override
  public ListenableFuture<RemoteWorkResponse> executeRemotely(
      Path execRoot,
      ActionInputFileCache actionCache,
      String actionOutputKey,
      Collection<String> arguments,
      Collection<ActionInput> inputs,
      ImmutableMap<String, String> environment,
      Collection<? extends ActionInput> outputs,
      int timeout)
      throws IOException, WorkTooLargeException {
    RemoteWorkRequest.Builder work = RemoteWorkRequest.newBuilder();
    work.setOutputKey(actionOutputKey);

    long workSize = 0;
    for (ActionInput input : inputs) {
      if (!(input instanceof Artifact)) {
        continue;
      }
      if (!actionCache.isFile((Artifact) input)) {
        continue;
      }
      workSize += actionCache.getSizeInBytes(input);
    }

    if (workSize > MAX_WORK_SIZE_BYTES) {
      throw new WorkTooLargeException("Work is too large: " + workSize + " bytes.");
    }

    // Save all input files to cache.
    for (ActionInput input : inputs) {
      Path file = execRoot.getRelative(input.getExecPathString());

      if (file.isDirectory()) {
        // TODO(alpha): Handle this case better.
        throw new UnsupportedOperationException(
            "Does not support directory artifacts: " + file + ".");
      }

      String contentKey = cache.putFileIfNotExist(actionCache, input);
      work.addInputFilesBuilder()
          .setPath(input.getExecPathString())
          .setContentKey(contentKey)
          .setExecutable(file.isExecutable());
    }

    work.addAllArguments(arguments);
    work.getMutableEnvironment().putAll(environment);
    for (ActionInput output : outputs) {
      work.addOutputFilesBuilder().setPath(output.getExecPathString());
    }

    RemoteWorkFutureStub stub = RemoteWorkGrpc.newFutureStub(channel);
    work.setTimeout(timeout);
    return stub.executeSynchronously(work.build());
  }

  /** Execute a work item locally. */
  public RemoteWorkResponse executeLocally(RemoteWorkRequest work) throws IOException {
    ByteArrayOutputStream stdout = new ByteArrayOutputStream();
    ByteArrayOutputStream stderr = new ByteArrayOutputStream();
    try {
      // Prepare directories and input files.
      for (FileEntry input : work.getInputFilesList()) {
        Path file = execRoot.getRelative(input.getPath());
        FileSystemUtils.createDirectoryAndParents(file.getParentDirectory());
        cache.writeFile(input.getContentKey(), file, input.getExecutable());
      }

      List<Path> outputs = new ArrayList<>(work.getOutputFilesList().size());
      for (FileEntry output : work.getOutputFilesList()) {
        Path file = execRoot.getRelative(output.getPath());
        if (file.exists()) {
          throw new FileAlreadyExistsException("Output file already exists: " + file);
        }
        FileSystemUtils.createDirectoryAndParents(file.getParentDirectory());
        outputs.add(file);
      }

      Command cmd =
          new Command(
              work.getArgumentsList().toArray(new String[] {}),
              work.getEnvironment(),
              new File(execRoot.getPathString()));
      CommandResult result =
          cmd.execute(Command.NO_INPUT, Command.NO_OBSERVER, stdout, stderr, true);
      cache.putActionOutput(work.getOutputKey(), execRoot, outputs);
      return RemoteWorkResponse.newBuilder()
          .setSuccess(result.getTerminationStatus().success())
          .setOut(stdout.toString())
          .setErr(stderr.toString())
          .build();
    } catch (CommandException e) {
      return RemoteWorkResponse.newBuilder()
          .setSuccess(false)
          .setOut(stdout.toString())
          .setErr(stderr.toString())
          .setException(e.toString())
          .build();
    }
  }
}
