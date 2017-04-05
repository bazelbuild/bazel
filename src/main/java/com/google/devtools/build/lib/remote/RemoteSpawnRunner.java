// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.RemoteProtocol.Action;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.Command;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.Platform;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Collection;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeSet;

/**
 * A client for the remote execution service.
 */
@ThreadSafe
final class RemoteSpawnRunner implements SpawnRunner {
  private final Path execRoot;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;

  private final GrpcRemoteExecutor executor;

  RemoteSpawnRunner(
      Path execRoot,
      RemoteOptions options,
      GrpcRemoteExecutor executor) {
    this.execRoot = execRoot;
    this.options = options;
    if (options.experimentalRemotePlatformOverride != null) {
      Platform.Builder platformBuilder = Platform.newBuilder();
      try {
        TextFormat.getParser().merge(options.experimentalRemotePlatformOverride, platformBuilder);
      } catch (ParseException e) {
        throw new RuntimeException("Failed to parse --experimental_remote_platform_override", e);
      }
      platform = platformBuilder.build();
    } else {
      platform = null;
    }
    this.executor = executor;
  }

  RemoteSpawnRunner(
      Path execRoot,
      RemoteOptions options) {
    this(execRoot, options, connect(options));
  }

  private static GrpcRemoteExecutor connect(RemoteOptions options) {
    Preconditions.checkArgument(GrpcRemoteExecutor.isRemoteExecutionOptions(options));
    ManagedChannel channel;
    try {
      channel = RemoteUtils.createChannel(options.remoteWorker);
    } catch (URISyntaxException e) {
      throw new RuntimeException(e);
    }
    return new GrpcRemoteExecutor(channel, options);
  }

  @Override
  public SpawnResult exec(
      Spawn spawn,
      SpawnExecutionPolicy policy) throws InterruptedException, IOException {
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    if (owner.getOwner() != null) {
      policy.report(ProgressStatus.EXECUTING);
    }

    try {
      // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
      TreeNodeRepository repository =
          new TreeNodeRepository(execRoot, policy.getActionInputFileCache());
      SortedMap<PathFragment, ActionInput> inputMap = policy.getInputMapping();
      TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
      repository.computeMerkleDigests(inputRoot);
      Command command = buildCommand(spawn.getArguments(), spawn.getEnvironment());
      Action action =
          buildAction(
              spawn.getOutputFiles(),
              ContentDigests.computeDigest(command),
              repository.getMerkleDigest(inputRoot));

      ActionKey actionKey = ContentDigests.computeActionKey(action);
      ActionResult result =
          this.options.remoteAcceptCached ? executor.getCachedActionResult(actionKey) : null;
      if (result == null) {
        // Cache miss or we don't accept cache hits.
        // Upload the command and all the inputs into the remote cache.
        executor.uploadBlob(command.toByteArray());
        // TODO(olaola): this should use the ActionInputFileCache for SHA1 digests!
        executor.uploadTree(repository, execRoot, inputRoot);
        // TODO(olaola): set BuildInfo and input total bytes as well.
        ExecuteRequest.Builder request =
            ExecuteRequest.newBuilder()
                .setAction(action)
                .setAcceptCached(this.options.remoteAcceptCached)
                .setTotalInputFileCount(inputMap.size())
                .setTimeoutMillis(policy.getTimeoutMillis());
        ExecuteReply reply = executor.executeRemotely(request.build());
        ExecutionStatus status = reply.getStatus();

        if (!status.getSucceeded()
            && (status.getError() != ExecutionStatus.ErrorCode.EXEC_FAILED)) {
          return new SpawnResult.Builder()
              // TODO(ulfjack): Improve the translation of the error status.
              .setStatus(Status.EXECUTION_FAILED)
              .setExitCode(-1)
              .build();
        }

        result = reply.getResult();
      }

      // TODO(ulfjack): Download stdout, stderr, and the output files in a single call.
      passRemoteOutErr(executor, result, policy.getFileOutErr());
      executor.downloadAllResults(result, execRoot);
      return new SpawnResult.Builder()
          .setStatus(Status.SUCCESS)
          .setExitCode(result.getReturnCode())
          .build();
    } catch (StatusRuntimeException | CacheNotFoundException e) {
      throw new IOException(e);
    }
  }

  private Action buildAction(
      Collection<? extends ActionInput> outputs, ContentDigest command, ContentDigest inputRoot) {
    Action.Builder action = Action.newBuilder();
    action.setCommandDigest(command);
    action.setInputRootDigest(inputRoot);
    // Somewhat ugly: we rely on the stable order of outputs here for remote action caching.
    for (ActionInput output : outputs) {
      action.addOutputPath(output.getExecPathString());
    }
    if (platform != null) {
      action.setPlatform(platform);
    }
    return action.build();
  }

  private Command buildCommand(List<String> arguments, ImmutableMap<String, String> environment) {
    Command.Builder command = Command.newBuilder();
    command.addAllArgv(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(environment.keySet());
    for (String var : variables) {
      command.addEnvironmentBuilder().setVariable(var).setValue(environment.get(var));
    }
    return command.build();
  }

  private static void passRemoteOutErr(
      RemoteActionCache cache, ActionResult result, FileOutErr outErr)
          throws CacheNotFoundException {
    ImmutableList<byte[]> streams =
        cache.downloadBlobs(ImmutableList.of(result.getStdoutDigest(), result.getStderrDigest()));
    outErr.printOut(new String(streams.get(0), UTF_8));
    outErr.printErr(new String(streams.get(1), UTF_8));
  }
}
