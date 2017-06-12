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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.Platform;
import com.google.protobuf.Duration;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeSet;

/** A client for the remote execution service. */
@ThreadSafe
final class RemoteSpawnRunner implements SpawnRunner {
  private final Path execRoot;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;

  private final GrpcRemoteExecutor executor;
  private final GrpcActionCache remoteCache;

  RemoteSpawnRunner(
      Path execRoot,
      RemoteOptions options,
      GrpcRemoteExecutor executor,
      GrpcActionCache remoteCache) {
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
    this.remoteCache = remoteCache;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionPolicy policy)
      throws ExecException, InterruptedException, IOException {
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
              Digests.computeDigest(command),
              repository.getMerkleDigest(inputRoot),
              // TODO(olaola): set sensible local and remote timouts.
              Spawns.getTimeoutSeconds(spawn, 120));

      ActionKey actionKey = Digests.computeActionKey(action);
      ActionResult result =
          options.remoteAcceptCached ? remoteCache.getCachedActionResult(actionKey) : null;
      if (result == null) {
        // Cache miss or we don't accept cache hits.
        // Upload the command and all the inputs into the remote cache.
        remoteCache.uploadBlob(command.toByteArray());
        remoteCache.uploadTree(repository, execRoot, inputRoot);
        // TODO(olaola): set BuildInfo and input total bytes as well.
        ExecuteRequest.Builder request =
            ExecuteRequest.newBuilder()
                .setInstanceName(options.remoteInstanceName)
                .setAction(action)
                .setWaitForCompletion(true)
                .setTotalInputFileCount(inputMap.size())
                .setSkipCacheLookup(!options.remoteAcceptCached);
        result = executor.executeRemotely(request.build()).getResult();
      }

      // TODO(ulfjack): Download stdout, stderr, and the output files in a single call.
      passRemoteOutErr(remoteCache, result, policy.getFileOutErr());
      remoteCache.downloadAllResults(result, execRoot);
      return new SpawnResult.Builder()
          .setStatus(Status.SUCCESS)  // Even if the action failed with non-zero exit code.
          .setExitCode(result.getExitCode())
          .build();
    } catch (StatusRuntimeException | CacheNotFoundException e) {
      throw new IOException(e);
    }
  }

  private Action buildAction(
      Collection<? extends ActionInput> outputs,
      Digest command,
      Digest inputRoot,
      long timeoutSeconds) {
    Action.Builder action = Action.newBuilder();
    action.setCommandDigest(command);
    action.setInputRootDigest(inputRoot);
    // Somewhat ugly: we rely on the stable order of outputs here for remote action caching.
    for (ActionInput output : outputs) {
      // TODO: output directories should be handled here, when they are supported.
      action.addOutputFiles(output.getExecPathString());
    }
    if (platform != null) {
      action.setPlatform(platform);
    }
    action.setTimeout(Duration.newBuilder().setSeconds(timeoutSeconds));
    return action.build();
  }

  private Command buildCommand(List<String> arguments, ImmutableMap<String, String> environment) {
    Command.Builder command = Command.newBuilder();
    command.addAllArguments(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(environment.keySet());
    for (String var : variables) {
      command.addEnvironmentVariablesBuilder().setName(var).setValue(environment.get(var));
    }
    return command.build();
  }

  private static void passRemoteOutErr(
      RemoteActionCache cache, ActionResult result, FileOutErr outErr) throws IOException {
    try {
      if (!result.getStdoutRaw().isEmpty()) {
        result.getStdoutRaw().writeTo(outErr.getOutputStream());
        outErr.getOutputStream().flush();
      } else if (result.hasStdoutDigest()) {
        byte[] stdoutBytes = cache.downloadBlob(result.getStdoutDigest());
        outErr.getOutputStream().write(stdoutBytes);
        outErr.getOutputStream().flush();
      }
      if (!result.getStderrRaw().isEmpty()) {
        result.getStderrRaw().writeTo(outErr.getErrorStream());
        outErr.getErrorStream().flush();
      } else if (result.hasStderrDigest()) {
        byte[] stderrBytes = cache.downloadBlob(result.getStderrDigest());
        outErr.getErrorStream().write(stderrBytes);
        outErr.getErrorStream().flush();
      }
    } catch (CacheNotFoundException e) {
      outErr.printOutLn("Failed to fetch remote stdout/err due to cache miss.");
      outErr.getOutputStream().flush();
    }
  }
}
