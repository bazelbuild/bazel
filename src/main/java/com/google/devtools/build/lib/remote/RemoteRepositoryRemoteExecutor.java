// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.remote.util.Utils.buildAction;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.CombinedCache.CachedActionResult;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.merkletree.v2.MerkleTreeComputer;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.Message;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import java.util.TreeSet;

/** The remote package's implementation of {@link RepositoryRemoteExecutor}. */
public class RemoteRepositoryRemoteExecutor implements RepositoryRemoteExecutor {

  private final RemoteExecutionCache remoteCache;
  private final RemoteExecutionClient remoteExecutor;
  private final DigestUtil digestUtil;
  private final String buildRequestId;
  private final String commandId;

  private final String remoteInstanceName;
  private final boolean acceptCached;

  public RemoteRepositoryRemoteExecutor(
      RemoteExecutionCache remoteCache,
      RemoteExecutionClient remoteExecutor,
      DigestUtil digestUtil,
      String buildRequestId,
      String commandId,
      String remoteInstanceName,
      boolean acceptCached) {
    this.remoteCache = remoteCache;
    this.remoteExecutor = remoteExecutor;
    this.digestUtil = digestUtil;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.remoteInstanceName = remoteInstanceName;
    this.acceptCached = acceptCached;
  }

  private ExecutionResult downloadOutErr(RemoteActionExecutionContext context, ActionResult result)
      throws IOException, InterruptedException {
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.REMOTE_DOWNLOAD, "download stdout/stderr")) {
      byte[] stdout = new byte[0];
      if (!result.getStdoutRaw().isEmpty()) {
        stdout = result.getStdoutRaw().toByteArray();
      } else if (result.hasStdoutDigest()) {
        stdout =
            Utils.getFromFuture(
                remoteCache.downloadBlob(
                    context, "<stdout>", /* execPath= */ null, result.getStdoutDigest()));
      }

      byte[] stderr = new byte[0];
      if (!result.getStderrRaw().isEmpty()) {
        stderr = result.getStderrRaw().toByteArray();
      } else if (result.hasStderrDigest()) {
        stderr =
            Utils.getFromFuture(
                remoteCache.downloadBlob(
                    context, "<stderr>", /* execPath= */ null, result.getStderrDigest()));
      }

      return new ExecutionResult(result.getExitCode(), stdout, stderr);
    }
  }

  @Override
  public ExecutionResult execute(
      ImmutableList<String> arguments,
      ImmutableSortedMap<PathFragment, Path> inputFiles,
      ImmutableMap<String, String> executionProperties,
      ImmutableMap<String, String> environment,
      String workingDirectory,
      Duration timeout)
      throws IOException, InterruptedException {
    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(buildRequestId, commandId, "repository_rule", null);
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(metadata);

    Platform platform = PlatformUtils.buildPlatformProto(executionProperties);

    Command.Builder commandBuilder = Command.newBuilder().addAllArguments(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(environment.keySet());
    for (String var : variables) {
      commandBuilder.addEnvironmentVariablesBuilder().setName(var).setValue(environment.get(var));
    }
    if (platform != null) {
      commandBuilder.setPlatform(platform);
    }
    if (workingDirectory != null) {
      commandBuilder.setWorkingDirectory(workingDirectory);
    }

    Command command = commandBuilder.build();
    Digest commandHash = digestUtil.compute(command);
    MerkleTreeComputer.MerkleTree merkleTree =
        new MerkleTreeComputer(
                digestUtil, /* remoteExecutionCache= */ null, buildRequestId, commandId)
            .buildForFiles(inputFiles);
    Action action =
        buildAction(
            commandHash,
            merkleTree.rootDigest(),
            platform,
            timeout,
            acceptCached,
            /* salt= */ null);
    Digest actionDigest = digestUtil.compute(action);
    ActionKey actionKey = new ActionKey(actionDigest);
    CachedActionResult cachedActionResult;
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.REMOTE_CACHE_CHECK, "check cache hit")) {
      cachedActionResult =
          remoteCache.downloadActionResult(
              context,
              actionKey,
              /* inlineOutErr= */ true,
              /* inlineOutputFiles= */ ImmutableSet.of());
    }
    ActionResult actionResult = null;
    if (cachedActionResult != null) {
      actionResult = cachedActionResult.actionResult();
    }
    if (actionResult == null || actionResult.getExitCode() != 0) {
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.UPLOAD_TIME, "upload missing inputs")) {
        Map<Digest, Message> additionalInputs = Maps.newHashMapWithExpectedSize(2);
        additionalInputs.put(actionDigest, action);
        additionalInputs.put(commandHash, command);

        remoteCache.ensureInputsPresent(
            context,
            merkleTree,
            additionalInputs,
            /* force= */ true,
            /* remotePathResolver= */ null);
      }

      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.REMOTE_EXECUTION, "execute remotely")) {
        ExecuteRequest executeRequest =
            ExecuteRequest.newBuilder()
                .setActionDigest(actionDigest)
                .setInstanceName(remoteInstanceName)
                .setDigestFunction(digestUtil.getDigestFunction())
                .setSkipCacheLookup(!acceptCached)
                .build();

        ExecuteResponse response =
            remoteExecutor.executeRemotely(context, executeRequest, OperationObserver.NO_OP);
        actionResult = response.getResult();
      }
    }
    return downloadOutErr(context, actionResult);
  }
}
