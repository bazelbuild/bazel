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

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.Platform;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.protobuf.Message;
import io.grpc.Context;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;

/** The remote package's implementation of {@link RepositoryRemoteExecutor}. */
public class RemoteRepositoryRemoteExecutor implements RepositoryRemoteExecutor {

  private final RemoteExecutionCache remoteCache;
  private final GrpcRemoteExecutor remoteExecutor;
  private final DigestUtil digestUtil;
  private final Context requestCtx;

  private final String remoteInstanceName;
  private final boolean acceptCached;

  public RemoteRepositoryRemoteExecutor(
      RemoteExecutionCache remoteCache,
      GrpcRemoteExecutor remoteExecutor,
      DigestUtil digestUtil,
      Context requestCtx,
      String remoteInstanceName,
      boolean acceptCached) {
    this.remoteCache = remoteCache;
    this.remoteExecutor = remoteExecutor;
    this.digestUtil = digestUtil;
    this.requestCtx = requestCtx;
    this.remoteInstanceName = remoteInstanceName;
    this.acceptCached = acceptCached;
  }

  private ExecutionResult downloadOutErr(ActionResult result)
      throws IOException, InterruptedException {
    byte[] stdout = new byte[0];
    if (!result.getStdoutRaw().isEmpty()) {
      stdout = result.getStdoutRaw().toByteArray();
    } else if (result.hasStdoutDigest()) {
      stdout = Utils.getFromFuture(remoteCache.downloadBlob(result.getStdoutDigest()));
    }

    byte[] stderr = new byte[0];
    if (!result.getStderrRaw().isEmpty()) {
      stderr = result.getStderrRaw().toByteArray();
    } else if (result.hasStderrDigest()) {
      stderr = Utils.getFromFuture(remoteCache.downloadBlob(result.getStderrDigest()));
    }

    return new ExecutionResult(result.getExitCode(), stdout, stderr);
  }

  @Override
  public ExecutionResult execute(
      ImmutableList<String> arguments,
      ImmutableMap<String, String> executionProperties,
      ImmutableMap<String, String> environment,
      String workingDirectory,
      Duration timeout)
      throws IOException, InterruptedException {
    Context prev = requestCtx.attach();
    try {
      Platform platform = PlatformUtils.buildPlatformProto(executionProperties);
      Command command =
          RemoteSpawnRunner.buildCommand(
              /* outputs= */ ImmutableList.of(),
              arguments,
              environment,
              platform,
              workingDirectory);
      Digest commandHash = digestUtil.compute(command);
      MerkleTree merkleTree =
          MerkleTree.build(
              ImmutableSortedMap.of(),
              /* metadataProvider= */ null,
              /* execRoot= */ null,
              digestUtil);
      Action action =
          RemoteSpawnRunner.buildAction(
              commandHash, merkleTree.getRootDigest(), timeout, acceptCached);
      Digest actionDigest = digestUtil.compute(action);
      ActionKey actionKey = new ActionKey(actionDigest);
      ActionResult actionResult =
          remoteCache.downloadActionResult(actionKey, /* inlineOutErr= */ true);
      if (actionResult == null || actionResult.getExitCode() != 0) {
        Map<Digest, Message> additionalInputs = Maps.newHashMapWithExpectedSize(2);
        additionalInputs.put(actionDigest, action);
        additionalInputs.put(commandHash, command);
        remoteCache.ensureInputsPresent(merkleTree, additionalInputs);

        ExecuteRequest executeRequest =
            ExecuteRequest.newBuilder()
                .setActionDigest(actionDigest)
                .setInstanceName(remoteInstanceName)
                .setSkipCacheLookup(!acceptCached)
                .build();

        ExecuteResponse response = remoteExecutor.executeRemotely(executeRequest);
        actionResult = response.getResult();
      }
      return downloadOutErr(actionResult);
    } finally {
      requestCtx.detach(prev);
    }
  }
}
