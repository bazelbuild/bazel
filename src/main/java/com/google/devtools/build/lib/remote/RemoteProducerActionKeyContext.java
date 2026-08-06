// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.util.concurrent.Futures.immediateVoidFuture;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Directory;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.common.ProducerActionKeyContext;
import com.google.devtools.build.lib.remote.common.ProducerActionKeyContext.SyntheticTestActionKey;
import com.google.devtools.build.lib.remote.common.ActionKey;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** Remote implementation of digest-only producer action key computation. */
final class RemoteProducerActionKeyContext implements ProducerActionKeyContext {
  private final RemoteExecutionService remoteExecutionService;
  private final SpawnInputExpander spawnInputExpander;

  RemoteProducerActionKeyContext(RemoteExecutionService remoteExecutionService, Path execRoot) {
    this.remoteExecutionService = remoteExecutionService;
    this.spawnInputExpander = new SpawnInputExpander();
  }

  @Override
  public ActionKey computeActionKey(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      ArtifactPathResolver artifactPathResolver)
      throws IOException, ExecException, InterruptedException {
    SpawnExecutionContext context =
        new DigestOnlySpawnExecutionContext(spawn, inputMetadataProvider, artifactPathResolver);
    return remoteExecutionService.buildRemoteAction(spawn, context).getActionKey();
  }

  @Override
  public SyntheticTestActionKey computeSyntheticTestActionKey(
      ByteString logicalIdentity, ActionKey producerActionKey) {
    var digestUtil = remoteExecutionService.getDigestUtilForProducerKeyedTestCache();
    Command command =
        Command.newBuilder()
            .addArguments("bazel.producer_keyed_test_cache.v1")
            .addArguments(HexFormat.of().formatHex(logicalIdentity.toByteArray()))
            .build();
    Directory inputRoot = Directory.getDefaultInstance();
    Action syntheticAction =
        Action.newBuilder()
            .setCommandDigest(digestUtil.compute(command))
            .setInputRootDigest(digestUtil.compute(inputRoot))
            .setSalt(ByteString.copyFromUtf8("bazel.producer_keyed_test_cache.v1"))
            .build();
    return new SyntheticTestActionKey(
        digestUtil.computeActionKey(syntheticAction), syntheticAction, command, inputRoot);
  }

  @Override
  public void registerSyntheticTestActionKey(
      ActionExecutionMetadata action,
      SyntheticTestActionKey syntheticActionKey,
      boolean debugEnabled)
      throws InterruptedException {
    remoteExecutionService.registerSyntheticTestActionKey(action, syntheticActionKey, debugEnabled);
  }

  @Override
  public boolean restoreSyntheticTestActionAlias(ActionExecutionMetadata action)
      throws InterruptedException {
    return remoteExecutionService.restoreSyntheticTestActionAlias(action);
  }

  @Override
  public void finalizeSyntheticTestActionAlias(ActionExecutionMetadata action)
      throws InterruptedException {
    remoteExecutionService.finalizeSyntheticTestActionAlias(action);
  }

  private final class DigestOnlySpawnExecutionContext implements SpawnExecutionContext {
    private final Spawn spawn;
    private final InputMetadataProvider inputMetadataProvider;
    private final ArtifactPathResolver artifactPathResolver;
    @Nullable private Digest digest;

    private DigestOnlySpawnExecutionContext(
        Spawn spawn,
        InputMetadataProvider inputMetadataProvider,
        ArtifactPathResolver artifactPathResolver) {
      this.spawn = spawn;
      this.inputMetadataProvider = inputMetadataProvider;
      this.artifactPathResolver = artifactPathResolver;
    }

    @Override
    public int getId() {
      return 0;
    }

    @Override
    public void setDigest(Digest digest) {
      this.digest = digest;
    }

    @Override
    @Nullable
    public Digest getDigest() {
      return digest;
    }

    @Override
    public ListenableFuture<Void> prefetchInputs() {
      return immediateVoidFuture();
    }

    @Override
    public InputMetadataProvider getInputMetadataProvider() {
      return inputMetadataProvider;
    }

    @Override
    public ArtifactPathResolver getPathResolver() {
      return artifactPathResolver;
    }

    @Override
    public void lockOutputFiles(int exitCode, String errorMessage, FileOutErr outErr) {}

    @Override
    public boolean speculating() {
      return false;
    }

    @Override
    public Duration getTimeout() {
      return Duration.ZERO;
    }

    @Override
    public FileOutErr getFileOutErr() {
      return new FileOutErr();
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping(
        PathFragment baseDirectory, boolean willAccessRepeatedly) {
      return spawnInputExpander.getInputMapping(spawn, inputMetadataProvider, baseDirectory);
    }

    @Override
    public void report(ProgressStatus progress) {}

    @Override
    @Nullable
    public <T extends ActionContext> T getContext(Class<T> identifyingType) {
      return null;
    }

    @Override
    public boolean isRewindingEnabled() {
      return false;
    }

    @Override
    public void checkForLostInputs() {}

    @Override
    @Nullable
    public FileSystem getActionFileSystem() {
      return null;
    }

    @Override
    public ImmutableMap<String, String> getClientEnv() {
      return ImmutableMap.of();
    }
  }
}
