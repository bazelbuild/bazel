// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.remote;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.common.NetworkTime;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import javax.annotation.Nullable;

/**
 * A value class representing an action which can be executed remotely.
 *
 * <p>Terminology note: "action" is used here in the remote execution protocol sense, which is
 * equivalent to a Bazel "spawn" (a Bazel "action" being a higher-level concept).
 */
public class RemoteAction {

  private final Spawn spawn;
  private final SpawnExecutionContext spawnExecutionContext;
  private final RemoteActionExecutionContext remoteActionExecutionContext;
  private final RemotePathResolver remotePathResolver;
  @Nullable private final MerkleTree merkleTree;
  private final long inputBytes;
  private final long inputFiles;
  private final Digest commandHash;
  private final Command command;
  private final Action action;
  private final ActionKey actionKey;

  RemoteAction(
      Spawn spawn,
      SpawnExecutionContext spawnExecutionContext,
      RemoteActionExecutionContext remoteActionExecutionContext,
      RemotePathResolver remotePathResolver,
      MerkleTree merkleTree,
      Digest commandHash,
      Command command,
      Action action,
      ActionKey actionKey,
      boolean remoteDiscardMerkleTrees) {
    this.spawn = spawn;
    this.spawnExecutionContext = spawnExecutionContext;
    this.remoteActionExecutionContext = remoteActionExecutionContext;
    this.remotePathResolver = remotePathResolver;
    this.merkleTree = remoteDiscardMerkleTrees ? null : merkleTree;
    this.inputBytes = merkleTree.getInputBytes();
    this.inputFiles = merkleTree.getInputFiles();
    this.commandHash = commandHash;
    this.command = command;
    this.action = action;
    this.actionKey = actionKey;
  }

  public RemoteActionExecutionContext getRemoteActionExecutionContext() {
    return remoteActionExecutionContext;
  }

  public SpawnExecutionContext getSpawnExecutionContext() {
    return spawnExecutionContext;
  }

  /** Returns the {@link Spawn} that owns this action. */
  public Spawn getSpawn() {
    return spawn;
  }

  /**
   * Returns the sum of file sizes plus protobuf sizes used to represent the inputs of this action.
   */
  public long getInputBytes() {
    return inputBytes;
  }

  /** Returns the number of input files of this action. */
  public long getInputFiles() {
    return inputFiles;
  }

  /** Returns the id this is action. */
  public String getActionId() {
    return actionKey.getDigest().getHash();
  }

  /** Returns the {@link ActionKey} of this action. */
  public ActionKey getActionKey() {
    return actionKey;
  }

  /** Returns underlying {@link Action} of this remote action. */
  public Action getAction() {
    return action;
  }

  public Digest getCommandHash() {
    return commandHash;
  }

  public Command getCommand() {
    return command;
  }

  public RemotePathResolver getRemotePathResolver() {
    return remotePathResolver;
  }

  @Nullable
  public MerkleTree getMerkleTree() {
    return merkleTree;
  }

  /**
   * Returns the {@link NetworkTime} instance used to measure the network time during the action
   * execution.
   */
  public NetworkTime getNetworkTime() {
    return remoteActionExecutionContext.getNetworkTime();
  }
}
