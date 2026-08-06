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

package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Directory;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.remote.common.ActionKey;
import com.google.protobuf.ByteString;
import java.io.IOException;

/** Action context for computing an exact remote action key without executing or uploading. */
public interface ProducerActionKeyContext extends ActionContext {
  record SyntheticTestActionKey(
      ActionKey actionKey, Action action, Command command, Directory inputRoot) {}

  ActionKey computeActionKey(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      ArtifactPathResolver artifactPathResolver)
      throws IOException, ExecException, InterruptedException;

  SyntheticTestActionKey computeSyntheticTestActionKey(
      ByteString logicalIdentity, ActionKey producerActionKey);

  void registerSyntheticTestActionKey(
      ActionExecutionMetadata action,
      SyntheticTestActionKey syntheticActionKey,
      boolean debugEnabled)
      throws InterruptedException;

  /** Returns whether a registered alias was fully restored to the action's output paths. */
  boolean restoreSyntheticTestActionAlias(ActionExecutionMetadata action)
      throws InterruptedException;

  /** Adds action-level test status to an alias after normal test completion. */
  void finalizeSyntheticTestActionAlias(ActionExecutionMetadata action) throws InterruptedException;
}
