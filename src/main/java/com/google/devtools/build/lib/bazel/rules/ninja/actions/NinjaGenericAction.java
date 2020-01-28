// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.TrackSourceDirectoriesFlag;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class NinjaGenericAction extends SpawnAction {
  private static final String MNEMONIC = "NinjaGenericAction";

  public NinjaGenericAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
      List<? extends Artifact> outputs,
      CommandLines commandLines,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo,
      RunfilesSupplier runfilesSupplier) {
    super(owner, tools, inputs, outputs, Iterables.getFirst(outputs, null),
        AbstractAction.DEFAULT_RESOURCE_SET, commandLines,
        CommandLineLimits.UNLIMITED,
        true, env, executionInfo, createProgressMessage(outputs), runfilesSupplier, MNEMONIC,
        false, null, null);
  }

  private static CharSequence createProgressMessage(List<? extends Artifact> outputs) {
    return String.format("running Ninja targets: '%s'",
        outputs.stream().map(Artifact::getFilename).collect(Collectors.joining(", ")));
  }

  @Override
  protected void beforeExecute(ActionExecutionContext actionExecutionContext) throws IOException {
    if (!TrackSourceDirectoriesFlag.trackSourceDirectories()) {
      checkInputsForDirectories(
          actionExecutionContext.getEventHandler(), actionExecutionContext.getMetadataProvider());
    }
  }

  @Override
  protected void afterExecute(ActionExecutionContext actionExecutionContext,
      List<SpawnResult> spawnResults) {
    checkOutputsForDirectories(actionExecutionContext);
  }
}
