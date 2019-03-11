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

package com.google.devtools.build.lib.rules.genrule;

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
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.skyframe.TrackSourceDirectoriesFlag;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.IOException;

/**
 * A spawn action for genrules. Genrules are handled specially in that inputs and outputs are
 * checked for directories.
 */
@AutoCodec
public class GenRuleAction extends SpawnAction {

  public static final String MNEMONIC = "Genrule";

  public GenRuleAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      CommandLines commandLines,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo,
      RunfilesSupplier runfilesSupplier,
      CharSequence progressMessage) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        Iterables.getFirst(outputs, null),
        AbstractAction.DEFAULT_RESOURCE_SET,
        commandLines,
        CommandLineLimits.UNLIMITED,
        false,
        env,
        executionInfo,
        progressMessage,
        runfilesSupplier,
        MNEMONIC,
        false,
        null);
  }

  @Override
  protected void beforeExecute(ActionExecutionContext actionExecutionContext) throws IOException {
    if (!TrackSourceDirectoriesFlag.trackSourceDirectories()) {
      checkInputsForDirectories(
          actionExecutionContext.getEventHandler(), actionExecutionContext.getMetadataProvider());
    }
  }

  @Override
  protected void afterExecute(ActionExecutionContext actionExecutionContext) {
    checkOutputsForDirectories(actionExecutionContext);
  }
}
