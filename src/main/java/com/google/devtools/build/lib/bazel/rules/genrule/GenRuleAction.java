// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.genrule;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/**
 * A spawn action for genrules. Genrules are handled specially in that inputs and outputs are
 * checked for directories.
 */
public final class GenRuleAction extends SpawnAction {

  private static final ResourceSet GENRULE_RESOURCES =
      // Not chosen scientifically/carefully.  300MB memory, 100% CPU, no I/O.
      ResourceSet.createWithRamCpuIo(300, 1.0, 0.0);

  public GenRuleAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      List<String> argv,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      ImmutableMap<PathFragment, Artifact> runfilesManifests,
      String progressMessage) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        GENRULE_RESOURCES,
        CommandLine.of(argv, false),
        environment,
        executionInfo,
        progressMessage,
        runfilesManifests,
        "Genrule",
        false,
        null,
        false);
  }

  @Override
  protected void internalExecute(
      ActionExecutionContext actionExecutionContext) throws ExecException, InterruptedException {
    EventHandler reporter = actionExecutionContext.getExecutor().getEventHandler();
    checkInputsForDirectories(reporter, actionExecutionContext.getMetadataHandler());
    super.internalExecute(actionExecutionContext);
    checkOutputsForDirectories(reporter);
  }
}
