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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineLimits;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;

/**
 * A spawn action for genrules. Genrules are handled specially in that inputs and outputs are
 * checked for directories.
 */
public final class GenRuleAction extends SpawnAction {

  public static final String MNEMONIC = "Genrule";

  public GenRuleAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
      ImmutableSet<Artifact> outputs,
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
        AbstractAction.DEFAULT_RESOURCE_SET,
        commandLines,
        env,
        executionInfo,
        progressMessage,
        runfilesSupplier,
        MNEMONIC,
        /* stripOutputPaths= */ false);
  }

  @Override
  protected CommandLineLimits getCommandLineLimits() {
    return CommandLineLimits.UNLIMITED;
  }
}
