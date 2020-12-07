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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/** Implementation for the {@code fdo_profile} rule. */
@Immutable
public final class FdoProfile implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, ActionConflictException {
    CcCommon.checkRuleLoadedThroughMacro(ruleContext);
    FdoInputFile inputFile = FdoInputFile.fromProfileRule(ruleContext);
    if (ruleContext.hasErrors()) {
      return null;
    }

    Artifact protoProfileArtifact = ruleContext.getPrerequisiteArtifact("proto_profile");
    if (protoProfileArtifact != null && !protoProfileArtifact.isSourceArtifact()) {
      ruleContext.attributeError("proto_profile", "the target is not an input file");
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addNativeDeclaredProvider(new FdoProfileProvider(inputFile, protoProfileArtifact))
        .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
