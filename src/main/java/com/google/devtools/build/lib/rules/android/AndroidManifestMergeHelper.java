// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import java.util.Collection;

public final class AndroidManifestMergeHelper {

  private AndroidManifestMergeHelper() {}

  public static void createMergeManifestAction(
      RuleContext ruleContext,
      Artifact merger,
      Iterable<Artifact> mergees,
      Collection<String> excludePermissions,
      Artifact mergedManifest) {
    createMergeManifestAction(
        ruleContext,
        ruleContext.getPrerequisite("$android_manifest_merge_tool", TransitionMode.HOST),
        merger,
        mergees,
        excludePermissions,
        mergedManifest);
  }

  public static void createMergeManifestAction(
      ActionConstructionContext context,
      TransitiveInfoCollection manifestMerger,
      Artifact merger,
      Iterable<Artifact> mergees,
      Collection<String> excludePermissions,
      Artifact mergedManifest) {
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();
    commandLine.addPrefixedExecPath("--merger=", merger);
    for (Artifact mergee : mergees) {
      commandLine.addPrefixedExecPath("--mergee=", mergee);
    }

    for (String excludePermission : excludePermissions) {
      commandLine.addPrefixed("--exclude_permission=", excludePermission);
    }

    commandLine.addPrefixedExecPath("--output=", mergedManifest);

    context.registerAction(
        new SpawnAction.Builder()
            .addInput(merger)
            .addInputs(mergees)
            .addOutput(mergedManifest)
            .setExecutable(manifestMerger)
            .setProgressMessage("Merging Android Manifests")
            .setMnemonic("AndroidManifestMerger")
            .addCommandLine(commandLine.build())
            .build(context));
  }
}
