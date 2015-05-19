// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public final class AndroidManifestMergeHelper {

  private AndroidManifestMergeHelper() {}

  public static void createMergeManifestAction(RuleContext ruleContext,
      Artifact merger, Iterable<Artifact> mergees,
      Collection<String> excludePermissions, Artifact mergedManifest) {
    List<String> args = new ArrayList<>();
    args.add("--merger=" + merger.getExecPathString());

    for (Artifact mergee : mergees) {
      args.add("--mergee=" + mergee.getExecPathString());
    }

    for (String excludePermission : excludePermissions) {
      args.add("--exclude_permission=" + excludePermission);
    }

    args.add("--output=" + mergedManifest.getExecPathString());

    ruleContext.registerAction(new SpawnAction.Builder()
        .addInput(merger)
        .addInputs(mergees)
        .addOutput(mergedManifest)
        .setExecutable(ruleContext.getPrerequisite("$android_manifest_merge_tool", Mode.HOST))
        .addArguments(args)
        .setProgressMessage("Merging Android Manifests")
        .setMnemonic("AndroidManifestMerger")
        .build(ruleContext));
  }
}

