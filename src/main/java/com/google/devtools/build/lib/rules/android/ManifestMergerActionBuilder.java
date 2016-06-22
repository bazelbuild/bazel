// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Builder for creating manifest merger actions.
 */
public class ManifestMergerActionBuilder {
  private final RuleContext ruleContext;
  private final SpawnAction.Builder spawnActionBuilder;

  private Artifact manifest;
  private List<Artifact> mergeeManifests;
  private boolean isLibrary;
  private Map<String, String> manifestValues;
  private String customPackage;
  private Artifact manifestOutput;

  public ManifestMergerActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.spawnActionBuilder = new SpawnAction.Builder();
  }

  public ManifestMergerActionBuilder setManifest(Artifact manifest) {
    this.manifest = manifest;
    return this;
  }

  public ManifestMergerActionBuilder setMergeeManifests(Iterable<Artifact> mergeeManifests) {
    this.mergeeManifests = ImmutableList.copyOf(mergeeManifests);
    return this;
  }

  public ManifestMergerActionBuilder setLibrary(boolean isLibrary) {
    this.isLibrary = isLibrary;
    return this;
  }

  public ManifestMergerActionBuilder setManifestValues(Map<String, String> manifestValues) {
    this.manifestValues = manifestValues;
    return this;
  }

  public ManifestMergerActionBuilder setCustomPackage(String customPackage) {
    this.customPackage = customPackage;
    return this;
  }

  public ManifestMergerActionBuilder setManifestOutput(Artifact manifestOutput) {
    this.manifestOutput = manifestOutput;
    return this;
  }

  public void build(ActionConstructionContext context) {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    ImmutableList.Builder<Artifact> outputs = ImmutableList.builder();
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    inputs.addAll(ruleContext.getExecutablePrerequisite("$android_manifest_merger", Mode.HOST)
        .getRunfilesSupport()
        .getRunfilesArtifactsWithoutMiddlemen());

    builder.addExecPath("--manifest", manifest);
    inputs.add(manifest);

    if (mergeeManifests != null && !mergeeManifests.isEmpty()) {
      builder.addJoinExecPaths("--mergeeManifests", ":", mergeeManifests);
      inputs.addAll(mergeeManifests);
    }

    if (isLibrary) {
      builder.add("--mergeType").add("LIBRARY");
    }

    if (manifestValues != null && !manifestValues.isEmpty()) {
      builder.add("--manifestValues").add(mapToDictionaryString(manifestValues));
    }

    if (customPackage != null && !customPackage.isEmpty()) {
      builder.add("--customPackage").add(customPackage);
    }

    builder.addExecPath("--manifestOutput", manifestOutput);
    outputs.add(manifestOutput);

    ruleContext.registerAction(
        this.spawnActionBuilder
            .addTransitiveInputs(inputs.build())
            .addOutputs(outputs.build())
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_manifest_merger", Mode.HOST))
            .setProgressMessage("Merging manifest")
            .setMnemonic("ManifestMerger")
            .build(context));
  }

  private <K, V> String mapToDictionaryString(Map<K, V> map) {
    StringBuilder sb = new StringBuilder();
    Iterator<Entry<K, V>> iter = map.entrySet().iterator();
    while (iter.hasNext()) {
      Entry<K, V> entry = iter.next();
      sb.append(entry.getKey().toString().replace(":", "\\:").replace(",", "\\,"));
      sb.append(':');
      sb.append(entry.getValue().toString().replace(":", "\\:").replace(",", "\\,"));
      if (iter.hasNext()) {
        sb.append(',');
      }
    }
    return sb.toString();
  }
}

