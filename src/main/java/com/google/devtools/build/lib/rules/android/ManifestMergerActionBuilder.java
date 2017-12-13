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

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParamFileInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.OS;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Builder for creating manifest merger actions.
 */
public class ManifestMergerActionBuilder {
  private final RuleContext ruleContext;
  private final SpawnAction.Builder spawnActionBuilder;

  private Artifact manifest;
  private Map<Artifact, Label> mergeeManifests;
  private boolean isLibrary;
  private Map<String, String> manifestValues;
  private String customPackage;
  private Artifact manifestOutput;
  private Artifact logOut;

  public ManifestMergerActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.spawnActionBuilder = new SpawnAction.Builder();
  }

  public ManifestMergerActionBuilder setManifest(Artifact manifest) {
    this.manifest = manifest;
    return this;
  }

  public ManifestMergerActionBuilder setMergeeManifests(Map<Artifact, Label> mergeeManifests) {
    this.mergeeManifests = ImmutableMap.copyOf(mergeeManifests);
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

  public ManifestMergerActionBuilder setLogOut(Artifact logOut) {
    this.logOut = logOut;
    return this;
  }

  public void build(ActionConstructionContext context) {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    ImmutableList.Builder<Artifact> outputs = ImmutableList.builder();
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("MERGE_MANIFEST").add("--");

    inputs.addAll(
        ruleContext
            .getExecutablePrerequisite("$android_resources_busybox", Mode.HOST)
            .getRunfilesSupport()
            .getRunfilesArtifactsWithoutMiddlemen());

    builder.addExecPath("--manifest", manifest);
    inputs.add(manifest);

    if (mergeeManifests != null && !mergeeManifests.isEmpty()) {
      builder.add(
          "--mergeeManifests",
          mapToDictionaryString(
              mergeeManifests, Artifact::getExecPathString, null /* valueConverter */));
      inputs.addAll(mergeeManifests.keySet());
    }

    if (isLibrary) {
      builder.add("--mergeType").add("LIBRARY");
    }

    if (manifestValues != null && !manifestValues.isEmpty()) {
      builder.add("--manifestValues", mapToDictionaryString(manifestValues));
    }

    if (customPackage != null && !customPackage.isEmpty()) {
      builder.add("--customPackage", customPackage);
    }

    builder.addExecPath("--manifestOutput", manifestOutput);
    outputs.add(manifestOutput);

    if (logOut != null) {
      builder.addExecPath("--log", logOut);
      outputs.add(logOut);
    }

    ParamFileInfo.Builder paramFileInfo = ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED);
    // Some flags (e.g. --mainData) may specify lists (or lists of lists) separated by special
    // characters (colon, semicolon, hashmark, ampersand) that don't work on Windows, and quoting
    // semantics are very complicated (more so than in Bash), so let's just always use a parameter
    // file.
    // TODO(laszlocsomor), TODO(corysmith): restructure the Android BusyBux's flags by deprecating
    // list-type and list-of-list-type flags that use such problematic separators in favor of
    // multi-value flags (to remove one level of listing) and by changing all list separators to a
    // platform-safe character (= comma).
    paramFileInfo.setUseAlways(OS.getCurrent() == OS.WINDOWS);

    ruleContext.registerAction(
        this.spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(outputs.build())
            .addCommandLine(builder.build(), paramFileInfo.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Merging manifest for %s", ruleContext.getLabel())
            .setMnemonic("ManifestMerger")
            .build(context));
  }

  private static final Function<String, String> ESCAPER =
      (String value) -> value.replace(":", "\\:").replace(",", "\\,");

  private <K, V> String mapToDictionaryString(Map<K, V> map) {
    return mapToDictionaryString(map, Functions.toStringFunction(), Functions.toStringFunction());
  }

  private <K, V> String mapToDictionaryString(Map<K, V> map,
      Function<? super K, String> keyConverter,
      Function<? super V, String> valueConverter) {
    if (keyConverter == null) {
      keyConverter = Functions.toStringFunction();
    }
    if (valueConverter == null) {
      valueConverter = Functions.toStringFunction();
    }

    StringBuilder sb = new StringBuilder();
    Iterator<Entry<K, V>> iter = map.entrySet().iterator();
    while (iter.hasNext()) {
      Entry<K, V> entry = iter.next();
      sb.append(Functions.compose(ESCAPER, keyConverter).apply(entry.getKey()));
      sb.append(':');
      sb.append(Functions.compose(ESCAPER, valueConverter).apply(entry.getValue()));
      if (iter.hasNext()) {
        sb.append(',');
      }
    }
    return sb.toString();
  }
}

