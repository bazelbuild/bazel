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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;
import com.google.devtools.build.lib.rules.android.ResourceContainerConverter.Builder.SeparatorType;
import java.util.ArrayList;
import java.util.List;

/**
 * Builder for creating $android_resource_merger action. The action merges resources and generates
 * the merged R classes for an android_library to hand off to java compilation of the library
 * sources. It also generates a merged resources zip file to pass on to the
 * $android_resource_validator action. For android_binary, see {@link
 * AndroidResourcesProcessorBuilder}.
 */
class AndroidResourceMergingActionBuilder {

  private static final ResourceContainerConverter.ToArtifacts RESOURCE_CONTAINER_TO_ARTIFACTS =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeSymbolsBin()
          .toArtifactConverter();
  private static final ResourceContainerConverter.ToArg RESOURCE_CONTAINER_TO_ARG =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeLabel()
          .includeSymbolsBin()
          .withSeparator(SeparatorType.SEMICOLON_AMPERSAND)
          .toArgConverter();

  private final RuleContext ruleContext;
  private final AndroidSdkProvider sdk;

  // Inputs
  private ResourceContainer primary;
  private ResourceDependencies dependencies;

  // Outputs
  private Artifact mergedResourcesOut;
  private Artifact classJarOut;
  private Artifact manifestOut;

  // Flags
  private String customJavaPackage;

  /** @param ruleContext The RuleContext that was used to create the SpawnAction.Builder. */
  public AndroidResourceMergingActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
  }

  /**
   * The primary resource for merging. This resource will overwrite any resource or data value in
   * the transitive closure.
   */
  public AndroidResourceMergingActionBuilder withPrimary(ResourceContainer primary) {
    this.primary = primary;
    return this;
  }

  public AndroidResourceMergingActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.dependencies = resourceDeps;
    return this;
  }

  public AndroidResourceMergingActionBuilder setMergedResourcesOut(Artifact mergedResourcesOut) {
    this.mergedResourcesOut = mergedResourcesOut;
    return this;
  }

  public AndroidResourceMergingActionBuilder setClassJarOut(Artifact classJarOut) {
    this.classJarOut = classJarOut;
    return this;
  }

  public AndroidResourceMergingActionBuilder setManifestOut(Artifact manifestOut) {
    this.manifestOut = manifestOut;
    return this;
  }

  public AndroidResourceMergingActionBuilder setJavaPackage(String customJavaPackage) {
    this.customJavaPackage = customJavaPackage;
    return this;
  }

  public ResourceContainer build(ActionConstructionContext context) {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Use a FluentIterable to avoid flattening the NestedSets
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    inputs.addAll(
        ruleContext
            .getExecutablePrerequisite("$android_resource_merger", Mode.HOST)
            .getRunfilesSupport()
            .getRunfilesArtifactsWithoutMiddlemen());

    builder.addExecPath("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    Preconditions.checkNotNull(primary);
    builder.add("--primaryData").add(RESOURCE_CONTAINER_TO_ARG.apply(primary));
    inputs.addTransitive(RESOURCE_CONTAINER_TO_ARTIFACTS.apply(primary));

    Preconditions.checkNotNull(primary.getManifest());
    builder.addExecPath("--primaryManifest", primary.getManifest());
    inputs.add(primary.getManifest());

    ResourceContainerConverter.convertDependencies(
        dependencies, builder, inputs, RESOURCE_CONTAINER_TO_ARG, RESOURCE_CONTAINER_TO_ARTIFACTS);

    Preconditions.checkNotNull(classJarOut);
    List<Artifact> outs = new ArrayList<>();
    builder.addExecPath("--classJarOutput", classJarOut);
    outs.add(classJarOut);

    if (mergedResourcesOut != null) {
      builder.addExecPath("--resourcesOutput", mergedResourcesOut);
      outs.add(mergedResourcesOut);
    }

    // For now, do manifest processing to remove placeholders that aren't handled by the legacy
    // manifest merger. Remove this once enough users migrate over to the new manifest merger.
    if (manifestOut != null) {
      builder.addExecPath("--manifestOutput", manifestOut);
      outs.add(manifestOut);
    }

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this allows android rules to generate resources outside of the java{,tests} tree.
      builder.add("--packageForR").add(customJavaPackage);
    }

    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    // Create the spawn action.
    ruleContext.registerAction(
        spawnActionBuilder
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.copyOf(outs))
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resource_merger", Mode.HOST))
            .setProgressMessage("Merging Android resources for " + ruleContext.getLabel())
            .setMnemonic("AndroidResourceMerger")
            .build(context));

    // Return the full set of processed transitive dependencies.
    return new ResourceContainer(
        primary.getLabel(),
        primary.getJavaPackage(),
        primary.getRenameManifestPackage(),
        primary.getConstantsInlined(),
        primary.getApk(),
        manifestOut != null ? manifestOut : primary.getManifest(),
        primary.getJavaSourceJar(),
        classJarOut,
        primary.getArtifacts(ResourceType.ASSETS),
        primary.getArtifacts(ResourceType.RESOURCES),
        primary.getRoots(ResourceType.ASSETS),
        primary.getRoots(ResourceType.RESOURCES),
        primary.isManifestExported(),
        primary.getRTxt(),
        primary.getSymbolsTxt());
  }
}
