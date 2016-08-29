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
import java.util.ArrayList;
import java.util.List;

/**
 * Builder for creating $android_resource_validator action. This action validates merged resources
 * of an android_library via aapt, and writes out an R.txt file (mostly to serve as a dependency for
 * the android_binary -- otherwise the merger step could have generated the R.txt).
 *
 * <p>This is split from merging, so that it can happen off of the compilation critical path.
 */
class AndroidResourceValidatorActionBuilder {

  private final RuleContext ruleContext;
  private final AndroidSdkProvider sdk;

  // Inputs
  private ResourceContainer primary;
  private Artifact mergedResources;

  // Outputs
  private Artifact rTxtOut;
  private Artifact sourceJarOut;

  // Flags
  private String customJavaPackage;
  private boolean debug;

  /** @param ruleContext The RuleContext that was used to create the SpawnAction.Builder. */
  public AndroidResourceValidatorActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
  }

  /** The primary resource container. We mostly propagate its values, but update the R.txt. */
  public AndroidResourceValidatorActionBuilder withPrimary(ResourceContainer primary) {
    this.primary = primary;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setMergedResources(Artifact mergedResources) {
    this.mergedResources = mergedResources;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setJavaPackage(String customJavaPackage) {
    this.customJavaPackage = customJavaPackage;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setDebug(boolean debug) {
    this.debug = debug;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setRTxtOut(Artifact rTxtOut) {
    this.rTxtOut = rTxtOut;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setSourceJarOut(Artifact sourceJarOut) {
    this.sourceJarOut = sourceJarOut;
    return this;
  }

  public ResourceContainer build(ActionConstructionContext context) {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    if (!Strings.isNullOrEmpty(sdk.getBuildToolsVersion())) {
      builder.add("--buildToolsVersion").add(sdk.getBuildToolsVersion());
    }

    builder.addExecPath("--aapt", sdk.getAapt().getExecutable());

    // Use a FluentIterable to avoid flattening the NestedSets
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    inputs.addAll(
        ruleContext
            .getExecutablePrerequisite("$android_resource_validator", Mode.HOST)
            .getRunfilesSupport()
            .getRunfilesArtifactsWithoutMiddlemen());

    builder.addExecPath("--annotationJar", sdk.getAnnotationsJar());
    inputs.add(sdk.getAnnotationsJar());

    builder.addExecPath("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    Preconditions.checkNotNull(mergedResources);
    builder.addExecPath("--mergedResources", mergedResources);
    inputs.add(mergedResources);

    builder.addExecPath("--manifest", primary.getManifest());
    inputs.add(primary.getManifest());

    if (debug) {
      builder.add("--debug");
    }

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this allows android rules to generate resources outside of the java{,tests} tree.
      builder.add("--packageForR").add(customJavaPackage);
    }
    List<Artifact> outs = new ArrayList<>();
    Preconditions.checkNotNull(rTxtOut);
    builder.addExecPath("--rOutput", rTxtOut);
    outs.add(rTxtOut);

    Preconditions.checkNotNull(sourceJarOut);
    builder.addExecPath("--srcJarOutput", sourceJarOut);
    outs.add(sourceJarOut);

    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    // Create the spawn action.
    ruleContext.registerAction(
        spawnActionBuilder
            .addTool(sdk.getAapt())
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.copyOf(outs))
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resource_validator", Mode.HOST))
            .setProgressMessage("Validating Android resources for " + ruleContext.getLabel())
            .setMnemonic("AndroidResourceValidator")
            .build(context));

    // Return the full set of validated transitive dependencies.
    return new ResourceContainer(
        primary.getLabel(),
        primary.getJavaPackage(),
        primary.getRenameManifestPackage(),
        primary.getConstantsInlined(),
        primary.getApk(),
        primary.getManifest(),
        sourceJarOut,
        primary.getJavaClassJar(),
        primary.getArtifacts(ResourceType.ASSETS),
        primary.getArtifacts(ResourceType.RESOURCES),
        primary.getRoots(ResourceType.ASSETS),
        primary.getRoots(ResourceType.RESOURCES),
        primary.isManifestExported(),
        rTxtOut,
        primary.getSymbolsTxt());
  }
}
