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
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.android.ResourceContainerConverter.Builder.SeparatorType;
import com.google.devtools.build.lib.util.OS;
import java.util.ArrayList;
import java.util.List;

/**
 * Builder for creating $android_resource_merger action. The action merges resources and generates
 * the merged R classes for an android_library to hand off to java compilation of the library
 * sources. It also generates a merged resources zip file to pass on to the
 * $android_resource_validator action. For android_binary, see {@link
 * AndroidResourcesProcessorBuilder}.
 */
public class AndroidResourceMergingActionBuilder {

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
  private Artifact dataBindingInfoZip;

  // Flags
  private String customJavaPackage;
  private boolean throwOnResourceConflict;

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

  /**
   * The output zip for resource-processed data binding expressions (i.e. a zip of .xml files).
   * If null, data binding processing is skipped (and data binding expressions aren't allowed in
   * layout resources).
   */
  public AndroidResourceMergingActionBuilder setDataBindingInfoZip(Artifact zip) {
    this.dataBindingInfoZip = zip;
    return this;
  }

  public AndroidResourceMergingActionBuilder setJavaPackage(String customJavaPackage) {
    this.customJavaPackage = customJavaPackage;
    return this;
  }

  public AndroidResourceMergingActionBuilder setThrowOnResourceConflict(
      boolean throwOnResourceConflict) {
    this.throwOnResourceConflict = throwOnResourceConflict;
    return this;
  }

  public ResourceContainer build(ActionConstructionContext context) {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("MERGE").add("--");

    // Use a FluentIterable to avoid flattening the NestedSets
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();

    builder.add("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    Preconditions.checkNotNull(primary);
    builder.add("--primaryData").add(RESOURCE_CONTAINER_TO_ARG.apply(primary));
    inputs.addTransitive(RESOURCE_CONTAINER_TO_ARTIFACTS.apply(primary));

    Preconditions.checkNotNull(primary.getManifest());
    builder.add("--primaryManifest", primary.getManifest());
    inputs.add(primary.getManifest());

    ResourceContainerConverter.convertDependencies(
        dependencies, builder, inputs, RESOURCE_CONTAINER_TO_ARG, RESOURCE_CONTAINER_TO_ARTIFACTS);

    List<Artifact> outs = new ArrayList<>();
    if (classJarOut != null) {
      builder.add("--classJarOutput", classJarOut);
      outs.add(classJarOut);
    }

    if (mergedResourcesOut != null) {
      builder.add("--resourcesOutput", mergedResourcesOut);
      outs.add(mergedResourcesOut);
    }

    // For now, do manifest processing to remove placeholders that aren't handled by the legacy
    // manifest merger. Remove this once enough users migrate over to the new manifest merger.
    if (manifestOut != null) {
      builder.add("--manifestOutput", manifestOut);
      outs.add(manifestOut);
    }

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this allows android rules to generate resources outside of the java{,tests} tree.
      builder.add("--packageForR").add(customJavaPackage);
    }

    // TODO(corysmith): Move the data binding parsing out of the merging pass to enable faster
    // aapt2 builds.
    if (dataBindingInfoZip != null) {
      builder.add("--dataBindingInfoOut", dataBindingInfoZip);
      outs.add(dataBindingInfoZip);
    }

    if (throwOnResourceConflict) {
      builder.add("--throwOnResourceConflict");
    }

    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();

    if (OS.getCurrent() == OS.WINDOWS) {
      // Some flags (e.g. --mainData) may specify lists (or lists of lists) separated by special
      // characters (colon, semicolon, hashmark, ampersand) that don't work on Windows, and quoting
      // semantics are very complicated (more so than in Bash), so let's just always use a parameter
      // file.
      // TODO(laszlocsomor), TODO(corysmith): restructure the Android BusyBux's flags by deprecating
      // list-type and list-of-list-type flags that use such problematic separators in favor of
      // multi-value flags (to remove one level of listing) and by changing all list separators to a
      // platform-safe character (= comma).
      spawnActionBuilder.alwaysUseParameterFile(ParameterFileType.UNQUOTED);
    } else {
      spawnActionBuilder.useParameterFile(ParameterFileType.UNQUOTED);
    }

    // Create the spawn action.
    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.copyOf(outs))
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Merging Android resources for %s", ruleContext.getLabel())
            .setMnemonic("AndroidResourceMerger")
            .build(context));

    // Return the full set of processed transitive dependencies.
    ResourceContainer.Builder result = primary.toBuilder();
    if (classJarOut != null) {
      // ensure the classJar is propagated if it exists. Otherwise, AndroidCommon tries to make it.
      // TODO(corysmith): Centralize the class jar generation.
      result.setJavaClassJar(classJarOut);
    }
    if (manifestOut != null) {
      result.setManifest(manifestOut);
    }
    return result.build();
  }
}
