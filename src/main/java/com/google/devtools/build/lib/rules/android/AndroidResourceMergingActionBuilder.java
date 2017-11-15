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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParamFileInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
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

  private static final ResourceContainerConverter.ToArg RESOURCE_CONTAINER_TO_ARG =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeLabel()
          .includeSymbolsBin()
          .withSeparator(SeparatorType.SEMICOLON_AMPERSAND)
          .toArgConverter();

  private static final ResourceContainerConverter.ToArg RESOURCE_CONTAINER_TO_ARG_FOR_COMPILED =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeLabel()
          .includeCompiledSymbols()
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
  private boolean useCompiledMerge;

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

  public AndroidResourceMergingActionBuilder setUseCompiledMerge(
      boolean useCompiledMerge) {
    this.useCompiledMerge = useCompiledMerge;
    return this;
  }

  private NestedSetBuilder<Artifact> createInputsForBuilder(CustomCommandLine.Builder builder) {
    // Use a FluentIterable to avoid flattening the NestedSets
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();

    builder.addExecPath("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    Preconditions.checkNotNull(primary.getManifest());
    builder.addExecPath("--primaryManifest", primary.getManifest());
    inputs.add(primary.getManifest());

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this allows android rules to generate resources outside of the java{,tests} tree.
      builder.add("--packageForR", customJavaPackage);
    }

    if (throwOnResourceConflict) {
      builder.add("--throwOnResourceConflict");
    }

    return inputs;
  }

  private void buildCompiledResourceMergingAction(
      CustomCommandLine.Builder builder,
      List<Artifact> outputs,
      ActionConstructionContext context) {
    NestedSetBuilder<Artifact> inputs = createInputsForBuilder(builder);

    Preconditions.checkNotNull(primary);
    builder.add("--primaryData",
        RESOURCE_CONTAINER_TO_ARG_FOR_COMPILED.apply(primary));
    inputs.addAll(primary.getArtifacts());
    inputs.add(primary.getCompiledSymbols());

    if (dependencies != null) {
      ResourceContainerConverter.addToCommandLine(dependencies, builder,
          RESOURCE_CONTAINER_TO_ARG_FOR_COMPILED);
      inputs.addTransitive(dependencies.getTransitiveResources());
      inputs.addTransitive(dependencies.getTransitiveAssets());
      inputs.addTransitive(dependencies.getTransitiveCompiledSymbols());
    }

    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    ParamFileInfo.Builder compiledParamFileInfo = ParamFileInfo.builder(ParameterFileType.UNQUOTED);
    compiledParamFileInfo.setUseAlways(OS.getCurrent() == OS.WINDOWS);
    // Create the spawn action.
    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.copyOf(outputs))
            .addCommandLine(builder.build(), compiledParamFileInfo.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Merging compiled Android resources for %s",
                ruleContext.getLabel())
            .setMnemonic("AndroidCompiledResourceMerger")
            .build(context));
  }


  private void buildParsedResourceMergingAction(
      CustomCommandLine.Builder builder,
      List<Artifact> outputs,
      ActionConstructionContext context) {
    NestedSetBuilder<Artifact> inputs = createInputsForBuilder(builder);

    Preconditions.checkNotNull(primary);
    builder.add("--primaryData", RESOURCE_CONTAINER_TO_ARG.apply(primary));
    inputs.addAll(primary.getArtifacts());
    inputs.add(primary.getSymbols());

    if (dependencies != null) {
      ResourceContainerConverter.addToCommandLine(dependencies, builder, RESOURCE_CONTAINER_TO_ARG);
      inputs.addTransitive(dependencies.getTransitiveResources());
      inputs.addTransitive(dependencies.getTransitiveAssets());
      inputs.addTransitive(dependencies.getTransitiveSymbolsBin());
    }

    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
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

    // Create the spawn action.
    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.copyOf(outputs))
            .addCommandLine(builder.build(), paramFileInfo.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Merging Android resources for %s", ruleContext.getLabel())
            .setMnemonic("AndroidResourceMerger")
            .build(context));
  }

  public ResourceContainer build(ActionConstructionContext context) {
    CustomCommandLine.Builder parsedMergeBuilder =
        new CustomCommandLine.Builder().add("--tool").add("MERGE").add("--");
    CustomCommandLine.Builder compiledMergeBuilder =
        new CustomCommandLine.Builder().add("--tool").add("MERGE_COMPILED").add("--");
    List<Artifact> parsedMergeOutputs = new ArrayList<>();
    List<Artifact> compiledMergeOutputs = new ArrayList<>();

    if (mergedResourcesOut != null) {
      parsedMergeBuilder.addExecPath("--resourcesOutput", mergedResourcesOut);
      parsedMergeOutputs.add(mergedResourcesOut);
    }

    // TODO(corysmith): Move the data binding parsing out of the merging pass to enable faster
    // aapt2 builds.
    if (dataBindingInfoZip != null) {
      parsedMergeBuilder.addExecPath("--dataBindingInfoOut", dataBindingInfoZip);
      parsedMergeOutputs.add(dataBindingInfoZip);
    }

    CustomCommandLine.Builder jarAndManifestBuilder =
        useCompiledMerge
            ? compiledMergeBuilder
            : parsedMergeBuilder;
    List<Artifact> jarAndManifestOutputs =
        useCompiledMerge
            ? compiledMergeOutputs
            : parsedMergeOutputs;

    if (classJarOut != null) {
      jarAndManifestBuilder.addExecPath("--classJarOutput", classJarOut);
      jarAndManifestOutputs.add(classJarOut);
    }

    // For now, do manifest processing to remove placeholders that aren't handled by the legacy
    // manifest merger. Remove this once enough users migrate over to the new manifest merger.
    if (manifestOut != null) {
      jarAndManifestBuilder.addExecPath("--manifestOutput", manifestOut);
      jarAndManifestOutputs.add(manifestOut);
    }

    if (!compiledMergeOutputs.isEmpty()) {
      buildCompiledResourceMergingAction(compiledMergeBuilder, compiledMergeOutputs, context);
    }

    if (!parsedMergeOutputs.isEmpty()) {
      buildParsedResourceMergingAction(parsedMergeBuilder, parsedMergeOutputs, context);
    }

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
