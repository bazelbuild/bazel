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
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParamFileInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import java.util.ArrayList;
import java.util.List;

/**
 * Builder for creating $android_resource_validator action. This action validates merged resources
 * of an android_library via aapt, and writes out an R.txt file (mostly to serve as a dependency for
 * the android_binary -- otherwise the merger step could have generated the R.txt).
 *
 * <p>This is split from merging, so that it can happen off of the compilation critical path.
 */
public class AndroidResourceValidatorActionBuilder {

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
  private Artifact staticLibraryOut;
  private ResourceDependencies resourceDeps;
  private Artifact aapt2SourceJarOut;
  private Artifact aapt2RTxtOut;
  private Artifact compiledSymbols;
  private Artifact apkOut;

  /** @param ruleContext The RuleContext that was used to create the SpawnAction.Builder. */
  public AndroidResourceValidatorActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
  }

  public AndroidResourceValidatorActionBuilder setStaticLibraryOut(Artifact staticLibraryOut) {
    this.staticLibraryOut = staticLibraryOut;
    return this;
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

  /** Used to add the static library from the dependencies. */
  public AndroidResourceValidatorActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.resourceDeps = resourceDeps;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setAapt2RTxtOut(Artifact aapt2RTxtOut) {
    this.aapt2RTxtOut = aapt2RTxtOut;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setAapt2SourceJarOut(Artifact aapt2SourceJarOut) {
    this.aapt2SourceJarOut = aapt2SourceJarOut;
    return this;
  }

  public ResourceContainer build(ActionConstructionContext context) {
    ResourceContainer container = primary;
    if (rTxtOut != null) {
      container = createValidateAction(container, context);
    }
    if (compiledSymbols != null) {
      container = createLinkStaticLibraryAction(container, context);
    }
    return container;
  }

  public AndroidResourceValidatorActionBuilder setCompiledSymbols(Artifact compiledSymbols) {
    this.compiledSymbols = compiledSymbols;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setApkOut(Artifact apkOut) {
    this.apkOut = apkOut;
    return this;
  }

  /**
   * This creates a static library using aapt2. It also generates a source jar and R.txt from aapt.
   *
   * <p>This allows the link action to replace the validate action for builds that use aapt2, as
   * opposed to executing both actions.
   */
  private ResourceContainer createLinkStaticLibraryAction(
      ResourceContainer validated, ActionConstructionContext context) {
    Preconditions.checkNotNull(staticLibraryOut);
    Preconditions.checkNotNull(aapt2SourceJarOut);
    Preconditions.checkNotNull(aapt2RTxtOut);
    Preconditions.checkNotNull(resourceDeps);

    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();
    ImmutableList.Builder<Artifact> inputs = ImmutableList.builder();
    ImmutableList.Builder<Artifact> outs = ImmutableList.builder();

    // Set the busybox tool.
    builder.add("--tool").add("LINK_STATIC_LIBRARY").add("--");

    builder.addExecPath("--aapt2", sdk.getAapt2().getExecutable());

    builder.add("--libraries").addExecPath(sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    builder.addExecPath("--compiled", compiledSymbols);
    inputs.add(compiledSymbols);

    builder.addExecPath("--manifest", primary.getManifest());
    inputs.add(validated.getManifest());

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this allows android rules to generate resources outside of the java{,tests} tree.
      builder.add("--packageForR", customJavaPackage);
    }

    if (!resourceDeps.getTransitiveCompiledSymbols().isEmpty()) {
      builder
          .addExecPaths(
              "--compiledDep",
              VectorArg.join(context.getConfiguration().getHostPathSeparator())
                  .each(resourceDeps.getTransitiveCompiledSymbols()));
      inputs.addAll(resourceDeps.getTransitiveCompiledSymbols());
    }

    builder.addExecPath("--sourceJarOut", aapt2SourceJarOut);
    outs.add(aapt2SourceJarOut);

    builder.addExecPath("--rTxtOut", aapt2RTxtOut);
    outs.add(aapt2RTxtOut);

    builder.addExecPath("--staticLibraryOut", staticLibraryOut);
    outs.add(staticLibraryOut);

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .addTool(sdk.getAapt2())
            .addInputs(inputs.build())
            .addOutputs(outs.build())
            .addCommandLine(
                builder.build(), ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage(
                "Linking static android resource library for %s", ruleContext.getLabel())
            .setMnemonic("AndroidResourceLink")
            .build(context));

    return validated
        .toBuilder()
        .setAapt2JavaSourceJar(aapt2SourceJarOut)
        .setAapt2RTxt(aapt2RTxtOut)
        .setStaticLibrary(staticLibraryOut)
        .build();
  }

  private ResourceContainer createValidateAction(
      ResourceContainer primary, ActionConstructionContext context) {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("VALIDATE").add("--");

    if (!Strings.isNullOrEmpty(sdk.getBuildToolsVersion())) {
      builder.add("--buildToolsVersion", sdk.getBuildToolsVersion());
    }

    builder.addExecPath("--aapt", sdk.getAapt().getExecutable());

    ImmutableList.Builder<Artifact> inputs = ImmutableList.builder();

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
      builder.add("--packageForR", customJavaPackage);
    }
    List<Artifact> outs = new ArrayList<>();
    Preconditions.checkNotNull(rTxtOut);
    builder.addExecPath("--rOutput", rTxtOut);
    outs.add(rTxtOut);

    Preconditions.checkNotNull(sourceJarOut);
    builder.addExecPath("--srcJarOutput", sourceJarOut);
    outs.add(sourceJarOut);

    if (apkOut != null) {
      builder.addExecPath("--packagePath", apkOut);
      outs.add(apkOut);
    }

    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    // Create the spawn action.
    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTool(sdk.getAapt())
            .addInputs(inputs.build())
            .addOutputs(ImmutableList.copyOf(outs))
            .addCommandLine(
                builder.build(), ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Validating Android resources for %s", ruleContext.getLabel())
            .setMnemonic("AndroidResourceValidator")
            .build(context));

    // Return the full set of validated transitive dependencies.
    return primary.toBuilder()
        .setJavaSourceJar(sourceJarOut)
        .setRTxt(rTxtOut)
        .build();
  }
}
