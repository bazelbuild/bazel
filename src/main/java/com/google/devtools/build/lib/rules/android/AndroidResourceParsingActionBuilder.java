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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/** Builder for creating $android_resource_parser action. */
public class AndroidResourceParsingActionBuilder {

  private final RuleContext ruleContext;
  private final AndroidSdkProvider sdk;

  // These are only needed when parsing resources with data binding
  @Nullable private Artifact manifest;
  @Nullable private String javaPackage;

  private AndroidResources resources = AndroidResources.empty();
  private AndroidAssets assets = AndroidAssets.empty();

  // The symbols file is a required output
  private Artifact output;

  // Optional outputs
  @Nullable private Artifact compiledSymbols;
  @Nullable private Artifact dataBindingInfoZip;

  /** @param ruleContext The RuleContext that was used to create the SpawnAction.Builder. */
  public AndroidResourceParsingActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
  }

  /** Set the artifact location for the output protobuf. */
  public AndroidResourceParsingActionBuilder setOutput(Artifact output) {
    this.output = output;
    return this;
  }

  /** Sets the manifest. Will be ignored except when parsing resources with data binding. */
  public AndroidResourceParsingActionBuilder setManifest(@Nullable Artifact manifest) {
    this.manifest = manifest;
    return this;
  }

  /** Sets the Java package. Will be ignored except when parsing resources with data binding. */
  public AndroidResourceParsingActionBuilder setJavaPackage(@Nullable String javaPackage) {
    this.javaPackage = javaPackage;
    return this;
  }

  public AndroidResourceParsingActionBuilder setResources(AndroidResources resources) {
    this.resources = resources;
    return this;
  }

  public AndroidResourceParsingActionBuilder setAssets(AndroidAssets assets) {
    this.assets = assets;
    return this;
  }

  public AndroidResourceParsingActionBuilder setCompiledSymbolsOutput(
      @Nullable Artifact compiledSymbols) {
    this.compiledSymbols = compiledSymbols;
    return this;
  }

  public AndroidResourceParsingActionBuilder setDataBindingInfoZip(Artifact dataBindingInfoZip) {
    this.dataBindingInfoZip = dataBindingInfoZip;
    return this;
  }

  private static String convertRoots(Iterable<PathFragment> roots) {
    return Streams.stream(roots).map(Object::toString).collect(joining("#"));
  }

  private void build(ActionConstructionContext context) {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("PARSE").add("--");

    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();

    String resourceDirectories =
        convertRoots(resources.getResourceRoots()) + ":" + convertRoots(assets.getAssetRoots());
    builder.add("--primaryData", resourceDirectories);
    inputs.addTransitive(
        NestedSetBuilder.<Artifact>naiveLinkOrder()
            .addAll(assets.getAssets())
            .addAll(resources.getResources())
            .build());

    Preconditions.checkNotNull(output);
    builder.addExecPath("--output", output);

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
            .addOutputs(ImmutableList.of(output))
            .addCommandLine(builder.build(), paramFileInfo.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Parsing Android resources for %s", ruleContext.getLabel())
            .setMnemonic("AndroidResourceParser")
            .build(context));

    if (compiledSymbols != null) {
      List<Artifact> outs = new ArrayList<>();
      CustomCommandLine.Builder flatFileBuilder = new CustomCommandLine.Builder();
      flatFileBuilder
          .add("--tool")
          .add("COMPILE_LIBRARY_RESOURCES")
          .add("--")
          .addExecPath("--aapt2", sdk.getAapt2().getExecutable())
          .add("--resources", resourceDirectories)
          .addExecPath("--output", compiledSymbols);
      inputs.add(sdk.getAapt2().getExecutable());
      outs.add(compiledSymbols);

      // The databinding needs to be processed before compilation, so the stripping happens here.
      if (dataBindingInfoZip != null) {
        flatFileBuilder.addExecPath("--manifest", manifest);
        inputs.add(manifest);
        if (!Strings.isNullOrEmpty(javaPackage)) {
          flatFileBuilder.add("--packagePath", javaPackage);
        }
        flatFileBuilder.addExecPath("--dataBindingInfoOut", dataBindingInfoZip);
        outs.add(dataBindingInfoZip);
      }
      // Create the spawn action.
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .useDefaultShellEnvironment()
              .addTransitiveInputs(inputs.build())
              .addOutputs(ImmutableList.copyOf(outs))
              .addCommandLine(flatFileBuilder.build(), paramFileInfo.build())
              .setExecutable(
                  ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
              .setProgressMessage("Compiling Android resources for %s", ruleContext.getLabel())
              .setMnemonic("AndroidResourceCompiler")
              .build(context));
    }
  }

  /**
   * Builds and registers the action, and returns a copy of the passed resources with artifacts for
   * parsed and compiled information.
   */
  public ParsedAndroidResources build(
      AndroidResources androidResources, StampedAndroidManifest manifest) {
    if (dataBindingInfoZip != null) {
      // Manifest information is needed for data binding
      setManifest(manifest.getManifest());
      setJavaPackage(manifest.getPackage());
    }

    setResources(androidResources);
    build(ruleContext);

    return ParsedAndroidResources.of(
        androidResources, output, compiledSymbols, ruleContext.getLabel(), manifest);
  }

  public ParsedAndroidAssets build(AndroidAssets assets) {
    setAssets(assets);
    build(ruleContext);

    return ParsedAndroidAssets.of(assets, output, ruleContext.getLabel());
  }

  /**
   * Builds and registers the action, and updates the given resourceContainer with the output
   * symbols.
   */
  public ResourceContainer buildAndUpdate(
      RuleContext ruleContext, ResourceContainer resourceContainer) {
    build(ruleContext);

    ResourceContainer.Builder builder =
        resourceContainer
            .toBuilder()
            .setSymbols(output)
            .setAndroidAssets(assets)
            .setAndroidResources(resources);

    if (compiledSymbols != null) {
      builder.setCompiledSymbols(compiledSymbols);
    }

    return builder.build();
  }
}
