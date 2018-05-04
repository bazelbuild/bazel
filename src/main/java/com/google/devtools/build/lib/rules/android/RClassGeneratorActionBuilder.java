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
import com.google.common.base.Strings;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.ResourceApk.ProcessedTransitiveData;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/** Builds up the spawn action for $android_rclass_generator. */
public class RClassGeneratorActionBuilder {

  private final RuleContext ruleContext;
  private ResourceDependencies dependencies;

  private Artifact classJarOut;

  private AndroidAaptVersion version;

  /** @param ruleContext The RuleContext that is used to create a SpawnAction.Builder. */
  public RClassGeneratorActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  public RClassGeneratorActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.dependencies = resourceDeps;
    return this;
  }

  public RClassGeneratorActionBuilder targetAaptVersion(AndroidAaptVersion version) {
    this.version = version;
    return this;
  }

  public RClassGeneratorActionBuilder setClassJarOut(Artifact classJarOut) {
    this.classJarOut = classJarOut;
    return this;
  }

  public ResourceContainer build(ResourceContainer primary) {
    build(primary.getRTxt(), ProcessedAndroidManifest.from(primary));

    return primary.toBuilder().setJavaClassJar(classJarOut).build();
  }

  public ResourceApk build(ProcessedAndroidData data) {
    build(data.getRTxt(), data.getManifest());

    return data.withValidatedResources(classJarOut);
  }

  public ResourceApk build(ProcessedTransitiveData data) {
    build(data.getRTxt(), data.getManifest());

    return ResourceApk.fromTransitiveResources(
        data.getResourceDeps(),
        data.getAssetDeps(),
        data.getManifest(),
        data.getRTxt(),
        classJarOut);
  }

  private void build(Artifact rTxt, ProcessedAndroidManifest manifest) {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("GENERATE_BINARY_R").add("--");

    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    inputs.addAll(
        ruleContext
            .getExecutablePrerequisite("$android_resources_busybox", Mode.HOST)
            .getRunfilesSupport()
            .getRunfilesArtifacts());

    List<Artifact> outs = new ArrayList<>();
    builder.addExecPath("--primaryRTxt", rTxt);
    inputs.add(rTxt);
    builder.addExecPath("--primaryManifest", manifest.getManifest());
    inputs.add(manifest.getManifest());
    if (!Strings.isNullOrEmpty(manifest.getPackage())) {
      builder.add("--packageForR", manifest.getPackage());
    }
    if (dependencies != null) {
      // TODO(corysmith): Remove NestedSet as we are already flattening it.
      Iterable<ValidatedAndroidData> depResources = dependencies.getResourceContainers();
      if (!Iterables.isEmpty(depResources)) {
        builder.addAll(
            VectorArg.addBefore("--library")
                .each(
                    ImmutableList.copyOf(
                        Iterables.transform(depResources, chooseDepsToArg(version)))));
        inputs.addTransitive(
            NestedSetBuilder.wrap(
                Order.NAIVE_LINK_ORDER,
                FluentIterable.from(depResources)
                    .transformAndConcat(chooseDepsToArtifacts(version))));
      }
    }
    builder.addExecPath("--classJarOutput", classJarOut);
    outs.add(classJarOut);
    builder.addLabel("--targetLabel", ruleContext.getLabel());

    // Create the spawn action.
    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();

    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.<Artifact>copyOf(outs))
            .addCommandLine(
                builder.build(), ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Generating R Classes: %s", ruleContext.getLabel())
            .setMnemonic("RClassGenerator")
            .build(ruleContext));
  }

  private static Artifact chooseRTxt(ValidatedAndroidData container, AndroidAaptVersion version) {
    return version == AndroidAaptVersion.AAPT2 ? container.getAapt2RTxt() : container.getRTxt();
  }

  private static Function<ValidatedAndroidData, NestedSet<Artifact>> chooseDepsToArtifacts(
      final AndroidAaptVersion version) {
    return new Function<ValidatedAndroidData, NestedSet<Artifact>>() {
      @Override
      public NestedSet<Artifact> apply(ValidatedAndroidData container) {
        NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.naiveLinkOrder();
        addIfNotNull(chooseRTxt(container, version), artifacts);
        addIfNotNull(container.getManifest(), artifacts);
        return artifacts.build();
      }

      private void addIfNotNull(@Nullable Artifact artifact, NestedSetBuilder<Artifact> artifacts) {
        if (artifact != null) {
          artifacts.add(artifact);
        }
      }
    };
  }

  private static Function<ValidatedAndroidData, String> chooseDepsToArg(
      final AndroidAaptVersion version) {
    return new Function<ValidatedAndroidData, String>() {
      @Override
      public String apply(ValidatedAndroidData container) {
        Artifact rTxt = chooseRTxt(container, version);
        return (rTxt != null ? rTxt.getExecPath() : "")
            + ","
            + (container.getManifest() != null ? container.getManifest().getExecPath() : "");
      }
    };
  }
}
