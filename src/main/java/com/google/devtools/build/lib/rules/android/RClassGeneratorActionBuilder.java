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
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Builds up the spawn action for $android_rclass_generator.
 */
public class RClassGeneratorActionBuilder {

  private final RuleContext ruleContext;
  private ResourceContainer primary;
  private ResourceDependencies dependencies;

  private Artifact classJarOut;

  private static final ContainerToArg DEPS_TO_ARG = new ContainerToArg();
  private static final ContainerToArtifacts DEPS_TO_ARTIFACTS = new ContainerToArtifacts();

  /**
   * @param ruleContext The RuleContext that is used to create a SpawnAction.Builder.
   */
  public RClassGeneratorActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  public RClassGeneratorActionBuilder withPrimary(ResourceContainer primary) {
    this.primary = primary;
    return this;
  }

  public RClassGeneratorActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.dependencies = resourceDeps;
    return this;
  }

  public RClassGeneratorActionBuilder setClassJarOut(Artifact classJarOut) {
    this.classJarOut = classJarOut;
    return this;
  }

  private static class ContainerToArg implements Function<ResourceContainer, String> {

    @Override
    public String apply(ResourceContainer container) {
      return (container.getRTxt() != null ? container.getRTxt().getExecPath() : "")
          + ":"
          + (container.getManifest() != null ? container.getManifest().getExecPath() : "");
    }
  }

  private static class ContainerToArtifacts implements
      Function<ResourceContainer, NestedSet<Artifact>> {

    @Override
    public NestedSet<Artifact> apply(ResourceContainer container) {
      NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.naiveLinkOrder();
      addIfNotNull(container.getRTxt(), artifacts);
      addIfNotNull(container.getManifest(), artifacts);
      return artifacts.build();
    }

    private void addIfNotNull(@Nullable Artifact artifact, NestedSetBuilder<Artifact> artifacts) {
      if (artifact != null) {
        artifacts.add(artifact);
      }
    }

  }

  public void build() {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    inputs.addAll(ruleContext.getExecutablePrerequisite("$android_rclass_generator", Mode.HOST)
        .getRunfilesSupport()
        .getRunfilesArtifactsWithoutMiddlemen());

    List<Artifact> outs = new ArrayList<>();
    if (primary.getRTxt() != null) {
      builder.addExecPath("--primaryRTxt", primary.getRTxt());
      inputs.add(primary.getRTxt());
    }
    if (primary.getManifest() != null) {
      builder.addExecPath("--primaryManifest", primary.getManifest());
      inputs.add(primary.getManifest());
    }
    if (!Strings.isNullOrEmpty(primary.getJavaPackage())) {
      builder.add("--packageForR").add(primary.getJavaPackage());
    }
    if (dependencies != null) {
      Iterable<ResourceContainer> depResources = dependencies.getResources();
      if (depResources.iterator().hasNext()) {
        builder.addJoinStrings("--libraries", ",", Iterables.transform(depResources, DEPS_TO_ARG));
        inputs.addTransitive(NestedSetBuilder.wrap(
            Order.NAIVE_LINK_ORDER,
            FluentIterable.from(depResources)
                .transformAndConcat(DEPS_TO_ARTIFACTS)));
      }
    }
    builder.addExecPath("--classJarOutput", classJarOut);
    outs.add(classJarOut);

    // Create the spawn action.
    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    ruleContext.registerAction(
        spawnActionBuilder
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.<Artifact>copyOf(outs))
            .useParameterFile(ParameterFileType.SHELL_QUOTED)
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_rclass_generator", Mode.HOST))
            .setProgressMessage("Generating R Classes: " + ruleContext.getLabel())
            .setMnemonic("RClassGenerator")
            .build(ruleContext));
  }
}
