// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Strings;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

/** Builder for the action that generates the R class for libraries. */
public class LibraryRGeneratorActionBuilder {
  private String javaPackage;
  private Iterable<ResourceContainer> deps = ImmutableList.<ResourceContainer>of();
  private ResourceContainer resourceContainer;
  private Artifact rJavaClassJar;

  public LibraryRGeneratorActionBuilder setJavaPackage(String javaPackage) {
    this.javaPackage = javaPackage;
    return this;
  }

  public LibraryRGeneratorActionBuilder withPrimary(ResourceContainer resourceContainer) {
    this.resourceContainer = resourceContainer;
    return this;
  }

  public LibraryRGeneratorActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.deps = resourceDeps.getResources();
    return this;
  }

  public LibraryRGeneratorActionBuilder setClassJarOut(Artifact rJavaClassJar) {
    this.rJavaClassJar = rJavaClassJar;
    return this;
  }

  public ResourceContainer build(RuleContext ruleContext) {
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);

    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    FilesToRunProvider executable =
        ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST);
    inputs.addAll(executable.getRunfilesSupport().getRunfilesArtifactsWithoutMiddlemen());

    builder.add("--tool").add("GENERATE_LIBRARY_R").add("--");

    if (!Strings.isNullOrEmpty(javaPackage)) {
      builder.add("--packageForR").add(javaPackage);
    }

    FluentIterable<ResourceContainer> symbolProviders =
        FluentIterable.from(deps).append(resourceContainer);

    if (!symbolProviders.isEmpty()) {
      ImmutableList<Artifact> symbols =
          symbolProviders.stream().map(ResourceContainer::getSymbols).collect(toImmutableList());
      builder.add("--symbols", symbols);
      inputs.addTransitive(NestedSetBuilder.wrap(Order.NAIVE_LINK_ORDER, symbols));
    }

    builder.add("--classJarOutput", rJavaClassJar);

    builder.add("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    // Create the spawn action.
    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    ruleContext.registerAction(
        spawnActionBuilder
            .useParameterFile(ParameterFileType.UNQUOTED)
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.<Artifact>of(rJavaClassJar))
            .setCommandLine(builder.build())
            .setExecutable(executable)
            .setProgressMessage("Generating Library R Classes: %s", ruleContext.getLabel())
            .setMnemonic("LibraryRClassGenerator")
            .build(ruleContext));
    return resourceContainer.toBuilder().setJavaClassJar(rJavaClassJar).build();
  }
}
