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

import com.google.common.collect.FluentIterable;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.android.ResourceContainerConverter.Builder.SeparatorType;
import java.util.ArrayList;
import java.util.List;

/**
 * Builder for generating R classes for robolectric action.
 *
 * <p>This builder will create an action that generates r classes with internally consistent
 * resource ids for the transitive closure of dependencies that provide resources.
 */
public class RobolectricResourceSymbolsActionBuilder {

  private static final ResourceContainerConverter.ToArtifacts RESOURCE_CONTAINER_TO_ARTIFACTS =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .includeRTxt()
          .includeSymbolsBin()
          .toArtifactConverter();
  private static final ResourceContainerConverter.ToArg RESOURCE_CONTAINER_TO_ARG =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .includeRTxt()
          .includeSymbolsBin()
          .withSeparator(SeparatorType.COLON_COMMA)
          .toArgConverter();

  private Artifact classJarOut;
  private final ResourceDependencies dependencies;
  private AndroidSdkProvider sdk;

  private RobolectricResourceSymbolsActionBuilder(ResourceDependencies dependencies) {
    this.dependencies = dependencies;
  }

  public static RobolectricResourceSymbolsActionBuilder create(ResourceDependencies dependencies) {
    return new RobolectricResourceSymbolsActionBuilder(dependencies);
  }

  public RobolectricResourceSymbolsActionBuilder setJarOut(Artifact classJar) {
    this.classJarOut = classJar;
    return this;
  }

  public RobolectricResourceSymbolsActionBuilder setSdk(AndroidSdkProvider sdk) {
    this.sdk = sdk;
    return this;
  }

  public NestedSet<Artifact> buildAsClassPathEntry(RuleContext ruleContext) {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();
    // Set the busybox tool.
    builder.add("--tool").add("GENERATE_ROBOLECTRIC_R").add("--");

    List<Artifact> inputs = new ArrayList<>();

    builder.addExecPath("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    if (!Iterables.isEmpty(dependencies.getResources())) {
      builder.addJoinValues(
          "--data",
          RESOURCE_CONTAINER_TO_ARG.listSeparator(),
          dependencies.getResources(),
          RESOURCE_CONTAINER_TO_ARG);
    }

    // This flattens the nested set.
    Iterables.addAll(inputs, FluentIterable.from(dependencies.getResources())
        .transformAndConcat(RESOURCE_CONTAINER_TO_ARTIFACTS));

    builder.addExecPath("--classJarOutput", classJarOut);
    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    ruleContext.registerAction(
        spawnActionBuilder
            .useParameterFile(ParameterFileType.UNQUOTED)
            .addInputs(inputs)
            .addOutput(classJarOut)
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Generating R classes for %s", ruleContext.getLabel())
            .setMnemonic("GenerateRobolectricRClasses")
            .build(ruleContext));

    return NestedSetBuilder.<Artifact>naiveLinkOrder().add(classJarOut).build();
  }
}
