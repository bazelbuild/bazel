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
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.android.ResourceContainerConverter.Builder.SeparatorType;
import com.google.devtools.build.lib.util.OS;
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
      builder.add(
          "--data",
          VectorArg.of(dependencies.getResources())
              .joinWithDynamicString(RESOURCE_CONTAINER_TO_ARG.listSeparator())
              .mapEach(RESOURCE_CONTAINER_TO_ARG));
    }

    // This flattens the nested set.
    Iterables.addAll(inputs, FluentIterable.from(dependencies.getResources())
        .transformAndConcat(RESOURCE_CONTAINER_TO_ARTIFACTS));

    builder.addExecPath("--classJarOutput", classJarOut);
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

    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
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
