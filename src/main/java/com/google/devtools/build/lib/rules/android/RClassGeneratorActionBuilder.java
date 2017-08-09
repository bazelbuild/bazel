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
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.util.OS;
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

  private AndroidAaptVersion version;

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

  public RClassGeneratorActionBuilder targetAaptVersion(AndroidAaptVersion version) {
    this.version = version;
    return this;
  }

  public RClassGeneratorActionBuilder setClassJarOut(Artifact classJarOut) {
    this.classJarOut = classJarOut;
    return this;
  }

  public void build() {
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("GENERATE_BINARY_R").add("--");

    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    inputs.addAll(
        ruleContext
            .getExecutablePrerequisite("$android_resources_busybox", Mode.HOST)
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
      // TODO(corysmith): Remove NestedSet as we are already flattening it.
      Iterable<ResourceContainer> depResources = dependencies.getResources();
      if (!Iterables.isEmpty(depResources)) {
        builder.addBeforeEach(
            "--library", Iterables.transform(depResources, chooseDepsToArg(version)));
        inputs.addTransitive(
            NestedSetBuilder.wrap(
                Order.NAIVE_LINK_ORDER,
                FluentIterable.from(depResources)
                    .transformAndConcat(chooseDepsToArtifacts(version))));
      }
    }
    builder.addExecPath("--classJarOutput", classJarOut);
    outs.add(classJarOut);

    // Create the spawn action.
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
      spawnActionBuilder.useParameterFile(ParameterFileType.SHELL_QUOTED);
    }

    ruleContext.registerAction(
        spawnActionBuilder
            .useParameterFile(ParameterFileType.SHELL_QUOTED)
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.<Artifact>copyOf(outs))
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Generating R Classes: %s", ruleContext.getLabel())
            .setMnemonic("RClassGenerator")
            .build(ruleContext));
  }

  private static Artifact chooseRTxt(ResourceContainer container, AndroidAaptVersion version) {
    return version == AndroidAaptVersion.AAPT2 ? container.getAapt2RTxt() : container.getRTxt();
  }

  private static Function<ResourceContainer, NestedSet<Artifact>> chooseDepsToArtifacts(
      final AndroidAaptVersion version) {
    return new Function<ResourceContainer, NestedSet<Artifact>>() {
      @Override
      public NestedSet<Artifact> apply(ResourceContainer container) {
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

  private static Function<ResourceContainer, String> chooseDepsToArg(
      final AndroidAaptVersion version) {
    return new Function<ResourceContainer, String>() {
      @Override
      public String apply(ResourceContainer container) {
        Artifact rTxt = chooseRTxt(container, version);
        return (rTxt != null ? rTxt.getExecPath() : "")
            + ","
            + (container.getManifest() != null ? container.getManifest().getExecPath() : "");
      }
    };
  }
}
