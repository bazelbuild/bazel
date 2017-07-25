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

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.android.deployinfo.AndroidDeployInfoOuterClass.AndroidDeployInfo;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.Preconditions;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** Common methods shared between Android related {@link BuildViewTestCase}s. */
public abstract class AndroidBuildViewTestCase extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    return builder
        // TODO(b/35097211): Remove this once the new testing rules are released.
        .addRuleDefinition(new AndroidDeviceScriptFixtureRule())
        .addRuleDefinition(new AndroidHostServiceFixtureRule())
        .addRuleDefinition(new AndroidInstrumentationRule())
        .addRuleDefinition(new AndroidInstrumentationTestRule())
        .build();
  }

  protected Iterable<Artifact> getNativeLibrariesInApk(ConfiguredTarget target) {
    SpawnAction compressedUnsignedApkaction = getCompressedUnsignedApkAction(target);
    ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (Artifact output : compressedUnsignedApkaction.getInputs()) {
      if (!output.getExecPathString().endsWith(".so")) {
        continue;
      }

      result.add(output);
    }

    return result.build();
  }

  protected Label getGeneratingLabelForArtifact(Artifact artifact) {
    Action generatingAction = getGeneratingAction(artifact);
    return generatingAction != null
        ? getGeneratingAction(artifact).getOwner().getLabel()
        : null;
  }

  protected void assertNativeLibrariesCopiedNotLinked(
      ConfiguredTarget target, String... expectedLibNames) {
    Iterable<Artifact> copiedLibs = getNativeLibrariesInApk(target);
    for (Artifact copiedLib : copiedLibs) {
      assertWithMessage("Native libraries were linked to produce " + copiedLib)
          .that(getGeneratingLabelForArtifact(copiedLib))
          .isNotEqualTo(target.getLabel());
    }
    assertThat(artifactsToStrings(copiedLibs))
        .containsAllIn(ImmutableSet.copyOf(Arrays.asList(expectedLibNames)));
  }

  protected String flagValue(String flag, List<String> args) {
    assertThat(args).contains(flag);
    return args.get(args.indexOf(flag) + 1);
  }

  /**
   * The unsigned APK is created in two actions. The first action adds everything that needs to be
   * unconditionally compressed in the APK. The second action adds everything else, preserving their
   * compression.
   */
  protected Artifact getCompressedUnsignedApk(ConfiguredTarget target) {
    return artifactByPath(
        actionsTestUtil().artifactClosureOf(getFinalUnsignedApk(target)),
        "_unsigned.apk",
        "_unsigned.apk");
  }

  protected SpawnAction getCompressedUnsignedApkAction(ConfiguredTarget target) {
    return getGeneratingSpawnAction(getCompressedUnsignedApk(target));
  }

  protected Artifact getFinalUnsignedApk(ConfiguredTarget target) {
    return getFirstArtifactEndingWith(
        target.getProvider(FileProvider.class).getFilesToBuild(), "_unsigned.apk");
  }

  protected SpawnAction getFinalUnsignedApkAction(ConfiguredTarget target) {
    return getGeneratingSpawnAction(getFinalUnsignedApk(target));
  }

  protected Artifact getResourceApk(ConfiguredTarget target) {
    Artifact resourceApk =
        getFirstArtifactEndingWith(
            getGeneratingAction(getFinalUnsignedApk(target)).getInputs(), ".ap_");
    assertThat(resourceApk).isNotNull();
    return resourceApk;
  }

  protected void assertProguardUsed(ConfiguredTarget binary) {
    assertWithMessage("proguard.jar is not in the rule output")
        .that(
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "_proguard.jar"))
        .isNotNull();
  }

  protected List<String> resourceArguments(ResourceContainer resource) {
    return resourceGeneratingAction(resource).getArguments();
  }

  protected SpawnAction resourceGeneratingAction(ResourceContainer resource) {
    return getGeneratingSpawnAction(resource.getApk());
  }

  protected static ResourceContainer getResourceContainer(ConfiguredTarget target) {
    return getResourceContainer(target, /* transitive= */ false);
  }

  protected static ResourceContainer getResourceContainer(
      ConfiguredTarget target, boolean transitive) {

    Preconditions.checkNotNull(target);
    final AndroidResourcesProvider provider = target.getProvider(AndroidResourcesProvider.class);
    assertThat(provider).named("No android resources exported from the target.").isNotNull();
    return getOnlyElement(
        transitive
            ? provider.getTransitiveAndroidResources()
            : provider.getDirectAndroidResources());
  }

  protected ActionAnalysisMetadata getResourceClassJarAction(final ConfiguredTarget target) {
    JavaRuleOutputJarsProvider jarProvider = target.getProvider(JavaRuleOutputJarsProvider.class);
    assertThat(jarProvider).isNotNull();
    return getGeneratingAction(
        Iterables.find(
                jarProvider.getOutputJars(),
                outputJar -> {
                  assertThat(outputJar).isNotNull();
                  assertThat(outputJar.getClassJar()).isNotNull();
                  return outputJar
                      .getClassJar()
                      .getFilename()
                      .equals(target.getTarget().getName() + "_resources.jar");
                })
            .getClassJar());
  }

  // android resources related tests
  protected void assertPrimaryResourceDirs(List<String> expectedPaths, List<String> actualArgs) {
    assertThat(actualArgs).contains("--primaryData");
    String actualFlagValue = actualArgs.get(actualArgs.indexOf("--primaryData") + 1);
    List<String> actualPaths = null;
    if (actualFlagValue.matches("[^;]*;[^;]*;.*")) {
      actualPaths = Arrays.asList(actualFlagValue.split(";")[0].split("#"));

    } else if (actualFlagValue.matches("[^:]*:[^:]*:.*")) {
      actualPaths = Arrays.asList(actualFlagValue.split(":")[0].split("#"));
    } else {
      fail(String.format("Failed to parse --primaryData: %s", actualFlagValue));
    }
    assertThat(actualPaths).containsAllIn(expectedPaths);
  }

  protected List<String> getDirectDependentResourceDirs(List<String> actualArgs) {
    assertThat(actualArgs).contains("--directData");
    String actualFlagValue = actualArgs.get(actualArgs.indexOf("--directData") + 1);
    return getDependentResourceDirs(actualFlagValue);
  }

  protected List<String> getTransitiveDependentResourceDirs(List<String> actualArgs) {
    assertThat(actualArgs).contains("--data");
    String actualFlagValue = actualArgs.get(actualArgs.indexOf("--data") + 1);
    return getDependentResourceDirs(actualFlagValue);
  }

  private static List<String> getDependentResourceDirs(String actualFlagValue) {
    String separator = null;
    if (actualFlagValue.matches("[^;]*;[^;]*;[^;]*;.*")) {
      separator = ";";
    } else if (actualFlagValue.matches("[^:]*:[^:]*:[^:]*:.*")) {
      separator = ":";
    } else {
      fail(String.format("Failed to parse flag: %s", actualFlagValue));
    }
    ImmutableList.Builder<String> actualPaths = ImmutableList.builder();
    for (String resourceDependency :  actualFlagValue.split(",")) {
      actualPaths.add(resourceDependency.split(separator)[0].split("#"));
    }
    return actualPaths.build();
  }

  protected String execPathEndingWith(Iterable<Artifact> inputs, String suffix) {
    return getFirstArtifactEndingWith(inputs, suffix).getExecPathString();
  }

  @Nullable
  protected AndroidDeployInfo getAndroidDeployInfo(Artifact artifact) throws IOException {
    Action generatingAction = getGeneratingAction(artifact);
    if (generatingAction instanceof AndroidDeployInfoAction) {
      AndroidDeployInfoAction writeAction = (AndroidDeployInfoAction) generatingAction;
      return writeAction.getDeployInfo();
    }
    return null;
  }

  protected List<String> getProcessorNames(String outputTarget) throws Exception {
    OutputFileConfiguredTarget out = (OutputFileConfiguredTarget)
        getFileConfiguredTarget(outputTarget);
    JavaCompileAction compileAction = (JavaCompileAction) getGeneratingAction(out.getArtifact());
    return compileAction.getProcessorNames();
  }

  // Returns an artifact that will be generated when a rule has resources.
  protected static Artifact getResourceArtifact(ConfiguredTarget target) {
    // the last provider is the provider from the target.
    return Iterables.getLast(
            target.getProvider(AndroidResourcesProvider.class).getDirectAndroidResources())
        .getJavaClassJar();
  }

  protected static Set<Artifact> getNonToolInputs(Action action) {
    return Sets.difference(
        ImmutableSet.copyOf(action.getInputs()), ImmutableSet.copyOf(action.getTools()));
  }
}
