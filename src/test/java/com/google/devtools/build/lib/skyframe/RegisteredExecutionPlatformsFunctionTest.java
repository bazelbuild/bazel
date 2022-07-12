// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.common.truth.IterableSubject;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.platform.ConstraintCollection;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link RegisteredExecutionPlatformsFunction} and {@link
 * RegisteredExecutionPlatformsValue}.
 */
@RunWith(JUnit4.class)
public class RegisteredExecutionPlatformsFunctionTest extends ToolchainTestCase {

  private Path moduleRoot;
  private FakeRegistry registry;

  protected EvaluationResult<RegisteredExecutionPlatformsValue>
      requestExecutionPlatformsFromSkyframe(SkyKey executionPlatformsKey)
          throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), executionPlatformsKey, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }

  protected static IterableSubject assertExecutionPlatformLabels(
      RegisteredExecutionPlatformsValue registeredExecutionPlatformsValue) {
    return assertExecutionPlatformLabels(registeredExecutionPlatformsValue, null);
  }

  protected static IterableSubject assertExecutionPlatformLabels(
      RegisteredExecutionPlatformsValue registeredExecutionPlatformsValue,
      @Nullable PackageIdentifier packageRoot) {
    assertThat(registeredExecutionPlatformsValue).isNotNull();
    ImmutableList<ConfiguredTargetKey> declaredExecutionPlatformKeys =
        registeredExecutionPlatformsValue.registeredExecutionPlatformKeys();
    List<Label> labels = collectExecutionPlatformLabels(declaredExecutionPlatformKeys, packageRoot);
    return assertThat(labels);
  }

  protected static List<Label> collectExecutionPlatformLabels(
      List<ConfiguredTargetKey> executionPlatformKeys, @Nullable PackageIdentifier packageRoot) {
    return executionPlatformKeys.stream()
        .map(ConfiguredTargetKey::getLabel)
        .filter(label -> filterLabel(packageRoot, label))
        .collect(Collectors.toList());
  }

  @Override
  protected boolean enableBzlmod() {
    return true;
  }

  @Override
  protected ImmutableList<Injected> extraPrecomputedValues() {
    try {
      moduleRoot = scratch.dir("modules");
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    registry = FakeRegistry.DEFAULT_FACTORY.newFakeRegistry(moduleRoot.getPathString());
    return ImmutableList.of(
        PrecomputedValue.injected(
            ModuleFileFunction.REGISTRIES, ImmutableList.of(registry.getUrl())),
        PrecomputedValue.injected(ModuleFileFunction.IGNORE_DEV_DEPS, false),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, CheckDirectDepsMode.WARNING));
  }

  @Before
  public void setUpForBzlmod() throws Exception {
    scratch.file("MODULE.bazel");
  }

  @Test
  public void testRegisteredExecutionPlatforms() throws Exception {
    // Request the executionPlatforms.
    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(executionPlatformsKey).isNotNull();

    RegisteredExecutionPlatformsValue value = result.get(executionPlatformsKey);
    assertThat(value.registeredExecutionPlatformKeys()).isEmpty();
  }

  @Test
  public void testRegisteredExecutionPlatforms_flagOverride() throws Exception {

    // Add an extra execution platform.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    rewriteWorkspace("register_execution_platforms('//extra:execution_platform_2')");
    useConfiguration("--extra_execution_platforms=//extra:execution_platform_1");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAtLeast(
            Label.parseAbsoluteUnchecked("//extra:execution_platform_1"),
            Label.parseAbsoluteUnchecked("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_flagOverride_multiple() throws Exception {

    // Add an extra execution platform.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    useConfiguration(
        "--extra_execution_platforms=//extra:execution_platform_1",
        "--extra_execution_platforms=//extra:execution_platform_2");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAtLeast(
            Label.parseAbsoluteUnchecked("//extra:execution_platform_1"),
            Label.parseAbsoluteUnchecked("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_targetPattern_workspace() throws Exception {

    // Add an extra execution platform.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    rewriteWorkspace("register_execution_platforms('//extra/...')");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAtLeast(
            Label.parseAbsoluteUnchecked("//extra:execution_platform_1"),
            Label.parseAbsoluteUnchecked("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_targetPattern_otherRepo() throws Exception {
    scratch.file("myrepo/WORKSPACE", "workspace(name='myrepo')");
    scratch.file("myrepo/BUILD");
    scratch.file(
        "myrepo/platforms/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");
    scratch.file(
        "myrepo/macro.bzl", "def reg(): native.register_execution_platforms('//platforms:all')");

    rewriteWorkspace(
        "local_repository(name='myrepo',path='myrepo')",
        "load('@myrepo//:macro.bzl', 'reg')",
        "reg()");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAtLeast(
            Label.parseAbsoluteUnchecked("@myrepo//platforms:execution_platform_1"),
            Label.parseAbsoluteUnchecked("@myrepo//platforms:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_targetPattern_mixed() throws Exception {

    // Add several targets, some of which are not actually platforms.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')",
        "filegroup(name = 'not_an_execution_platform')");

    rewriteWorkspace("register_execution_platforms('//extra:all')");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // There should only be two execution platforms registered from //extra.
    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(
            result.get(executionPlatformsKey), PackageIdentifier.createInMainRepo("extra"))
        .containsExactly(
            Label.parseAbsoluteUnchecked("//extra:execution_platform_1"),
            Label.parseAbsoluteUnchecked("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_targetPattern_flagOverride() throws Exception {

    // Add an extra execution platform.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    useConfiguration("--extra_execution_platforms=//extra/...");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAtLeast(
            Label.parseAbsoluteUnchecked("//extra:execution_platform_1"),
            Label.parseAbsoluteUnchecked("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_notExecutionPlatform() throws Exception {
    rewriteWorkspace("register_execution_platforms(", "    '//error:not_an_execution_platform')");
    // Have to use a rule that doesn't require a target platform, or else there will be a cycle.
    scratch.file("error/BUILD", "toolchain_type(name = 'not_an_execution_platform')");

    // Request the executionPlatforms.
    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(executionPlatformsKey)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(executionPlatformsKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//error:not_an_execution_platform");
  }

  @Test
  public void testRegisteredExecutionPlatforms_reload() throws Exception {
    scratch.file(
        "platform/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    rewriteWorkspace("register_execution_platforms('//platform:execution_platform_1')");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .contains(Label.parseAbsoluteUnchecked("//platform:execution_platform_1"));

    // Re-write the WORKSPACE.
    rewriteWorkspace("register_execution_platforms('//platform:execution_platform_2')");

    executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    result = requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .contains(Label.parseAbsoluteUnchecked("//platform:execution_platform_2"));
  }

  @Test
  public void testRegisteredExecutionPlatforms_bzlmod() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(execution_platforms_to_register=['//:plat'])",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='1.1')");
    registry
        .addModule(
            createModuleKey("B", "1.0"),
            "module(",
            "    name='B',",
            "    version='1.0',",
            "    execution_platforms_to_register=['//:plat'],",
            ")",
            "bazel_dep(name='D',version='1.0')")
        .addModule(
            createModuleKey("C", "1.1"),
            "module(",
            "    name='C',",
            "    version='1.1',",
            "    execution_platforms_to_register=['//:plat'],",
            ")",
            "bazel_dep(name='D',version='1.1')")
        // D@1.0 is not selected
        .addModule(
            createModuleKey("D", "1.0"),
            "module(",
            "    name='D',",
            "    version='1.0',",
            "    execution_platforms_to_register=['//:plat'],",
            ")")
        .addModule(
            createModuleKey("D", "1.1"),
            "module(",
            "    name='D',",
            "    version='1.1',",
            "    execution_platforms_to_register=['@E//:plat', '//:plat'],",
            ")",
            "bazel_dep(name='E',version='1.0')")
        .addModule(createModuleKey("E", "1.0"), "module(name='E', version='1.0')");
    for (String repo : ImmutableList.of("@B~1.0", "@C~1.1", "@D~1.0", "@D~1.1", "@E~1.0")) {
      scratch.file(moduleRoot.getRelative(repo).getRelative("WORKSPACE").getPathString());
      scratch.file(
          moduleRoot.getRelative(repo).getRelative("BUILD").getPathString(),
          "platform(name='plat')");
    }
    scratch.overwriteFile("BUILD", "platform(name='plat');platform(name='wsplat')");
    rewriteWorkspace("register_execution_platforms('//:wsplat')");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the execution platforms registered with bzlmod come in the BFS order and before
    // WORKSPACE registrations.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsExactly(
            Label.parseCanonical("//:plat"),
            Label.parseCanonical("@@B~1.0//:plat"),
            Label.parseCanonical("@@C~1.1//:plat"),
            Label.parseCanonical("@@E~1.0//:plat"),
            Label.parseCanonical("@@D~1.1//:plat"),
            Label.parseCanonical("//:wsplat"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatformsValue_equalsAndHashCode()
      throws ConstraintCollection.DuplicateConstraintException {
    ConfiguredTargetKey executionPlatformKey1 =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseAbsoluteUnchecked("//test:executionPlatform1"))
            .setConfigurationKey(null)
            .build();
    ConfiguredTargetKey executionPlatformKey2 =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseAbsoluteUnchecked("//test:executionPlatform2"))
            .setConfigurationKey(null)
            .build();

    new EqualsTester()
        .addEqualityGroup(
            // Two platforms registered.
            RegisteredExecutionPlatformsValue.create(
                ImmutableList.of(executionPlatformKey1, executionPlatformKey2)),
            RegisteredExecutionPlatformsValue.create(
                ImmutableList.of(executionPlatformKey1, executionPlatformKey2)))
        .addEqualityGroup(
            // A single platform registered.
            RegisteredExecutionPlatformsValue.create(ImmutableList.of(executionPlatformKey1)))
        .addEqualityGroup(
            // A single, different, platform registered.
            RegisteredExecutionPlatformsValue.create(ImmutableList.of(executionPlatformKey2)))
        .addEqualityGroup(
            // The same as the first group, but the order is different.
            RegisteredExecutionPlatformsValue.create(
                ImmutableList.of(executionPlatformKey2, executionPlatformKey1)))
        .testEquals();
  }

  /*
   * Regression test for https://github.com/bazelbuild/bazel/issues/10101.
   */
  @Test
  public void testInvalidExecutionPlatformLabelDoesntCrash() throws Exception {
    rewriteWorkspace("register_execution_platforms('//test:bad_exec_platform_label')");
    scratch.file(
        "test/BUILD", "genrule(name = 'g', srcs = [], outs = ['g.out'], cmd = 'echo hi > $@')");
    reporter.removeHandler(failFastHandler);
    assertThrows(
        "invalid registered execution platform '//test:bad_exec_platform_label': "
            + "no such target '//test:bad_exec_platform_label'",
        ViewCreationFailedException.class,
        () ->
            update(
                ImmutableList.of("//test:g"),
                /*keepGoing=*/ false,
                /*loadingPhaseThreads=*/ 1,
                /*doAnalysis=*/ true,
                eventBus));
  }
}
