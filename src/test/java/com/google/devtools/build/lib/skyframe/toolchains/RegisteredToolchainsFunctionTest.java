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

package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.YankedVersionsUtil;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RegisteredToolchainsFunction} and {@link RegisteredToolchainsValue}. */
@RunWith(JUnit4.class)
public class RegisteredToolchainsFunctionTest extends ToolchainTestCase {

  private Path moduleRoot;
  private FakeRegistry registry;

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
        PrecomputedValue.injected(ModuleFileFunction.MODULE_OVERRIDES, ImmutableMap.of()),
        PrecomputedValue.injected(YankedVersionsUtil.ALLOWED_YANKED_VERSIONS, ImmutableList.of()),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, CheckDirectDepsMode.WARNING),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE, BazelCompatibilityMode.ERROR),
        PrecomputedValue.injected(BazelLockFileFunction.LOCKFILE_MODE, LockfileMode.UPDATE));
  }

  @Test
  public void testRegisteredToolchains() throws Exception {
    // Request the toolchains.
    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(toolchainsKey).isNotNull();

    RegisteredToolchainsValue value = result.get(toolchainsKey);

    // Check that the number of toolchains created for this test is correct.
    assertThat(
            value.registeredToolchains().stream()
                .filter(toolchain -> toolchain.toolchainType().equals(testToolchainTypeInfo))
                .collect(Collectors.toList()))
        .hasSize(2);

    assertThat(
            value.registeredToolchains().stream()
                .anyMatch(
                    toolchain ->
                        toolchain.toolchainType().equals(testToolchainTypeInfo)
                            && toolchain.execConstraints().get(setting).equals(linuxConstraint)
                            && toolchain.targetConstraints().get(setting).equals(macConstraint)
                            && toolchain
                                .toolchainLabel()
                                .equals(
                                    Label.parseCanonicalUnchecked("//toolchain:toolchain_1_impl"))))
        .isTrue();

    assertThat(
            value.registeredToolchains().stream()
                .anyMatch(
                    toolchain ->
                        toolchain.toolchainType().equals(testToolchainTypeInfo)
                            && toolchain.execConstraints().get(setting).equals(macConstraint)
                            && toolchain.targetConstraints().get(setting).equals(linuxConstraint)
                            && toolchain
                                .toolchainLabel()
                                .equals(
                                    Label.parseCanonicalUnchecked("//toolchain:toolchain_2_impl"))))
        .isTrue();
  }

  @Test
  public void testRegisteredToolchains_flagOverride() throws Exception {

    // Add an extra toolchain.
    scratch.file(
        "extra/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain(",
        "    name = 'extra_toolchain',",
        "    toolchain_type = '//toolchain:test_toolchain',",
        "    exec_compatible_with = ['//constraints:linux'],",
        "    target_compatible_with = ['//constraints:linux'],",
        "    toolchain = ':extra_toolchain_impl')",
        "test_toolchain(",
        "  name='extra_toolchain_impl',",
        "  data = 'extra')");

    rewriteWorkspace("register_toolchains('//toolchain:toolchain_1')");
    useConfiguration("--extra_toolchains=//extra:extra_toolchain");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_toolchains flag is first in the list.
    assertToolchainLabels(result.get(toolchainsKey))
        .containsAtLeast(
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_impl"),
            Label.parseCanonicalUnchecked("//toolchain:toolchain_1_impl"))
        .inOrder();
  }

  @Test
  public void testRegisteredToolchains_flagOverride_multiple() throws Exception {

    // Add an extra toolchain.
    scratch.file(
        "extra/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain(",
        "    name = 'extra_toolchain_1',",
        "    toolchain_type = '//toolchain:test_toolchain',",
        "    exec_compatible_with = ['//constraints:linux'],",
        "    target_compatible_with = ['//constraints:linux'],",
        "    toolchain = ':extra_toolchain_impl_1')",
        "test_toolchain(",
        "  name='extra_toolchain_impl_1',",
        "  data = 'extra')",
        "toolchain(",
        "    name = 'extra_toolchain_2',",
        "    toolchain_type = '//toolchain:test_toolchain',",
        "    exec_compatible_with = ['//constraints:mac'],",
        "    target_compatible_with = ['//constraints:linux'],",
        "    toolchain = ':extra_toolchain_impl_2')",
        "test_toolchain(",
        "  name='extra_toolchain_impl_2',",
        "  data = 'extra2')");

    useConfiguration(
        "--extra_toolchains=//extra:extra_toolchain_1",
        "--extra_toolchains=//extra:extra_toolchain_2");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_toolchains flag is first in the list.
    assertToolchainLabels(result.get(toolchainsKey))
        .containsAtLeast(
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_impl_1"),
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_impl_2"),
            Label.parseCanonicalUnchecked("//toolchain:toolchain_1_impl"))
        .inOrder();
  }

  @Test
  public void testRegisteredToolchains_notToolchain() throws Exception {
    rewriteWorkspace("register_toolchains('//error:not_a_toolchain')");
    scratch.file("error/BUILD", "filegroup(name = 'not_a_toolchain')");

    // Request the toolchains.
    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(toolchainsKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "invalid registered toolchain '//error:not_a_toolchain': "
                + "target does not provide the DeclaredToolchainInfo provider");
  }

  @Test
  public void testRegisteredToolchains_targetPattern_workspace() throws Exception {
    scratch.appendFile("extra/BUILD", "filegroup(name = 'not_a_platform')");
    addToolchain(
        "extra",
        "extra_toolchain1",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "foo");
    addToolchain(
        "extra",
        "extra_toolchain2",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:mac"),
        "bar");
    addToolchain(
        "extra/more",
        "more_toolchain",
        ImmutableList.of("//constraints:mac"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    rewriteWorkspace("register_toolchains('//extra/...')");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertToolchainLabels(result.get(toolchainsKey), PackageIdentifier.createInMainRepo("extra"))
        .containsExactly(
            Label.parseCanonicalUnchecked("//extra:extra_toolchain1_impl"),
            Label.parseCanonicalUnchecked("//extra:extra_toolchain2_impl"),
            Label.parseCanonicalUnchecked("//extra/more:more_toolchain_impl"));
  }

  @Test
  public void testRegisteredToolchains_targetPattern_flagOverride() throws Exception {
    scratch.appendFile("extra/BUILD", "filegroup(name = 'not_a_platform')");
    addToolchain(
        "extra",
        "extra_toolchain1",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "foo");
    addToolchain(
        "extra",
        "extra_toolchain2",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:mac"),
        "bar");
    addToolchain(
        "extra/more",
        "more_toolchain",
        ImmutableList.of("//constraints:mac"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    useConfiguration("--extra_toolchains=//extra/...");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertToolchainLabels(result.get(toolchainsKey))
        .containsAtLeast(
            Label.parseCanonicalUnchecked("//extra:extra_toolchain1_impl"),
            Label.parseCanonicalUnchecked("//extra:extra_toolchain2_impl"),
            Label.parseCanonicalUnchecked("//extra/more:more_toolchain_impl"));
  }

  @Test
  public void testRegisteredToolchains_reload() throws Exception {
    rewriteWorkspace("register_toolchains('//toolchain:toolchain_1')");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertToolchainLabels(result.get(toolchainsKey))
        .contains(Label.parseCanonicalUnchecked("//toolchain:toolchain_1_impl"));

    // Re-write the WORKSPACE.
    rewriteWorkspace("register_toolchains('//toolchain:toolchain_2')");

    toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    result = requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertToolchainLabels(result.get(toolchainsKey))
        .contains(Label.parseCanonicalUnchecked("//toolchain:toolchain_2_impl"));
  }

  @Test
  public void testRegisteredToolchains_bzlmod() throws Exception {
    setBuildLanguageOptions("--enable_bzlmod");
    scratch.overwriteFile(
        "MODULE.bazel",
        "register_toolchains('//:tool')",
        "register_toolchains('//:dev_tool',dev_dependency=True)",
        "bazel_dep(name='bbb',version='1.0')",
        "bazel_dep(name='ccc',version='1.1')",
        "bazel_dep(name='toolchain_def',version='1.0')");
    registry
        .addModule(
            createModuleKey("bbb", "1.0"),
            "module(name='bbb',version='1.0')",
            "register_toolchains('//:tool')",
            "register_toolchains('//:dev_tool',dev_dependency=True)",
            "bazel_dep(name='ddd',version='1.0')",
            "bazel_dep(name='toolchain_def',version='1.0')")
        .addModule(
            createModuleKey("ccc", "1.1"),
            "module(name='ccc',version='1.1')",
            "register_toolchains('//:tool')",
            "register_toolchains('//:dev_tool',dev_dependency=True)",
            "bazel_dep(name='ddd',version='1.1')",
            "bazel_dep(name='toolchain_def',version='1.0')")
        // ddd@1.0 is not selected
        .addModule(
            createModuleKey("ddd", "1.0"),
            "module(name='ddd',version='1.0')",
            "register_toolchains('//:tool')",
            "register_toolchains('//:dev_tool',dev_dependency=True)",
            "bazel_dep(name='toolchain_def',version='1.0')")
        .addModule(
            createModuleKey("ddd", "1.1"),
            "module(name='ddd',version='1.1')",
            "register_toolchains('@eee//:tool', '//:tool')",
            "register_toolchains('@eee//:dev_tool',dev_dependency=True)",
            "bazel_dep(name='eee',version='1.0')",
            "bazel_dep(name='toolchain_def',version='1.0')")
        .addModule(
            createModuleKey("eee", "1.0"),
            "module(name='eee',version='1.0')",
            "bazel_dep(name='toolchain_def',version='1.0')")
        .addModule(
            createModuleKey("toolchain_def", "1.0"), "module(name='toolchain_def',version='1.0')");

    // Everyone depends on toolchain_def@1.0 for the declare_toolchain macro.
    Path toolchainDefDir = moduleRoot.getRelative("toolchain_def~1.0");
    scratch.file(toolchainDefDir.getRelative("WORKSPACE").getPathString());
    scratch.file(
        toolchainDefDir.getRelative("BUILD").getPathString(),
        "toolchain_type(name = 'test_toolchain')");
    scratch.file(
        toolchainDefDir.getRelative("toolchain_def.bzl").getPathString(),
        "def _impl(ctx):",
        "    toolchain = platform_common.ToolchainInfo(data = ctx.attr.data)",
        "    return [toolchain]",
        "test_toolchain = rule(implementation = _impl, attrs = {'data': attr.string()})",
        "def declare_toolchain(name):",
        "    native.toolchain(",
        "        name = name,",
        "        toolchain_type = Label('//:test_toolchain'),",
        "        toolchain = ':' + name + '_impl')",
        "    test_toolchain(",
        "        name = name + '_impl',",
        "        data = 'stuff')");

    // Now create the toolchains for each module.
    for (String repo : ImmutableList.of("bbb~1.0", "ccc~1.1", "ddd~1.0", "ddd~1.1", "eee~1.0")) {
      scratch.file(moduleRoot.getRelative(repo).getRelative("WORKSPACE").getPathString());
      scratch.file(
          moduleRoot.getRelative(repo).getRelative("BUILD").getPathString(),
          "load('@toolchain_def//:toolchain_def.bzl', 'declare_toolchain')",
          "declare_toolchain(name='tool')",
          "declare_toolchain(name='dev_tool')");
    }
    scratch.overwriteFile(
        "BUILD",
        "load('@toolchain_def//:toolchain_def.bzl', 'declare_toolchain')",
        "declare_toolchain(name='dev_tool')",
        "declare_toolchain(name='tool')",
        "declare_toolchain(name='wstool')");
    rewriteWorkspace("register_toolchains('//:wstool')");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfigKey);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the toolchains registered with bzlmod come in the BFS order and before WORKSPACE
    // registrations.
    assertToolchainLabels(result.get(toolchainsKey))
        .containsAtLeast(
            Label.parseCanonical("//:tool_impl"),
            Label.parseCanonical("//:dev_tool_impl"),
            Label.parseCanonical("@@bbb~1.0//:tool_impl"),
            Label.parseCanonical("@@ccc~1.1//:tool_impl"),
            Label.parseCanonical("@@eee~1.0//:tool_impl"),
            Label.parseCanonical("@@ddd~1.1//:tool_impl"),
            Label.parseCanonical("//:wstool_impl"))
        .inOrder();
  }

  @Test
  public void testRegisteredToolchainsValue_equalsAndHashCode() throws Exception {
    DeclaredToolchainInfo toolchain1 =
        DeclaredToolchainInfo.builder()
            .toolchainType(
                ToolchainTypeInfo.create(Label.parseCanonicalUnchecked("//test:toolchain")))
            .addExecConstraints(ImmutableList.of())
            .addTargetConstraints(ImmutableList.of())
            .toolchainLabel(Label.parseCanonicalUnchecked("//test/toolchain_impl_1"))
            .build();
    DeclaredToolchainInfo toolchain2 =
        DeclaredToolchainInfo.builder()
            .toolchainType(
                ToolchainTypeInfo.create(Label.parseCanonicalUnchecked("//test:toolchain")))
            .addExecConstraints(ImmutableList.of())
            .addTargetConstraints(ImmutableList.of())
            .toolchainLabel(Label.parseCanonicalUnchecked("//test/toolchain_impl_2"))
            .build();

    new EqualsTester()
        .addEqualityGroup(
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain1, toolchain2)),
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain1, toolchain2)))
        .addEqualityGroup(RegisteredToolchainsValue.create(ImmutableList.of(toolchain1)))
        .addEqualityGroup(RegisteredToolchainsValue.create(ImmutableList.of(toolchain2)))
        .addEqualityGroup(
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain2, toolchain1)))
        .testEquals();
  }
}
