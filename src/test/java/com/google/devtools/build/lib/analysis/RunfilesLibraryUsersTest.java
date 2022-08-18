// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RunfilesProvider.RepositoryNameAndMapping;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RunfilesLibraryUsersTest extends BuildViewTestCase {

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
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, CheckDirectDepsMode.WARNING));
  }

  @Test
  public void testRunfilesLibraryUsersCollectedInRunfilesProvider() throws Exception {
    setBuildLanguageOptions("--enable_bzlmod");
    scratch.file("MODULE.bazel",
        "module()",
        "bazel_dep(name='starlark_non_user',version='1.0')");
    registry
        .addModule(
            createModuleKey("utils", "1.0"),
            "module(name='utils',version='1.0')",
            "bazel_dep(name='bar_user',version='1.0')")
        .addModule(
            createModuleKey("cc_user", "1.0"),
            "module(name='cc_user',version='1.0')",
            "bazel_dep(name='utils',version='1.0')")
        .addModule(
            createModuleKey("cc_non_user", "1.0"),
            "module(name='cc_non_user',version='1.0')",
            "bazel_dep(name='cc_user',version='1.0')")
        .addModule(
            createModuleKey("java_user", "1.0"),
            "module(name='java_user',version='1.0')",
            "bazel_dep(name='utils',version='1.0')")
        .addModule(
            createModuleKey("java_non_user", "1.0"),
            "module(name='java_non_user',version='1.0')",
            "bazel_dep(name='java_user',version='1.0')")
        .addModule(
            createModuleKey("bar_user", "1.0"),
            "module(name='bar_user',version='1.0')",
            "bazel_dep(name='utils',version='1.0')",
            "register_toolchains('//:bar_toolchain')")
        .addModule(
            createModuleKey("starlark_user", "1.0"),
            "module(name='starlark_user',version='1.0')",
            "bazel_dep(name='utils',version='1.0')")
        .addModule(
            createModuleKey("starlark_tool_user", "1.0"),
            "module(name='starlark_tool_user',version='1.0')",
            "bazel_dep(name='utils',version='1.0')")
        .addModule(
            createModuleKey("starlark_non_user", "1.0"),
            "module(name='starlark_non_user',version='1.0')",
            "bazel_dep(name='utils',version='1.0')",
            "bazel_dep(name='starlark_user',version='1.0')",
            "bazel_dep(name='starlark_tool_user',version='1.0')",
            "bazel_dep(name='java_non_user',version='1.0')",
            "bazel_dep(name='cc_non_user',version='1.0')",
            "bazel_dep(name='bar_user',version='1.0')");

    scratch.file(moduleRoot.getRelative("utils~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(moduleRoot.getRelative("utils~1.0").getRelative("BUILD").getPathString(),
        "load(':utils.bzl', 'runfiles_lib')",
        "runfiles_lib(name='runfiles_lib',visibility=['//visibility:public'])");
    scratch.file(
        moduleRoot.getRelative("utils~1.0").getRelative("utils.bzl").getPathString(),
        "def _runfiles_lib_impl(ctx):",
        "  return [",
        "    CcInfo(),",
        "    RunfilesLibraryInfo(),",
        "  ]",
        "runfiles_lib = rule(_runfiles_lib_impl)",
        "",
        "def _starlark_rule_impl(ctx):",
        // Explicitly do not merge in runfiles, just create a new runfiles object.
        "  return [DefaultInfo(runfiles = ctx.runfiles())]",
        "starlark_rule = rule(",
        "  implementation = _starlark_rule_impl,",
        "  attrs = {",
        "    'deps': attr.label_list(),",
        "    'tool_deps': attr.label_list(cfg = 'exec'),",
        "  },",
        "  toolchains = ['@bar_user//:toolchain_type'],",
        ")");

    scratch.file(moduleRoot.getRelative("cc_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("cc_user~1.0").getRelative("BUILD").getPathString(),
        "cc_library(",
        "  name = 'cc_user',",
        "  deps = ['@utils//:runfiles_lib'],",
        "  visibility = ['//visibility:public'],",
        ")");

    scratch.file(
        moduleRoot.getRelative("cc_non_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("cc_non_user~1.0").getRelative("BUILD").getPathString(),
        "cc_library(",
        "  name = 'cc_non_user',",
        "  deps = ['@cc_user//:cc_user'],",
        ")");

    scratch.file(moduleRoot.getRelative("java_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("java_user~1.0").getRelative("BUILD").getPathString(),
        "java_library(",
        "  name = 'java_user',",
        "  runtime_deps = ['@utils//:runfiles_lib'],",
        "  visibility = ['//visibility:public'],",
        ")");

    scratch.file(
        moduleRoot.getRelative("java_non_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("java_non_user~1.0").getRelative("BUILD").getPathString(),
        "java_library(",
        "  name = 'java_non_user',",
        "  srcs = ['Lib.java'],",
        "  deps = ['@java_user//:java_user'],",
        ")");

    scratch.file(
        moduleRoot.getRelative("bar_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("bar_user~1.0").getRelative("BUILD").getPathString(),
        "load(':toolchain.bzl', 'bar_toolchain')",
        "java_library(",
        "  name = 'bar_user',",
        "  srcs = ['Runtime.java'],",
        "  deps = ['@utils//:runfiles_lib'],",
        ")",
        "bar_toolchain(",
        "  name = 'bar',",
        "  runtime = ':bar_user',",
        ")",
        "toolchain_type(name = 'toolchain_type')",
        "toolchain(",
        "  name = 'bar_toolchain',",
        "  toolchain = ':bar',",
        "  toolchain_type = ':toolchain_type',",
        ")");
    scratch.file(
        moduleRoot.getRelative("bar_user~1.0").getRelative("toolchain.bzl").getPathString(),
        "def _bar_toolchain_impl(ctx):",
        "  return [platform_common.ToolchainInfo(type = 'bar')]",
        "bar_toolchain = rule(",
        "  implementation = _bar_toolchain_impl,",
        "  attrs = {'runtime': attr.label()},",
        ")");

    scratch.file(
        moduleRoot.getRelative("starlark_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("starlark_user~1.0").getRelative("BUILD").getPathString(),
        "load('@utils//:utils.bzl', 'starlark_rule')",
        "starlark_rule(",
        "  name = 'starlark_user',",
        "  deps = ['@utils//:runfiles_lib'],",
        "  visibility = ['//visibility:public'],",
        ")");

    scratch.file(
        moduleRoot.getRelative("starlark_tool_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("starlark_tool_user~1.0").getRelative("BUILD").getPathString(),
        "load('@utils//:utils.bzl', 'starlark_rule')",
        "starlark_rule(",
        "  name = 'starlark_tool_user',",
        "  deps = ['@utils//:runfiles_lib'],",
        "  visibility = ['//visibility:public'],",
        ")");

    scratch.file(
        moduleRoot.getRelative("starlark_non_user~1.0").getRelative("WORKSPACE").getPathString());
    scratch.file(
        moduleRoot.getRelative("starlark_non_user~1.0").getRelative("BUILD").getPathString(),
        "load('@utils//:utils.bzl', 'starlark_rule')",
        "starlark_rule(",
        "  name = 'starlark_non_user',",
        "  deps = [",
        "    '@starlark_user//:starlark_user',",
        "    '@java_non_user//:java_non_user',",
        "    '@cc_non_user//:cc_non_user',",
        "  ],",
        "  tool_deps = [",
        "    '@starlark_tool_user//:starlark_tool_user',",
        "    '@utils//:runfiles_lib',",
        "  ],",
        "  visibility = ['//visibility:public'],",
        ")");

    assertThat(getRunfilesLibraryUsers("@@utils~1.0//:runfiles_lib")).isEmpty();

    RepositoryName ccUser = RepositoryName.createUnvalidated("cc_user~1.0");
    Map<RepositoryName, RepositoryMapping> ccUserRunfilesLibraryUsers = getRunfilesLibraryUsers(
        "@@cc_user~1.0//:cc_user");
    RepositoryMapping ccUserMapping = getRepositoryMapping(ccUser);
    assertThat(ccUserRunfilesLibraryUsers).containsExactly(ccUser, ccUserMapping);

    Map<RepositoryName, RepositoryMapping> ccNonUserRunfilesLibraryUsers = getRunfilesLibraryUsers(
        "@@cc_non_user~1.0//:cc_non_user");
    assertThat(ccNonUserRunfilesLibraryUsers).containsExactly(ccUser, ccUserMapping);

    RepositoryName javaUser = RepositoryName.createUnvalidated("java_user~1.0");
    Map<RepositoryName, RepositoryMapping> javaUserRunfilesLibraryUsers = getRunfilesLibraryUsers(
        "@@java_user~1.0//:java_user");
    RepositoryMapping javaUserMapping = getRepositoryMapping(javaUser);
    assertThat(javaUserRunfilesLibraryUsers).containsExactly(javaUser, javaUserMapping);

    Map<RepositoryName, RepositoryMapping> javaNonUserRunfilesLibraryUsers = getRunfilesLibraryUsers(
        "@@java_non_user~1.0//:java_non_user");
    assertThat(javaNonUserRunfilesLibraryUsers).containsExactly(javaUser, javaUserMapping);

    RepositoryName barUser = RepositoryName.createUnvalidated("bar_user~1.0");
    Map<RepositoryName, RepositoryMapping> barUserRunfilesLibraryUsers = getRunfilesLibraryUsers(
        "@@bar_user~1.0//:bar_user");
    RepositoryMapping barUserMapping = getRepositoryMapping(barUser);
    assertThat(barUserRunfilesLibraryUsers).containsExactly(barUser, barUserMapping);

    RepositoryName starlarkUser = RepositoryName.createUnvalidated("starlark_user~1.0");
    Map<RepositoryName, RepositoryMapping> starlarkUserRunfilesLibraryUsers = getRunfilesLibraryUsers(
        "@@starlark_user~1.0//:starlark_user");
    RepositoryMapping starlarkUserMapping = getRepositoryMapping(starlarkUser);
    assertThat(starlarkUserRunfilesLibraryUsers).containsExactly(starlarkUser, starlarkUserMapping);

    Map<RepositoryName, RepositoryMapping> starlarkNonUserRunfilesLibraryUsers = getRunfilesLibraryUsers(
        "@@starlark_non_user~1.0//:starlark_non_user");
    assertThat(starlarkNonUserRunfilesLibraryUsers).containsExactly(
        javaUser, javaUserMapping, ccUser, ccUserMapping, starlarkUser, starlarkUserMapping,
        barUser, barUserMapping);
  }

  private RepositoryMapping getRepositoryMapping(RepositoryName ccUser)
      throws InterruptedException {
    SkyKey key = RepositoryMappingValue.key(ccUser);
    EvaluationResult<RepositoryMappingValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
    return result.get(key).getRepositoryMapping();
  }

  private Map<RepositoryName, RepositoryMapping> getRunfilesLibraryUsers(String label)
      throws Exception {
    ConfiguredTarget target = getConfiguredTarget(Label.parseCanonical(label), targetConfig);
    RunfilesProvider runfilesProvider = target.getProvider(RunfilesProvider.class);

    assertThat(runfilesProvider).isNotNull();
    assertThat(runfilesProvider.getRunfilesLibraryUsers()).isNotNull();
    return runfilesProvider.getRunfilesLibraryUsers().toList().stream().collect(
        Collectors.toMap(RepositoryNameAndMapping::getRepositoryName,
            RepositoryNameAndMapping::getRepositoryMapping));
  }
}
