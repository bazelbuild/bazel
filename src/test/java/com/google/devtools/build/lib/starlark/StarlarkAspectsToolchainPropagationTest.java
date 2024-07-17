// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlark;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for Starlark aspects propagation to targets toolchain dependencies. */
@RunWith(TestParameterInjector.class)
public final class StarlarkAspectsToolchainPropagationTest extends AnalysisTestCase {

  /**
   * Sets up 3 toolchain rules:
   *
   * <p>test_toolchain: has no attribute dependency and no advertised providers
   *
   * <p>test_toolchain_with_provider: has an advertised provider but no attribute dependency
   *
   * <p>test_toolchain_with_dep: has an attribute dependency but no advertised providers
   *
   * <p>We also set up 3 toolchain types:
   *
   * <p>toolchain_type_1: resolved by `foo` of rule `test_toolchain`
   *
   * <p>toolchain_type_2: resolved by `foo_with_provider` of rule `test_toolchain_with_provider`
   *
   * <p>toolchain_type_3: resolved by `foo_with_dep` of rule `test_toolchain_with_dep`
   *
   * <p>Toolchain `foo_for_all` resolved both toolchain_type_2 and toolchain_type_3
   */
  public void createToolchainsAndPlatforms() throws Exception {
    scratch.overwriteFile(
        "rule/test_toolchain.bzl",
        """
        MyProvider = provider()

        def _impl(ctx):
            return [platform_common.ToolchainInfo(
                tool = ctx.executable._tool,
                files_to_run = ctx.attr._tool[DefaultInfo].files_to_run,
            ), MyProvider(value = str(ctx.label))]

        test_toolchain = rule(
            implementation = _impl,
            attrs = {
                "_tool": attr.label(
                    default = "//toolchain:a_tool",
                    executable = True,
                    cfg = "exec",
                ),
            },
        )

        test_toolchain_with_provider = rule(
            implementation = _impl,
            attrs = {
                "_tool": attr.label(
                    default = "//toolchain:a_tool",
                    executable = True,
                    cfg = "exec",
                ),
            },
            provides = [MyProvider]
        )

        test_toolchain_with_dep = rule(
            implementation = _impl,
            attrs = {
                "_tool": attr.label(
                    default = "//toolchain:a_tool",
                    executable = True,
                    cfg = "exec",
                ),
                "dep": attr.label(),
            },
        )

        """);
    scratch.overwriteFile(
        "rule/BUILD",
        """
        exports_files(["test_toolchain/bzl"])

        toolchain_type(name = "toolchain_type_1")
        alias(name = "toolchain_type_1_alias", actual = ":toolchain_type_1")

        toolchain_type(name = "toolchain_type_2")

        toolchain_type(name = "toolchain_type_3")
        """);
    scratch.overwriteFile(
        "toolchain/BUILD",
        """
        load("//rule:test_toolchain.bzl", "test_toolchain",
              "test_toolchain_with_provider", "test_toolchain_with_dep")

        genrule(
            name = "a_tool",
            outs = ["atool"],
            cmd = "",
            executable = True,
        )

        test_toolchain(
            name = "foo",
        )

        toolchain(
            name = "foo_toolchain",
            toolchain = ":foo",
            toolchain_type = "//rule:toolchain_type_1",
        )

        test_toolchain_with_provider(
            name = "foo_with_provider",
        )

        toolchain(
          name = "foo_toolchain_with_provider",
          toolchain = ":foo_with_provider",
          toolchain_type = "//rule:toolchain_type_2",
        )

        sh_library(name = "toolchain_dep")

        test_toolchain_with_dep(
          name = "foo_with_dep",
          dep = ":toolchain_dep",
        )

        toolchain(
          name = "foo_toolchain_with_dep",
          toolchain = ":foo_with_dep",
          toolchain_type = "//rule:toolchain_type_3",
        )

        test_toolchain(name = "foo_for_all")

        toolchain(
          name = "foo_type_2",
          toolchain = ":foo_for_all",
          toolchain_type = "//rule:toolchain_type_2",
        )

        toolchain(
          name = "foo_type_3",
          toolchain = ":foo_for_all",
          toolchain_type = "//rule:toolchain_type_3",
        )
        """);

    scratch.overwriteFile(
        "platforms/BUILD",
        """
        constraint_setting(name = "setting")

        constraint_value(
            name = "constraint_1",
            constraint_setting = ":setting",
        )

        constraint_value(
            name = "constraint_2",
            constraint_setting = ":setting",
        )

        platform(
            name = "platform_1",
            constraint_values = [":constraint_1"],
        )

        platform(
            name = "platform_2",
            constraint_values = [":constraint_2"],
            exec_properties = {
                "watermelon.ripeness": "unripe",
                "watermelon.color": "red",
            },
        )
        """);
  }

  @Before
  public void setup() throws Exception {
    createToolchainsAndPlatforms();
  }

  @Test
  @TestParameters({
    "{autoExecGroups: True}",
    "{autoExecGroups: False}",
  })
  public void aspectPropagatesToToolchain_singleDepAdded(String autoExecGroups) throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _impl(target, ctx):
          return []

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
        )

        no_toolchain_aspect = aspect(
          implementation = _impl,
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          toolchains = ['//rule:toolchain_type_1'],
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain",
        "--incompatible_auto_exec_groups=" + autoExecGroups);

    var unused =
        update(
            ImmutableList.of(
                "//test:defs.bzl%toolchain_aspect", "//test:defs.bzl%no_toolchain_aspect"),
            "//test:t1");

    var toolchainAspect =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var toolchainAspectNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(toolchainAspect))
            .findFirst()
            .orElse(null);
    assertThat(toolchainAspectNode).isNotNull();

    var noToolchainAspect =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%no_toolchain_aspect"));
    var noToolchainAspectNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(noToolchainAspect))
            .findFirst()
            .orElse(null);
    assertThat(noToolchainAspectNode).isNotNull();

    var toolchainAspectDirectDeps =
        ImmutableSet.copyOf(Iterables.filter(toolchainAspectNode.getDirectDeps(), SkyKey.class));
    var noToolchainAspectDirectDeps =
        ImmutableSet.copyOf(Iterables.filter(noToolchainAspectNode.getDirectDeps(), SkyKey.class));

    // only one extra dependency is added for the toolchain propagating aspect
    assertThat(toolchainAspectDirectDeps.size() - noToolchainAspectDirectDeps.size()).isEqualTo(1);
    assertThat(toolchainAspectDirectDeps).containsAtLeastElementsIn(noToolchainAspectDirectDeps);

    // the extra dependency is the aspect application on the target's resolved toolchain
    var aspectOnToolchainDep =
        Iterables.getOnlyElement(
            Sets.difference(toolchainAspectDirectDeps, noToolchainAspectDirectDeps));
    assertThat(aspectOnToolchainDep).isInstanceOf(AspectKey.class);
    assertThat(((AspectKey) aspectOnToolchainDep).getAspectName())
        .isEqualTo("//test:defs.bzl%toolchain_aspect");
    assertThat(((AspectKey) aspectOnToolchainDep).getLabel().toString())
        .isEqualTo("//toolchain:foo");
  }

  @Test
  public void aspectPropagatesToExecGpToolchain_singleDepAdded() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _impl(target, ctx):
          return []

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
        )

        no_toolchain_aspect = aspect(
          implementation = _impl,
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          exec_groups = {"gp": exec_group(toolchains = ['//rule:toolchain_type_1'])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration("--extra_toolchains=//toolchain:foo_toolchain");

    var unused =
        update(
            ImmutableList.of(
                "//test:defs.bzl%toolchain_aspect", "//test:defs.bzl%no_toolchain_aspect"),
            "//test:t1");

    var toolchainAspect =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var toolchainAspectNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(toolchainAspect))
            .findFirst()
            .orElse(null);
    assertThat(toolchainAspectNode).isNotNull();

    var noToolchainAspect =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%no_toolchain_aspect"));
    var noToolchainAspectNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(noToolchainAspect))
            .findFirst()
            .orElse(null);
    assertThat(noToolchainAspectNode).isNotNull();

    var toolchainAspectDirectDeps =
        ImmutableSet.copyOf(Iterables.filter(toolchainAspectNode.getDirectDeps(), SkyKey.class));
    var noToolchainAspectDirectDeps =
        ImmutableSet.copyOf(Iterables.filter(noToolchainAspectNode.getDirectDeps(), SkyKey.class));

    // only one extra dependency is added for the toolchain propagating aspect
    assertThat(toolchainAspectDirectDeps.size() - noToolchainAspectDirectDeps.size()).isEqualTo(1);
    assertThat(toolchainAspectDirectDeps).containsAtLeastElementsIn(noToolchainAspectDirectDeps);

    // the extra dependency is the aspect application on the target's resolved toolchain
    var aspectOnToolchainDep =
        Iterables.getOnlyElement(
            Sets.difference(toolchainAspectDirectDeps, noToolchainAspectDirectDeps));
    assertThat(aspectOnToolchainDep).isInstanceOf(AspectKey.class);
    assertThat(((AspectKey) aspectOnToolchainDep).getAspectName())
        .isEqualTo("//test:defs.bzl%toolchain_aspect");
    assertThat(((AspectKey) aspectOnToolchainDep).getLabel().toString())
        .isEqualTo("//toolchain:foo");
  }

  @Test
  @TestParameters({
    "{autoExecGroups: True}",
    "{autoExecGroups: False}",
  })
  public void aspectHasToolchains_dependencyEdgeCreated(String autoExecGroups) throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _impl(target, ctx):
          return []

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
          toolchains = ['//rule:toolchain_type_2'],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          toolchains = ['//rule:toolchain_type_1'],
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain,//toolchain:foo_toolchain_with_provider",
        "--incompatible_auto_exec_groups=" + autoExecGroups);

    var unused = update(ImmutableList.of("//test:defs.bzl%toolchain_aspect"), "//test:t1");

    var toolchainAspect =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var toolchainAspectNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(toolchainAspect))
            .findFirst()
            .orElse(null);
    assertThat(toolchainAspectNode).isNotNull();

    // A dependency edge is created from the aspect to its own toolchain but not to the target's
    // toolchain.
    var aspectConfiguredTargetDeps =
        Iterables.transform(
            Iterables.filter(
                toolchainAspectNode.getDirectDeps(), d -> d instanceof ConfiguredTargetKey),
            d -> ((ConfiguredTargetKey) d).getLabel().toString());
    assertThat(aspectConfiguredTargetDeps)
        .containsExactly("//toolchain:foo_with_provider", "//test:t1");
  }

  @Test
  @TestParameters({
    "{autoExecGroups: True}",
    "{autoExecGroups: False}",
  })
  public void aspectPropagatesToToolchainUsingToolchainTypeAlias(String autoExecGroups)
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _impl(target, ctx):
          return []

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
        )

        no_toolchain_aspect = aspect(
          implementation = _impl,
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          toolchains = ['//rule:toolchain_type_1_alias'],
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain",
        "--incompatible_auto_exec_groups=" + autoExecGroups);

    var unused =
        update(
            ImmutableList.of(
                "//test:defs.bzl%toolchain_aspect", "//test:defs.bzl%no_toolchain_aspect"),
            "//test:t1");

    var toolchainAspect =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var toolchainAspectNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(toolchainAspect))
            .findFirst()
            .orElse(null);
    assertThat(toolchainAspectNode).isNotNull();

    var noToolchainAspect =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%no_toolchain_aspect"));
    var noToolchainAspectNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(noToolchainAspect))
            .findFirst()
            .orElse(null);
    assertThat(noToolchainAspectNode).isNotNull();

    var toolchainAspectDirectDeps =
        ImmutableSet.copyOf(Iterables.filter(toolchainAspectNode.getDirectDeps(), SkyKey.class));
    var noToolchainAspectDirectDeps =
        ImmutableSet.copyOf(Iterables.filter(noToolchainAspectNode.getDirectDeps(), SkyKey.class));

    // only one extra dependency is added for the toolchain propagating aspect
    assertThat(toolchainAspectDirectDeps.size() - noToolchainAspectDirectDeps.size()).isEqualTo(1);
    assertThat(toolchainAspectDirectDeps).containsAtLeastElementsIn(noToolchainAspectDirectDeps);

    // the extra dependency is the aspect application on the target's resolved toolchain
    var aspectOnToolchainDep =
        Iterables.getOnlyElement(
            Sets.difference(toolchainAspectDirectDeps, noToolchainAspectDirectDeps));
    assertThat(aspectOnToolchainDep).isInstanceOf(AspectKey.class);
    assertThat(((AspectKey) aspectOnToolchainDep).getAspectName())
        .isEqualTo("//test:defs.bzl%toolchain_aspect");
    assertThat(((AspectKey) aspectOnToolchainDep).getLabel().toString())
        .isEqualTo("//toolchain:foo");
  }

  @Test
  @TestParameters({
    "{autoExecGroups: True}",
    "{autoExecGroups: False}",
  })
  public void toolchainPropagationBasedOnAspectRequiredProviders(String autoExecGroups)
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        load("//rule:test_toolchain.bzl", "MyProvider")

        def _impl(target, ctx):
          return []

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1', '//rule:toolchain_type_2'],
          required_providers = [MyProvider],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          toolchains = ['//rule:toolchain_type_1'],
          exec_groups = {"gp": exec_group(toolchains = ['//rule:toolchain_type_2'])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain,//toolchain:foo_toolchain_with_provider",
        "--incompatible_auto_exec_groups=" + autoExecGroups);

    var unused = update(ImmutableList.of("//test:defs.bzl%toolchain_aspect"), "//test:t1");

    var aspectOnTarget =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var aspectOnTargetNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnTarget))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnTargetNode).isNotNull();

    // aspect propagated only to //toolchain:foo_with_provider
    var aspectOnToolchain =
        Iterables.getOnlyElement(
            Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class));
    assertThat(aspectOnToolchain.getLabel().toString()).isEqualTo("//toolchain:foo_with_provider");
    assertThat(aspectOnToolchain.getAspectName()).isEqualTo("//test:defs.bzl%toolchain_aspect");
  }

  @Test
  public void aspectPropagatesToToolchainDeps() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _impl(target, ctx):
          return []

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_3'],
          attr_aspects = ['dep'],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          exec_groups = {"gp": exec_group(toolchains = ['//rule:toolchain_type_3'])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration("--extra_toolchains=//toolchain:foo_toolchain_with_dep");

    var unused = update(ImmutableList.of("//test:defs.bzl%toolchain_aspect"), "//test:t1");

    var aspectOnTarget =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var aspectOnTargetNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnTarget))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnTargetNode).isNotNull();

    var aspectOnToolchain =
        Iterables.getOnlyElement(
            Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class));
    var aspectOnToolchainNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnToolchain))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnToolchainNode).isNotNull();
    assertThat(aspectOnToolchain.getLabel().toString()).isEqualTo("//toolchain:foo_with_dep");

    var aspectOnToolchainDep =
        Iterables.getOnlyElement(
            Iterables.filter(aspectOnToolchainNode.getDirectDeps(), AspectKey.class));
    assertThat(aspectOnToolchainDep.getLabel().toString()).isEqualTo("//toolchain:toolchain_dep");
    assertThat(aspectOnToolchainDep.getAspectName()).isEqualTo("//test:defs.bzl%toolchain_aspect");
  }

  @Test
  public void requiredAspectPropagatesToToolchain() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _impl(target, ctx):
          return []

        required_aspect = aspect(implementation = _impl)

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
          requires = [required_aspect],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          exec_groups = {"gp": exec_group(toolchains = ['//rule:toolchain_type_1'])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration("--extra_toolchains=//toolchain:foo_toolchain");

    var unused = update(ImmutableList.of("//test:defs.bzl%toolchain_aspect"), "//test:t1");

    var aspectOnTarget =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var aspectOnTargetNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnTarget))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnTargetNode).isNotNull();

    var aspectsDeps =
        Iterables.transform(
            Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class),
            k -> k.getAspectName() + " on " + k.getLabel().toString());
    assertThat(aspectsDeps).hasSize(3);
    // toolchain_aspect requires required_aspect so required_aspect will be propagated before
    // toolchain_aspect to //test:t1 and its toolchain
    assertThat(aspectsDeps)
        .containsExactly(
            "//test:defs.bzl%required_aspect on //test:t1",
            "//test:defs.bzl%toolchain_aspect on //toolchain:foo",
            "//test:defs.bzl%required_aspect on //toolchain:foo");
  }

  @Test
  @TestParameters({
    "{autoExecGroups: True}",
    "{autoExecGroups: False}",
  })
  public void aspectOnAspectPropagateToToolchain(String autoExecGroups) throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        Prov1 = provider()
        Prov2 = provider()

        def _impl(target, ctx):
          return []

        def _impl_1(target, ctx):
          return [Prov1()]

        def _impl_2(target, ctx):
          return [Prov2()]

        toolchain_aspect_1 = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
          required_aspect_providers = [Prov1]
        )

        no_toolchain_aspect = aspect(
          implementation = _impl_1,
          provides = [Prov1],
          required_aspect_providers = [Prov2]
        )

        toolchain_aspect_2 = aspect(
          implementation = _impl_2,
          toolchains_aspects = ['//rule:toolchain_type_1'],
          provides = [Prov2],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          toolchains = ['//rule:toolchain_type_1'],
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain",
        "--incompatible_auto_exec_groups=" + autoExecGroups);

    var unused =
        update(
            ImmutableList.of(
                "//test:defs.bzl%toolchain_aspect_2",
                "//test:defs.bzl%no_toolchain_aspect", "//test:defs.bzl%toolchain_aspect_1"),
            "//test:t1");

    var aspectOnTarget =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect_1"));
    assertThat(aspectOnTarget.getBaseKeys()).hasSize(1);
    assertThat(aspectOnTarget.getBaseKeys().get(0).getAspectName())
        .isEqualTo("//test:defs.bzl%no_toolchain_aspect");

    var aspectOnTargetNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnTarget))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnTargetNode).isNotNull();

    var aspectsOnToolchain =
        Iterables.transform(
            Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class),
            k -> k.getAspectName() + " on " + k.getLabel().toString());
    assertThat(aspectsOnToolchain).hasSize(4);
    // Only `toolchain_aspect_1` and `toolchain_aspect_2` are propagated to the toolchain
    assertThat(aspectsOnToolchain)
        .containsExactly(
            "//test:defs.bzl%toolchain_aspect_2 on //test:t1",
            "//test:defs.bzl%no_toolchain_aspect on //test:t1",
            "//test:defs.bzl%toolchain_aspect_1 on //toolchain:foo",
            "//test:defs.bzl%toolchain_aspect_2 on //toolchain:foo");

    var toolchainAspect1 =
        Iterables.getOnlyElement(
            Iterables.filter(
                Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class),
                k ->
                    k.getAspectName().equals("//test:defs.bzl%toolchain_aspect_1")
                        && k.getLabel().toString().equals("//toolchain:foo")));
    // Since `toolchain_aspect_1` only depends on `no_toolchain_aspect`, it will have no base keys
    // when applied on the toolchain.
    assertThat(toolchainAspect1.getBaseKeys()).isEmpty();
  }

  @Test
  public void execGroupWithMultipleToolchainTypes_aspectsPropagateToRelevantTypes()
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        Prov1 = provider()
        Prov2 = provider()

        def _impl(target, ctx):
          return []

        def _impl_1(target, ctx):
          return [Prov1()]

        def _impl_2(target, ctx):
          return [Prov2()]

        toolchain_aspect_0 = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
          required_aspect_providers = [[Prov1], [Prov2]]
        )

        toolchain_aspect_1 = aspect(
          implementation = _impl_1,
          toolchains_aspects = ['//rule:toolchain_type_3'],
          provides = [Prov1],
          required_aspect_providers = [Prov2]
        )

        toolchain_aspect_2 = aspect(
          implementation = _impl_2,
          toolchains_aspects = ['//rule:toolchain_type_1'],
          provides = [Prov2],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          attrs = {
            'dep': attr.label(),
          },
          exec_groups = {"gp": exec_group(
              toolchains = ['//rule:toolchain_type_1', '//rule:toolchain_type_3'])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain,//toolchain:foo_toolchain_with_dep");

    var unused =
        update(
            ImmutableList.of(
                "//test:defs.bzl%toolchain_aspect_2",
                "//test:defs.bzl%toolchain_aspect_1", "//test:defs.bzl%toolchain_aspect_0"),
            "//test:t1");

    var aspectOnTarget =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect_0"));
    var aspectOnTargetNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnTarget))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnTargetNode).isNotNull();

    var aspectsOnToolchain =
        Iterables.transform(
            Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class),
            k -> k.getAspectName() + " on " + k.getLabel().toString());
    assertThat(aspectsOnToolchain).hasSize(5);
    assertThat(aspectsOnToolchain)
        .containsExactly(
            "//test:defs.bzl%toolchain_aspect_1 on //test:t1",
            "//test:defs.bzl%toolchain_aspect_2 on //test:t1",
            // toolchain_aspect_0 and toolchain_aspect_2 propagate to //toolchain:foo of
            // //rule:toolchain_type_1
            "//test:defs.bzl%toolchain_aspect_0 on //toolchain:foo",
            "//test:defs.bzl%toolchain_aspect_2 on //toolchain:foo",
            // toolchain_aspect_1 propagates to //toolchain:foo_with_dep of //rule:toolchain_type_3
            "//test:defs.bzl%toolchain_aspect_1 on //toolchain:foo_with_dep");

    var toolchainAspect1 =
        Iterables.getOnlyElement(
            Iterables.filter(
                Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class),
                k ->
                    k.getAspectName().equals("//test:defs.bzl%toolchain_aspect_0")
                        && k.getLabel().toString().equals("//toolchain:foo")));
    // Since `toolchain_aspect_0` depends on `toolchain_aspect_2` when applied on //toolchain:foo,
    assertThat(Iterables.getOnlyElement(toolchainAspect1.getBaseKeys()).getAspectName())
        .isEqualTo("//test:defs.bzl%toolchain_aspect_2");
  }

  @Test
  public void toolchainTypesResolvedToSameToolchain_aspectsPropagateToSameToolchain()
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        prov = provider()

        def _impl(target, ctx):
          return []

        def _impl_1(target, ctx):
          return [prov()]

        toolchain_aspect_1 = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_2'],
          required_aspect_providers = [prov]
        )

        toolchain_aspect_2 = aspect(
          implementation = _impl_1,
          toolchains_aspects = ['//rule:toolchain_type_3'],
          provides = [prov],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          exec_groups = {"gp": exec_group(
              toolchains = ['//rule:toolchain_type_2', '//rule:toolchain_type_3'])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_type_2", "--extra_toolchains=//toolchain:foo_type_3");

    var unused =
        update(
            ImmutableList.of(
                "//test:defs.bzl%toolchain_aspect_2", "//test:defs.bzl%toolchain_aspect_1"),
            "//test:t1");

    var aspectOnTarget =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect_1"));
    var aspectOnTargetNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnTarget))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnTargetNode).isNotNull();

    var aspectsOnToolchain =
        Iterables.transform(
            Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class),
            k -> k.getAspectName() + " on " + k.getLabel().toString());
    assertThat(aspectsOnToolchain).hasSize(3);

    assertThat(aspectsOnToolchain)
        .containsExactly(
            "//test:defs.bzl%toolchain_aspect_2 on //test:t1",
            // both aspects propagated to //toolchain:foo_for_all because it resolves both the
            // toolchain types
            "//test:defs.bzl%toolchain_aspect_1 on //toolchain:foo_for_all",
            "//test:defs.bzl%toolchain_aspect_2 on //toolchain:foo_for_all");
  }

  @Test
  public void toolchainTypesResolvedToSameToolchainDiffExecPlatform_aspectPropagateTwice()
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _impl(target, ctx):
          return []

        toolchain_aspect = aspect(
          implementation = _impl,
          toolchains_aspects = ['//rule:toolchain_type_1'],
        )

        def _rule_impl(ctx):
          pass

        r1 = rule(
          implementation = _rule_impl,
          exec_groups = {
            "gp1": exec_group(
              toolchains = ['//rule:toolchain_type_1'],
              exec_compatible_with = ['//platforms:constraint_2']
              ),
            "gp2": exec_group(toolchains = ['//rule:toolchain_type_1'])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'r1')
        r1(name = 't1')
        """);
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain",
        "--extra_execution_platforms=//platforms:platform_1,//platforms:platform_2");

    var unused = update(ImmutableList.of("//test:defs.bzl%toolchain_aspect"), "//test:t1");

    var aspectOnTarget =
        Iterables.getOnlyElement(getAspectKeys("//test:t1", "//test:defs.bzl%toolchain_aspect"));
    var aspectOnTargetNode =
        skyframeExecutor.getEvaluator().getInMemoryGraph().getAllNodeEntries().stream()
            .filter(n -> n.getKey().equals(aspectOnTarget))
            .findFirst()
            .orElse(null);
    assertThat(aspectOnTargetNode).isNotNull();

    var aspectsOnToolchain =
        Iterables.transform(
            Iterables.filter(aspectOnTargetNode.getDirectDeps(), AspectKey.class),
            k ->
                k.getAspectName()
                    + " on "
                    + k.getLabel().toString()
                    + ", exec_platform: "
                    + k.getBaseConfiguredTargetKey().getExecutionPlatformLabel().toString());
    assertThat(aspectsOnToolchain).hasSize(2);
    // aspect propagated twice on the same toolchain target but with different execution platform
    assertThat(aspectsOnToolchain)
        .containsExactly(
            "//test:defs.bzl%toolchain_aspect on //toolchain:foo, exec_platform:"
                + " //platforms:platform_2",
            "//test:defs.bzl%toolchain_aspect on //toolchain:foo, exec_platform:"
                + " //platforms:platform_1");
  }

  private ImmutableList<AspectKey> getAspectKeys(String targetLabel, String aspectLabel) {
    return skyframeExecutor.getEvaluator().getDoneValues().entrySet().stream()
        .filter(
            entry ->
                entry.getKey() instanceof AspectKey
                    && ((AspectKey) entry.getKey()).getAspectClass().getName().equals(aspectLabel)
                    && ((AspectKey) entry.getKey()).getLabel().toString().equals(targetLabel))
        .map(e -> (AspectKey) e.getKey())
        .collect(toImmutableList());
  }
}
