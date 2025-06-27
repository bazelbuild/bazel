// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for visibility of targets. */
@RunWith(JUnit4.class)
public class VisibilityTest extends AnalysisTestCase {

  void setupArgsScenario() throws Exception {
    scratch.file("tool/tool.sh", "#!/bin/sh", "echo Hello > $2", "cat $1 >> $2");
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/rule.bzl",
        """
        def _impl(ctx):
            output = ctx.actions.declare_file(ctx.label.name + ".out")
            ctx.actions.run(
                inputs = ctx.files._tool + ctx.files.data,
                executable = ctx.files._tool[0].path,
                arguments = [f.path for f in ctx.files.data] + [output.path],
                outputs = [output],
            )

        greet = rule(
            implementation = _impl,
            attrs = {
                "data": attr.label(allow_files = True),
                "_tool": attr.label(
                    cfg = "exec",
                    allow_files = True,
                    default = Label("//tool:tool.sh"),
                ),
            },
            outputs = {"out": "%{name}.out"},
        )
        """);
    scratch.file("data/data.txt", "World");
    scratch.file(
        "use/BUILD",
        """
        load("//rule:rule.bzl", "greet")

        greet(
            name = "world",
            data = "//data:data.txt",
        )
        """);
  }

  @Test
  public void testToolVisibilityRuleCheckAtRule() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//rule:__pkg__'])");
    update("//use:world");
    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testToolVisibilityUseCheckAtUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//use:__pkg__'])");
    update("//use:world");
    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testToolVisibilityUseCheckAtRule_fallbackToUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//use:__pkg__'])");

    update("//use:world");

    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testToolVisibilityPrivateCheckAtUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:private'])");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testToolVisibilityPrivate() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:private'])");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testDataVisibilityUseCheckPrivateAtRule() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//use:__pkg__'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:public'])");
    update("//use:world");
    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testDataVisibilityPrivate() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:private'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:public'])");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testConfigSettingVisibilityAlwaysCheckedAtUse() throws Exception {
    scratch.file(
        "BUILD",
        "load('//build_defs:defs.bzl', 'my_rule')",
        "my_rule(",
        "    name = 'my_target',",
        "    value = select({",
        "        '//config_setting:my_setting': 'foo',",
        "        '//conditions:default': 'bar',",
        "    }),",
        ")");
    scratch.file("build_defs/BUILD");
    scratch.file(
        "build_defs/defs.bzl",
        """
        def _my_rule_impl(ctx):
            pass

        my_rule = rule(
            implementation = _my_rule_impl,
            attrs = {
                "value": attr.string(mandatory = True),
            },
        )
        """);
    scratch.file(
        "config_setting/BUILD",
        """
        config_setting(
            name = "my_setting",
            values = {"cpu": "does_not_matter"},
            visibility = ["//:__pkg__"],
        )
        """);

    update("//:my_target");
    assertThat(hasErrors(getConfiguredTarget("//:my_target"))).isFalse();
  }

  @Test
  public void testImplicitDependency_samePackageAsDefinition_visible() throws Exception {
    scratch.file(
        "aspect_def/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = "aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "aspect_def/lib.bzl",
        """
        def _impl_my_aspect(ctx, target):
            return []

        my_aspect = aspect(
            _impl_my_aspect,
            attrs = {"_aspect_tool": attr.label(default = "//aspect_def:aspect_tool")},
        )
        """);
    scratch.file(
        "rule_def/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = "rule_tool",
            srcs = ["a.sh"],
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "rule_def/lib.bzl",
        """
        load("//aspect_def:lib.bzl", "my_aspect")

        def _impl(ctx):
            pass

        my_rule = rule(
            _impl,
            attrs = {
                "dep": attr.label(aspects = [my_aspect]),
                "_rule_tool": attr.label(default = "//rule_def:rule_tool"),
            },
        )
        simple_starlark_rule = rule(
            _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("//rule_def:lib.bzl", "my_rule", "simple_starlark_rule")

        simple_starlark_rule(name = "simple_dep")

        my_rule(
            name = "my_target",
            dep = ":simple_dep",
        )
        """);

    update("//foo:my_target");

    assertThat(hasErrors(getConfiguredTarget("//foo:my_target"))).isFalse();
  }

  @Test
  public void testAspectImplicitDependencyCheckedAtDefinition_visible() throws Exception {
    scratch.file("inner_aspect/BUILD");
    scratch.file(
        "inner_aspect/lib.bzl",
        """
        InnerAspectInfo = provider()

        def _impl_inner_aspect(ctx, target):
            return [InnerAspectInfo()]

        inner_aspect = aspect(
            _impl_inner_aspect,
            attrs = {"_inner_aspect_tool": attr.label(default = "//tool:inner_aspect_tool")},
            provides = [InnerAspectInfo],
        )
        """);
    scratch.file("outer_aspect/BUILD");
    scratch.file(
        "outer_aspect/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "InnerAspectInfo")

        def _impl_outer_aspect(ctx, target):
            return []

        outer_aspect = aspect(
            _impl_outer_aspect,
            attrs = {"_outer_aspect_tool": attr.label(default = "//tool:outer_aspect_tool")},
            required_aspect_providers = [InnerAspectInfo],
        )
        """);
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "inner_aspect")
        load("//outer_aspect:lib.bzl", "outer_aspect")

        def _impl(ctx):
            pass

        my_rule = rule(
            _impl,
            attrs = {
                "dep": attr.label(aspects = [inner_aspect, outer_aspect]),
                "_rule_tool": attr.label(default = "//tool:rule_tool"),
            },
        )
        simple_starlark_rule = rule(
            _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("//rule:lib.bzl", "my_rule", "simple_starlark_rule")

        simple_starlark_rule(name = "simple_dep")

        my_rule(
            name = "target_with_aspects",
            dep = ":simple_dep",
        )
        """);
    scratch.file(
        "tool/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        foo_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        foo_binary(
            name = "rule_tool",
            srcs = ["a.sh"],
            visibility = ["//rule:__pkg__"],
        )
        """);

    update("//foo:target_with_aspects");

    assertThat(hasErrors(getConfiguredTarget("//foo:target_with_aspects"))).isFalse();
  }

  @Test
  public void testAspectImplicitDependencyCheckedAtDefinition_visibleWithNameCollision()
      throws Exception {
    scratch.file("inner_aspect/BUILD");
    scratch.file(
        "inner_aspect/lib.bzl",
        """
        InnerAspectInfo = provider()

        def _impl_inner_aspect(ctx, target):
            return [InnerAspectInfo()]

        inner_aspect = aspect(
            _impl_inner_aspect,
            attrs = {"_tool": attr.label(default = "//tool:inner_aspect_tool")},
            provides = [InnerAspectInfo],
        )
        """);
    scratch.file("outer_aspect/BUILD");
    scratch.file(
        "outer_aspect/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "InnerAspectInfo")

        def _impl_outer_aspect(ctx, target):
            return []

        outer_aspect = aspect(
            _impl_outer_aspect,
            attrs = {"_tool": attr.label(default = "//tool:outer_aspect_tool")},
            required_aspect_providers = [InnerAspectInfo],
        )
        """);
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "inner_aspect")
        load("//outer_aspect:lib.bzl", "outer_aspect")

        def _impl(ctx):
            pass

        my_rule = rule(
            _impl,
            attrs = {
                "dep": attr.label(aspects = [inner_aspect, outer_aspect]),
                "_tool": attr.label(default = "//tool:rule_tool"),
            },
        )
        simple_starlark_rule = rule(
            _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("//rule:lib.bzl", "my_rule", "simple_starlark_rule")

        simple_starlark_rule(name = "simple_dep")

        my_rule(
            name = "target_with_aspects",
            dep = ":simple_dep",
        )
        """);
    scratch.file(
        "tool/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        foo_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        foo_binary(
            name = "rule_tool",
            srcs = ["a.sh"],
            visibility = ["//rule:__pkg__"],
        )
        """);

    update("//foo:target_with_aspects");

    assertThat(hasErrors(getConfiguredTarget("//foo:target_with_aspects"))).isFalse();
  }

  @Test
  public void testAspectImplicitDependencyCheckedAtDefinition_outerAspectToolNotVisible()
      throws Exception {
    scratch.file("inner_aspect/BUILD");
    scratch.file(
        "inner_aspect/lib.bzl",
        """
        InnerAspectInfo = provider()

        def _impl_inner_aspect(ctx, target):
            return [InnerAspectInfo()]

        inner_aspect = aspect(
            _impl_inner_aspect,
            attrs = {"_inner_aspect_tool": attr.label(default = "//tool:inner_aspect_tool")},
            provides = [InnerAspectInfo],
        )
        """);
    scratch.file("outer_aspect/BUILD");
    scratch.file(
        "outer_aspect/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "InnerAspectInfo")

        def _impl_outer_aspect(ctx, target):
            return []

        outer_aspect = aspect(
            _impl_outer_aspect,
            attrs = {"_outer_aspect_tool": attr.label(default = "//tool:outer_aspect_tool")},
            required_aspect_providers = [InnerAspectInfo],
        )
        """);
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "inner_aspect")
        load("//outer_aspect:lib.bzl", "outer_aspect")

        def _impl(ctx):
            pass

        my_rule = rule(
            _impl,
            attrs = {
                "dep": attr.label(aspects = [inner_aspect, outer_aspect]),
                "_rule_tool": attr.label(default = "//tool:rule_tool"),
            },
        )
        simple_starlark_rule = rule(
            _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("//rule:lib.bzl", "my_rule", "simple_starlark_rule")

        simple_starlark_rule(name = "simple_dep")

        my_rule(
            name = "target_with_aspects",
            dep = ":simple_dep",
        )
        """);
    scratch.file(
        "tool/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = [
                "//inner_aspect:__pkg__",
                "//rule:__pkg__",
            ],
        )

        foo_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        foo_binary(
            name = "rule_tool",
            srcs = ["a.sh"],
            visibility = ["//rule:__pkg__"],
        )
        """);
    reporter.removeHandler(failFastHandler);

    assertThrows(ViewCreationFailedException.class, () -> update("//foo:target_with_aspects"));
    assertContainsEvent(
        "in //inner_aspect:lib.bzl%inner_aspect,//outer_aspect:lib.bzl%outer_aspect "
            + "aspect on simple_starlark_rule rule //foo:simple_dep: Visibility error:\n"
            + "target '//tool:outer_aspect_tool' is not visible from\n"
            + "target '//outer_aspect:lib.bzl'");
  }

  @Test
  public void testAspectImplicitDependencyCheckedAtDefinition_innerAspectToolNotVisible()
      throws Exception {
    scratch.file("inner_aspect/BUILD");
    scratch.file(
        "inner_aspect/lib.bzl",
        """
        InnerAspectInfo = provider()

        def _impl_inner_aspect(ctx, target):
            return [InnerAspectInfo()]

        inner_aspect = aspect(
            _impl_inner_aspect,
            attrs = {"_inner_aspect_tool": attr.label(default = "//tool:inner_aspect_tool")},
            provides = [InnerAspectInfo],
        )
        """);
    scratch.file("outer_aspect/BUILD");
    scratch.file(
        "outer_aspect/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "InnerAspectInfo")

        def _impl_outer_aspect(ctx, target):
            return []

        outer_aspect = aspect(
            _impl_outer_aspect,
            attrs = {"_outer_aspect_tool": attr.label(default = "//tool:outer_aspect_tool")},
            required_aspect_providers = [InnerAspectInfo],
        )
        """);
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "inner_aspect")
        load("//outer_aspect:lib.bzl", "outer_aspect")

        def _impl(ctx):
            pass

        my_rule = rule(
            _impl,
            attrs = {
                "dep": attr.label(aspects = [inner_aspect, outer_aspect]),
                "_rule_tool": attr.label(default = "//tool:rule_tool"),
            },
        )
        simple_starlark_rule = rule(
            _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("//rule:lib.bzl", "my_rule", "simple_starlark_rule")

        simple_starlark_rule(name = "simple_dep")

        my_rule(
            name = "target_with_aspects",
            dep = ":simple_dep",
        )
        """);
    scratch.file(
        "tool/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        foo_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = [
                "//outer_aspect:__pkg__",
                "//rule:__pkg__",
            ],
        )

        foo_binary(
            name = "rule_tool",
            srcs = ["a.sh"],
            visibility = ["//rule:__pkg__"],
        )
        """);
    reporter.removeHandler(failFastHandler);

    assertThrows(ViewCreationFailedException.class, () -> update("//foo:target_with_aspects"));
    assertContainsEvent(
        "in //inner_aspect:lib.bzl%inner_aspect aspect on simple_starlark_rule "
            + "rule //foo:simple_dep: Visibility error:\n"
            + "target '//tool:inner_aspect_tool' is not visible from\n"
            + "target '//inner_aspect:lib.bzl'");
  }

  @Test
  public void testAspectImplicitDependencyCheckedAtDefinition_ruleToolNotVisible()
      throws Exception {
    scratch.file("inner_aspect/BUILD");
    scratch.file(
        "inner_aspect/lib.bzl",
        """
        InnerAspectInfo = provider()

        def _impl_inner_aspect(ctx, target):
            return [InnerAspectInfo()]

        inner_aspect = aspect(
            _impl_inner_aspect,
            attrs = {"_inner_aspect_tool": attr.label(default = "//tool:inner_aspect_tool")},
            provides = [InnerAspectInfo],
        )
        """);
    scratch.file("outer_aspect/BUILD");
    scratch.file(
        "outer_aspect/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "InnerAspectInfo")

        def _impl_outer_aspect(ctx, target):
            return []

        outer_aspect = aspect(
            _impl_outer_aspect,
            attrs = {"_outer_aspect_tool": attr.label(default = "//tool:outer_aspect_tool")},
            required_aspect_providers = [InnerAspectInfo],
        )
        """);
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/lib.bzl",
        """
        load("//inner_aspect:lib.bzl", "inner_aspect")
        load("//outer_aspect:lib.bzl", "outer_aspect")

        def _impl(ctx):
            pass

        my_rule = rule(
            _impl,
            attrs = {
                "dep": attr.label(aspects = [inner_aspect, outer_aspect]),
                "_rule_tool": attr.label(default = "//tool:rule_tool"),
            },
        )
        simple_starlark_rule = rule(
            _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("//rule:lib.bzl", "my_rule", "simple_starlark_rule")

        simple_starlark_rule(name = "simple_dep")

        my_rule(
            name = "target_with_aspects",
            dep = ":simple_dep",
        )
        """);
    scratch.file(
        "tool/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        foo_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        foo_binary(
            name = "rule_tool",
            srcs = ["a.sh"],
            visibility = [
                "//inner_aspect:__pkg__",
                "//outer_aspect:__pkg__",
            ],
        )
        """);
    reporter.removeHandler(failFastHandler);

    assertThrows(ViewCreationFailedException.class, () -> update("//foo:target_with_aspects"));
    assertContainsEvent(
        "in my_rule rule //foo:target_with_aspects: Visibility error:\n"
            + "target '//tool:rule_tool' is not visible from\n"
            + "target '//rule:lib.bzl'");
  }

  void setupFilesScenario(String wantRead) throws Exception {
    scratch.file("src/source.txt", "source");
    scratch.file("src/BUILD", "exports_files(['source.txt'], visibility=['//pkg:__pkg__'])");
    scratch.file("pkg/foo.txt", "foo");
    scratch.file("pkg/bar.txt", "bar");
    scratch.file("pkg/groupfile.txt", "groupfile");
    scratch.file("pkg/unused.txt", "unused");
    scratch.file("pkg/exported.txt", "exported");
    scratch.file(
        "pkg/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        exports_files(["exported.txt"])

        genrule(
            name = "foobar",
            srcs = [
                "foo.txt",
                "bar.txt",
            ],
            outs = ["foobar.txt"],
            cmd = "cat $(SRCS) > $@",
        )

        filegroup(
            name = "remotegroup",
            srcs = ["//src:source.txt"],
        )

        filegroup(
            name = "localgroup",
            srcs = [":groupfile.txt"],
        )
        """);
    scratch.file(
        "otherpkg/BUILD",
        "genrule(",
        "  name = 'it',",
        "  srcs = ['//pkg:" + wantRead + "'],",
        "  outs = ['it.xt'],",
        "  cmd = 'cp $< $@',",
        ")");
  }

  @Test
  public void testTargetImplicitExport() throws Exception {
    setupFilesScenario("foobar");
    useConfiguration("--noincompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testTargetNoImplicitExport() throws Exception {
    setupFilesScenario("foobar");
    useConfiguration("--incompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testLocalFilegroupImplicitExport() throws Exception {
    setupFilesScenario("localgroup");
    useConfiguration("--noincompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testLocalFilegroupNoImplicitExport() throws Exception {
    setupFilesScenario("localgroup");
    useConfiguration("--incompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testRemoteFilegroupImplicitExport() throws Exception {
    setupFilesScenario("remotegroup");
    useConfiguration("--noincompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testRemoteFilegroupNoImplicitExport() throws Exception {
    setupFilesScenario("remotegroup");
    useConfiguration("--incompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testExportedImplicitExport() throws Exception {
    setupFilesScenario("exported.txt");
    useConfiguration("--noincompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testExportedNoImplicitExport() throws Exception {
    setupFilesScenario("exported.txt");
    useConfiguration("--incompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testUnusedImplicitExport() throws Exception {
    setupFilesScenario("unused.txt");
    useConfiguration("--noincompatible_no_implicit_file_export");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//otherpkg:it"));
  }

  @Test
  public void testUnusedNoImplicitExport() throws Exception {
    setupFilesScenario("unused.txt");
    useConfiguration("--incompatible_no_implicit_file_export");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//otherpkg:it"));
  }

  @Test
  public void testSourcefileImplicitExport() throws Exception {
    setupFilesScenario("foo.txt");
    useConfiguration("--noincompatible_no_implicit_file_export");
    update("//otherpkg:it");
    assertThat(hasErrors(getConfiguredTarget("//otherpkg:it"))).isFalse();
  }

  @Test
  public void testSourcefileNoImplicitExport() throws Exception {
    setupFilesScenario("foo.txt");
    useConfiguration("--incompatible_no_implicit_file_export");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//otherpkg:it"));
  }

  @Test
  public void testVerboseDiagnostics_ruleImplicitDep() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/defs.bzl",
        """
        def _impl(ctx):
            pass

        my_rule = rule(
            implementation = _impl,
            attrs = {"_implicit_dep": attr.label(default="//tool:tool")},
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rule:defs.bzl", "my_rule")

        my_rule(name = "foo")
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * The dependency is an implicit dependency of the consuming target's rule, my_rule, which \
        is defined in //rule:defs.bzl. Since that file's package, //rule, does not match\
        """);
  }

  @Test
  public void testVerboseDiagnostics_aspectImplicitDep() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("aspect/BUILD");
    scratch.file(
        "aspect/defs.bzl",
        """
        def _impl(ctx):
            pass

        my_aspect = aspect(
            implementation = _impl,
            attrs = {"_implicit_dep": attr.label(default="//tool:tool")},
        )
        """);
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/defs.bzl",
        """
        load("//aspect:defs.bzl", "my_aspect")

        def _impl(ctx):
            pass

        my_rule = rule(
            implementation = _impl,
            attrs = {"deps": attr.label_list(aspects=[my_aspect])},
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rule:defs.bzl", "my_rule")

        cc_library(name = "dep")

        my_rule(
            name = "foo",
            deps = [":dep"],
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * The dependency is an implicit dependency of the consuming target's aspect, my_aspect, \
        which is defined in //aspect:defs.bzl. Since that file's package, //aspect, does not\
        """);
  }

  @Test
  public void testVerboseDiagnostics_consumingLocation_isNotInMacro() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        cc_library(
            name = "foo",
            deps = ["//tool:tool"],
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * The location being checked is the package where the consuming target lives, //pkg.\
        """);
  }

  @Test
  public void testVerboseDiagnostics_consumingLocation_isInMacro() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        def _impl(name, visibility):
            native.cc_library(
                name = name + "_bar",
                deps = ["//tool:tool"],
            )

        my_macro = macro(implementation = _impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        my_macro(name = "foo")
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo_bar"));
    assertContainsEvent(
        """
        * Because the consuming target was declared in the body of the symbolic macro my_macro \
        defined in //macro:defs.bzl, the location being checked is this file's package, //macro.\
        """);
  }

  @Test
  public void testVerboseDiagnostics_consumingLocation_isDelegatedFromPackage() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        def _impl(name, visibility, deps):
            native.cc_library(
                name = name + "_baz",
                deps = deps,
            )

        my_submacro = macro(
            implementation = _impl,
            attrs = {"deps": attr.label_list()},
        )
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("//submacro:defs.bzl", "my_submacro")

        def _impl(name, visibility, deps, use_submacro):
            callable = my_submacro if use_submacro else native.cc_library
            callable(
                name = name + "_bar",
                deps = deps,
            )

        my_macro = macro(
            implementation = _impl,
            attrs = {
                "deps": attr.label_list(),
                "use_submacro": attr.bool(configurable=False),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        my_macro(
            name = "foo",
            deps = ["//tool:tool"],
            use_submacro = False,
        )

        my_macro(
            name = "foo2",
            deps = ["//tool:tool"],
            use_submacro = True,
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo_bar"));
    assertContainsEvent(
        """
        * Because the dependency was passed to the consuming target from an attribute of the \
        symbolic macro //pkg:foo, the location being checked is the place where this macro is \
        declared: package //pkg.\
        """);

    eventCollector.clear();
    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo2_bar_baz"));
    // Should say "transitive", and should still identify outer macro (foo2), not inner macro
    // (foo2_bar).
    assertContainsEvent(
        """
        * Because the dependency was transitively passed to the consuming target from an attribute \
        of the symbolic macro //pkg:foo2, the location being checked is the place where this macro \
        is declared: package //pkg.\
        """);
  }

  @Test
  public void testVerboseDiagnostics_consumingLocation_isDelegatedFromMacro() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("subsubmacro/BUILD");
    scratch.file(
        "subsubmacro/defs.bzl",
        """
        def _impl(name, visibility, deps):
            native.cc_library(
                name = name + "_qux",
                deps = deps,
            )

        my_subsubmacro = macro(
            implementation = _impl,
            attrs = {"deps": attr.label_list()},
        )
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        load("//subsubmacro:defs.bzl", "my_subsubmacro")

        def _impl(name, visibility, deps):
            my_subsubmacro(
                name = name + "_baz",
                deps = deps,
            )

        my_submacro = macro(
            implementation = _impl,
            attrs = {"deps": attr.label_list()},
        )
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("//submacro:defs.bzl", "my_submacro")
        load("//subsubmacro:defs.bzl", "my_subsubmacro")

        def _impl(name, visibility, extra_level_deep):
            callable = my_submacro if extra_level_deep else my_subsubmacro
            callable(
                name = name + "_bar",
                deps = ["//tool:tool"],
            )

        my_macro = macro(
            implementation = _impl,
            attrs = {"extra_level_deep": attr.bool(configurable=False)},
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        my_macro(
            name = "foo",
            extra_level_deep = False,
        )

        my_macro(
            name = "foo2",
            extra_level_deep = True,
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo_bar_qux"));
    assertContainsEvent(
        """
        * Because the dependency was passed to the consuming target from an attribute of the \
        symbolic macro //pkg:foo_bar, the location being checked is the place where this macro is \
        declared: the body of the calling macro my_macro, defined in //macro:defs.bzl of package \
        //macro.\
        """);

    eventCollector.clear();
    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo2_bar_baz_qux"));
    // Should say "transitive", and should still identify outer macro (foo2_bar), not inner macro
    // (foo2_bar_baz).
    assertContainsEvent(
        """
        * Because the dependency was transitively passed to the consuming target from an attribute \
        of the symbolic macro //pkg:foo2_bar, the location being checked is the place where this \
        macro is declared: the body of the calling macro my_macro, defined in //macro:defs.bzl of \
        package //macro.\
        """);
  }

  @Test
  public void testVerboseDiagnostics_aliasDisclaimer_shownForAlias() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//pkg:__pkg__"],
        )

        alias(
            name = "indirect",
            actual = ":tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        cc_library(
            name = "foo",
            deps = ["//tool:indirect"],
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * The dependency is an alias target. Note that it is the visibility of the alias we care \
        about, not the visibility of the underlying target it refers to.
        """);
  }

  @Test
  public void testVerboseDiagnostics_aliasDisclaimer_notShownForNonAlias() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        cc_library(
            name = "foo",
            deps = ["//tool:tool"],
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertDoesNotContainEvent("The dependency is an alias");
  }

  @Test
  public void testVerboseDiagnostics_samePackageDisclaimer() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        def _impl(name, visibility):
            native.cc_library(
                name = name,
                deps = ["//pkg:tool"],
            )

        my_macro = macro(implementation = _impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )

        my_macro(
            name = "foo",
        )
        """);
    scratch.file(
        "pkg2/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        my_macro(
            name = "foo",
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * Although both targets live in the same package, they cannot automatically see each other \
        because they are declared by different symbolic macros.\
        """);

    eventCollector.clear();
    assertThrows(ViewCreationFailedException.class, () -> update("//pkg2:foo"));
    assertDoesNotContainEvent("both targets live in the same package");
  }

  @Test
  public void testVerboseDiagnostics_samePackageDisclaimer_shownForImplicitDepOfMacro()
      throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        def _impl(name, visibility, _implicit_dep):
            native.cc_library(
                name = name,
                deps = [_implicit_dep],
            )

        my_macro = macro(
            implementation = _impl,
            attrs = {"_implicit_dep": attr.label(default="//pkg:tool", configurable=False)},
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )

        my_macro(
            name = "foo",
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * Although both targets live in the same package, they cannot automatically see each other \
        because they are declared by different symbolic macros.\
        """);
  }

  @Test
  public void testVerboseDiagnostics_samePackageDisclaimer_notShownForImplicitDepOfRule()
      throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file("rule/BUILD");
    scratch.file(
        "rule/defs.bzl",
        """
        def _impl(ctx):
            pass

        my_rule = rule(
            implementation = _impl,
            attrs = {"_implicit_dep": attr.label(default="//pkg:tool")},
        )
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("//rule:defs.bzl", "my_rule")

        def _impl(name, visibility):
            my_rule(name = name)

        my_macro = macro(implementation = _impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )

        my_macro(name = "foo")
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertDoesNotContainEvent("both targets live in the same package");
  }

  @Test
  public void testVerboseDiagnostics_moreDelegationNeeded_fromAncestorMacro() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//macro:__pkg__"],
        )
        """);
    scratch.file("subsubmacro/BUILD");
    scratch.file(
        "subsubmacro/defs.bzl",
        """
        def _impl(name, visibility):
            native.cc_library(
                name = name + "_qux",
                deps = ["//tool:tool"],
            )

        my_subsubmacro = macro(implementation = _impl)
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        load("//subsubmacro:defs.bzl", "my_subsubmacro")

        def _impl(name, visibility):
            my_subsubmacro(name = name + "_baz")

        my_submacro = macro(implementation = _impl)
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("//submacro:defs.bzl", "my_submacro")
        load("//subsubmacro:defs.bzl", "my_subsubmacro")

        def _impl(name, visibility, extra_level_deep):
            callable = my_submacro if extra_level_deep else my_subsubmacro
            callable(name = name + "_bar")

        my_macro = macro(
            implementation = _impl,
            attrs = {"extra_level_deep": attr.bool(configurable=False)},
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        my_macro(
            name = "foo",
            extra_level_deep = False,
        )

        my_macro(
            name = "foo2",
            extra_level_deep = True,
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo_bar_qux"));
    assertContainsEvent(
        """
        * Although the dependency is not visible to the location being checked, it is visible to \
        this location's caller, //pkg:foo, a my_macro macro defined in //macro. (Perhaps the \
        caller needs to pass in the dependency as an argument?)\
        """);

    eventCollector.clear();
    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo2_bar_baz_qux"));
    // Should say "transitive", and should still identify outer macro (foo2_bar), not inner macro
    // (foo2_bar_baz).
    assertContainsEvent(
        """
        * Although the dependency is not visible to the location being checked, it is visible to \
        this location's transitive caller, //pkg:foo2, a my_macro macro defined in //macro. \
        (Perhaps this caller, or an intermediate caller, needs to pass in the dependency as an \
        argument?)\
        """);
  }

  @Test
  public void testVerboseDiagnostics_moreDelegationNeeded_fromBuildFile() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//pkg:__pkg__"],
        )
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        def _impl(name, visibility):
            native.cc_library(
                name = name + "_baz",
                deps = ["//tool:tool"],
            )

        my_submacro = macro(implementation = _impl)
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("//submacro:defs.bzl", "my_submacro")

        def _impl(name, visibility, extra_level_deep):
            if extra_level_deep:
                my_submacro(name = name + "_bar")
            else:
                native.cc_library(
                    name = name + "_bar",
                    deps = ["//tool:tool"],
                )

        my_macro = macro(
            implementation = _impl,
            attrs = {"extra_level_deep": attr.bool(configurable=False)},
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        my_macro(
            name = "foo",
            extra_level_deep = False,
        )

        my_macro(
            name = "foo2",
            extra_level_deep = True,
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo_bar"));
    assertContainsEvent(
        """
        * Although the dependency is not visible to the location being checked, it is visible to \
        this location's caller, the BUILD file of package //pkg. (Perhaps the caller needs to pass \
        in the dependency as an argument?)\
        """);

    eventCollector.clear();
    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo2_bar_baz"));
    // Should say "transitive".
    assertContainsEvent(
        """
        * Although the dependency is not visible to the location being checked, it is visible to \
        this location's transitive caller, the BUILD file of package //pkg. (Perhaps this caller, \
        or an intermediate caller, needs to pass in the dependency as an argument?)\
        """);
  }

  @Test
  public void testVerboseDiagnostics_moreDelegationNeeded_incompleteDelegation() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "tool/BUILD",
        """
        cc_library(
            name = "tool",
            visibility = ["//pkg:__pkg__"],
        )
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        def _impl(name, visibility, deps):
            if not deps:
                deps = ["//tool:tool"]
            native.cc_library(
                name = name + "_baz",
                deps = deps,
            )

        my_submacro = macro(
            implementation = _impl,
            attrs = {"deps": attr.label_list(configurable=False)},
        )
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("//submacro:defs.bzl", "my_submacro")

        def _impl(name, visibility, deps, pass_in_tool):
            my_submacro(
                name = name + "_bar",
                deps = ["//tool:tool"] if pass_in_tool else [],
            )

        my_macro = macro(
            implementation = _impl,
            attrs = {
                "deps": attr.label_list(),
                "pass_in_tool": attr.bool(configurable=False),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macro:defs.bzl", "my_macro")

        my_macro(
            name = "foo",
            pass_in_tool = True,
        )

        my_macro(
            name = "foo2",
            deps = ["//tool:tool"],
            pass_in_tool = False,
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo_bar_baz"));
    // my_macro passed it to my_submacro, but BUILD didn't pass it to my_macro.
    assertContainsEvent(
        """
        * Although the dependency is not visible to the location being checked, it is visible to \
        this location's caller, the BUILD file of package //pkg. (Perhaps the caller needs to pass \
        in the dependency as an argument?)\
        """);

    eventCollector.clear();
    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo2_bar_baz"));
    // BUILD passed it to my_macro, but my_macro didn't pass it to my_submacro.
    assertContainsEvent(
        """
        * Although the dependency is not visible to the location being checked, it is visible to \
        this location's transitive caller, the BUILD file of package //pkg. (Perhaps this caller, \
        or an intermediate caller, needs to pass in the dependency as an argument?)\
        """);
  }

  @Test
  public void testVerboseDiagnostics_editVisibilitySuggestion_forRuleTarget() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "dep/BUILD",
        """
        cc_library(
            name = "dep",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        cc_library(
            name = "foo",
            deps = ["//dep:dep"],
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * If you think the dependency is legitimate, consider updating its visibility declaration.\
        """);
  }

  @Test
  public void testVerboseDiagnostics_editVisibilitySuggestion_forFileTarget() throws Exception {
    useConfiguration("--verbose_visibility_errors");
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "dep/BUILD",
        """
        exports_files(
            ["dep"],
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        cc_library(
            name = "foo",
            deps = ["//dep:dep"],
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * If you think the dependency on this source file is legitimate, consider updating its \
        visibility declaration using exports_files().\
        """);
  }
}
