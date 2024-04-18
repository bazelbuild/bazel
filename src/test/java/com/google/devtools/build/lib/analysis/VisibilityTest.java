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
        sh_binary(
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
        sh_binary(
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
        sh_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        sh_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        sh_binary(
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
        sh_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        sh_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        sh_binary(
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
        sh_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = [
                "//inner_aspect:__pkg__",
                "//rule:__pkg__",
            ],
        )

        sh_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        sh_binary(
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
        sh_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        sh_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = [
                "//outer_aspect:__pkg__",
                "//rule:__pkg__",
            ],
        )

        sh_binary(
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
        sh_binary(
            name = "outer_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//outer_aspect:__pkg__"],
        )

        sh_binary(
            name = "inner_aspect_tool",
            srcs = ["a.sh"],
            visibility = ["//inner_aspect:__pkg__"],
        )

        sh_binary(
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
}
