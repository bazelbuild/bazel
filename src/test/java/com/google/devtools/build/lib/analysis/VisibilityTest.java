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

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for visibility of targets. */
@RunWith(JUnit4.class)
public class VisibilityTest extends AnalysisTestCase {

  @Before
  public final void setUpToolsConfigMock() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
  }

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
            values = {"compilation_mode": "dbg"},
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
        """
        in my_rule rule //foo:target_with_aspects: Visibility error:
        target '//tool:rule_tool' is not visible from
        target '//rule:lib.bzl'\
        """);
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility):
            cc_library(
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility, deps):
            cc_library(
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load("//submacro:defs.bzl", "my_submacro")

        def _impl(name, visibility, deps, use_submacro):
            callable = my_submacro if use_submacro else cc_library
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file("subsubmacro/BUILD");
    scratch.file(
        "subsubmacro/defs.bzl",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility, deps):
            cc_library(
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility):
            cc_library(
                name = name,
                deps = ["//pkg:tool"],
            )

        my_macro = macro(implementation = _impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility, _implicit_dep):
            cc_library(
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//macro:__pkg__"],
        )
        """);
    scratch.file("subsubmacro/BUILD");
    scratch.file(
        "subsubmacro/defs.bzl",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility):
            cc_library(
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//pkg:__pkg__"],
        )
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility):
            cc_library(
                name = name + "_baz",
                deps = ["//tool:tool"],
            )

        my_submacro = macro(implementation = _impl)
        """);
    scratch.file("macro/BUILD");
    scratch.file(
        "macro/defs.bzl",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load("//submacro:defs.bzl", "my_submacro")

        def _impl(name, visibility, extra_level_deep):
            if extra_level_deep:
                my_submacro(name = name + "_bar")
            else:
                cc_library(
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "tool",
            visibility = ["//pkg:__pkg__"],
        )
        """);
    scratch.file("submacro/BUILD");
    scratch.file(
        "submacro/defs.bzl",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        def _impl(name, visibility, deps):
            if not deps:
                deps = ["//tool:tool"]
            cc_library(
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "dep",
            visibility = ["//visibility:private"],
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "foo",
            deps = ["//dep:dep"],
        )
        """);

    assertThrows(ViewCreationFailedException.class, () -> update("//pkg:foo"));
    assertContainsEvent(
        """
        * If you think the dependency on this source file is legitimate, consider either \
        depending on the file via an appropriate filegroup() target, or updating the file's \
        visibility using exports_files().\
        """);
  }

  @Test
  public void testSimpleAllows() throws Exception {
    addPackageGroupWithPackages("p", "oneapple", "//idared");
    addRuleWithVisibility("lib", "lib", "//p:oneapple");
    addRuleWithDeps("idared", "fruit", "//lib:lib");
    outputVisibilityTestRules();

    update("//idared:fruit");
  }

  @Test
  public void testSimpleDenies() throws Exception {
    addPackageGroupWithPackages("p", "oneapple", "//jonagold");
    addRuleWithVisibility("lib", "lib", "//p:oneapple");
    addRuleWithDeps("idared", "fruit", "//lib:lib");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//idared:fruit"));
    assertIllegalDependency("//idared:fruit", "//lib:lib");
  }

  @Test
  public void testPrivateDenies() throws Exception {
    addRuleWithVisibility("lib", "lib", "//visibility:private");
    addRuleWithDeps("code", "code", "//lib:lib");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//code:code"));
    assertIllegalDependency("//code:code", "//lib:lib");
  }

  @Test
  public void testPrivateAllows() throws Exception {
    addRuleWithVisibility("lib", "lib", "//visibility:private");
    addRuleWithDeps("lib", "code", "//lib:lib");
    outputVisibilityTestRules();

    update("//lib:code");
  }

  @Test
  public void testPublicAllows() throws Exception {
    addRuleWithVisibility("lib", "lib", "//visibility:public");
    addRuleWithDeps("code", "code", "//lib:lib");
    outputVisibilityTestRules();

    update("//code:code");
  }

  @Test
  public void testRulesInSamePackagesAlwaysVisible() throws Exception {
    addPackageGroupWithPackages("people", "adults", "//adults");
    addRuleWithVisibility("medicine", "paracetamol", "//people:adults");
    addRuleWithDeps("medicine", "pill", "//medicine:paracetamol");
    outputVisibilityTestRules();

    update("//medicine:pill");
  }

  @Test
  public void testDefaultCanDeny() throws Exception {
    addPackageGroupWithPackages("p", "oneapple", "//granny_smith");
    addDefaultVisibility("land", "//p:oneapple");
    addRule("land", "land");
    addRuleWithDeps("red_delicious", "apple", "//land:land");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//red_delicious:apple"));
    assertIllegalDependency("//red_delicious:apple", "//land:land");
  }

  @Test
  public void testDefaultCanAllow() throws Exception {
    addPackageGroupWithPackages("p", "oneapple", "//granny_smith");
    addDefaultVisibility("land", "//p:oneapple");
    addRule("land", "land");
    addRuleWithDeps("granny_smith", "apple", "//land:land");
    outputVisibilityTestRules();

    update("//granny_smith:apple");
  }

  @Test
  public void testAttributeOverridesDefault() throws Exception {
    addPackageGroupWithPackages("p", "heroes", "//rambo"); // Yes! This is a type of red apple!
    addPackageGroupWithPackages("p", "people", "//granny_smith");
    addDefaultVisibility("land", "//p:people");
    addRuleWithVisibility("land", "land", "//p:heroes");
    addRuleWithDeps("rambo", "apple", "//land:land");
    outputVisibilityTestRules();

    update("//rambo:apple");
  }

  @Test
  public void testCommandLine() throws Exception {
    useConfiguration("--default_visibility=public");

    addRule("land", "land");
    addRuleWithDeps("red_delicious", "apple", "//land:land");
    outputVisibilityTestRules();

    update("//red_delicious:apple");

    // checks that the package is correctly reloaded
    useConfiguration("--default_visibility=private");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//red_delicious:apple"));
    assertIllegalDependency("//red_delicious:apple", "//land:land");
  }

  @Test
  public void testDefaultDenyOverridesCommandlinePublic() throws Exception {
    useConfiguration("--default_visibility=public");

    addPackageGroupWithPackages("p", "oneapple", "//granny_smith");
    addDefaultVisibility("land", "//p:oneapple");
    addRule("land", "land");
    addRuleWithDeps("red_delicious", "apple", "//land:land");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//red_delicious:apple"));
    assertIllegalDependency("//red_delicious:apple", "//land:land");
  }

  @Test
  public void testDefaultAllowOverridesCommandlinePrivate() throws Exception {
    useConfiguration("--default_visibility=private");

    addPackageGroupWithPackages("p", "oneapple", "//granny_smith");
    addDefaultVisibility("land", "//p:oneapple");
    addRule("land", "land");
    addRuleWithDeps("granny_smith", "apple", "//land:land");
    outputVisibilityTestRules();

    update("//granny_smith:apple");
  }

  @Test
  public void testRuleOverridesCommandLinePrivate() throws Exception {
    useConfiguration("--default_visibility=private");

    addRuleWithVisibility("lib", "lib", "//visibility:public");
    addRuleWithDeps("code", "code", "//lib:lib");
    outputVisibilityTestRules();

    update("//code:code");
  }

  @Test
  public void testRuleOverridesCommandLinePublic() throws Exception {
    useConfiguration("--default_visibility=public");

    addRuleWithVisibility("lib", "lib", "//visibility:private");
    addRuleWithDeps("code", "code", "//lib:lib");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//code:code"));
    assertIllegalDependency("//code:code", "//lib:lib");
  }

  @Test
  public void testInvalidDefaultVisibility() throws Exception {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> useConfiguration("--default_visibility=foo"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "While parsing option --default_visibility=foo: "
                + "Not a valid default visibility: 'foo' (should be 'public' or 'private'");
  }

  @Test
  public void testExportsFilesDefaultVisibilityPublic() throws Exception {
    scratch.file(
        "kiwi/BUILD",
        """
        package(default_visibility = ["//visibility:private"])

        exports_files(["melon.py"])
        """);
    scratch.file(
        "apple/BUILD",
        """
        load('@rules_python//python:py_library.bzl', 'py_library')
        py_library(name='apple', srcs=['//kiwi:melon.py'])
        """);

    update("//apple:apple");
  }

  @Test
  public void testExportsFilesCanOverrideDefaultVisibility() throws Exception {
    scratch.file(
        "kiwi/BUILD",
        """
        package(default_visibility = ["//visibility:private"])

        exports_files(
            ["melon.py"],
            visibility = ["//visibility:public"],
        )
        """);
    scratch.file(
        "apple/BUILD",
        """
        load('@rules_python//python:py_library.bzl', 'py_library')
        py_library(name='apple', srcs=['//kiwi:melon.py'])
        """);

    update("//apple:apple");
  }

  @Test
  public void testExportsFilesCanUsePackageGroupInOtherPackage() throws Exception {
    scratch.file("banana/BUILD", "package_group(name='banana', packages=['//mango'])");

    scratch.file("papaya/BUILD", "exports_files(['papaya.py'], visibility=['//banana:banana'])");

    scratch.file(
        "mango/BUILD",
        """
        load('@rules_python//python:py_library.bzl', 'py_library')
        py_library(name='mango', srcs=['//papaya:papaya.py'])
        """);

    update("//mango:mango");
  }

  @Test
  public void testExportedFilesArePublicByDefault() throws Exception {
    scratch.file("kiwi/BUILD", "exports_files(['kiwi.py'])");

    scratch.file(
        "lychee/BUILD",
        """
        load('@rules_python//python:py_library.bzl', 'py_library')
        py_library(name='lychee', srcs=['//kiwi:kiwi.py'])
        """);

    update("//lychee:lychee");
  }

  @Test
  public void testDefaultVisibilityCanUsePackageGroupInOtherPackage() throws Exception {
    addDefaultVisibility("pear", "//grape:grape");
    addPackageGroupWithPackages("grape", "grape", "//cherry");
    addRule("pear", "pear");
    addRuleWithDeps("cherry", "cherry", "//pear:pear");
    outputVisibilityTestRules();

    update("//cherry:cherry");
  }

  @Test
  public void testRuleVisibilityAppliesToInputFiles() throws Exception {
    scratch.file(
        "java/cantaloupe/BUILD",
        """
        load("@rules_java//java:java_binary.bzl", "java_binary")
        package(default_visibility = ["//visibility:private"])

        package_group(
            name = "bees",
            packages = ["//bee"],
        )

        java_binary(
            name = "cantaloupe",
            srcs = ["Cantaloupe.java"],
            visibility = [":bees"],
        )
        """);

    scratch.file(
        "bee/BUILD",
        """
        genrule(
            name = "bee",
            srcs = ["//java/cantaloupe:cantaloupe_deploy.jar"],
            outs = ["honey"],
            cmd = "#",
        )
        """);

    update("//bee");
  }

  @Test
  public void testPackageGroupVisibilityAllowsAccessToInputFiles() throws Exception {
    scratch.file(
        "honeydew/BUILD",
        """
        package_group(
            name = "melonusers",
            packages = ["//peach"],
        )

        exports_files(
            ["melon.py"],
            visibility = [":melonusers"],
        )
        """);
    scratch.file(
        "peach/BUILD",
        """
        load('@rules_python//python:py_library.bzl', 'py_library')
        py_library(name='peach', srcs=['//honeydew:melon.py'])
        """);

    update("//peach:peach");
  }

  @Test
  public void testPackageGroupVisibilityDeniesAccessToInputFiles() throws Exception {
    scratch.file(
        "honeydew/BUILD",
        """
        package_group(
            name = "melonusers",
            packages = ["//apricot"],
        )

        exports_files(
            ["melon.py"],
            visibility = [":melonusers"],
        )
        """);
    scratch.file(
        "peach/BUILD",
        """
        load('@rules_python//python:py_library.bzl', 'py_library')
        py_library(name='peach', srcs=['//honeydew:melon.py'])
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//peach:peach"));
    assertIllegalDependency("//peach:peach", "//honeydew:melon.py");
  }

  @Test
  public void testIllegalPackageGroupEntries() throws Exception {
    addPackageGroupWithPackages("p", "rotten", "-=-");
    addRuleWithVisibility("sickness", "sickness", "//p:rotten");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//sickness:sickness"));
    assertContainsEvent("invalid package name");
  }

  @Test
  public void testUnknownPackageGroupIncludeDirect() throws Exception {
    addPackageGroupWithIncludes("good", "virtue", ":charity");
    addRuleWithVisibility("afterlife", "heaven", "//good:virtue");
    addRuleWithDeps("people", "bombadil_toma", "//afterlife:heaven");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//people:bombadil_toma"));
    // The error message "label '//good:charity' does not refer to a package group" would also be
    // fine here.
    assertContainsEvent("no such target '//good:charity'");
  }

  @Test
  public void testUnknownPackageGroupIncludeIndirect() throws Exception {
    addPackageGroupWithIncludes("evil", "sin", ":greed");
    addPackageGroupWithIncludes("evil", "greed", ":mammon");
    addRuleWithVisibility("afterlife", "hell", "//evil:sin");
    addRuleWithDeps("people", "dagobert", "//afterlife:hell");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//people:dagobert"));
    assertContainsEvent("no such target '//evil:mammon'");
  }

  @Test
  public void testPackageGroupIncludesNonGroup() throws Exception {
    addPackageGroupWithIncludes("fruits", "apple", ":jonathan");
    addRule("fruits", "jonathan");
    addRuleWithVisibility("banana", "banana", "//fruits:apple");
    addRuleWithDeps("monkey", "monkey", "//banana:banana");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//monkey:monkey"));
    assertContainsEvent("Label '//fruits:jonathan' does not refer to a package group");
  }

  @Test
  public void testPackageGroupIncludesCanAllow() throws Exception {
    addPackageGroupWithIncludes("fruit", "tropical", "citrus");
    addPackageGroupWithPackages("fruit", "citrus", "//fruit/orange");
    addRuleWithVisibility("land", "land", "//fruit:tropical");
    addRuleWithDeps("fruit/orange", "root", "//land:land");
    outputVisibilityTestRules();

    update("//fruit/orange:root");
  }

  @Test
  public void testPackageGroupIncludesCanDeny() throws Exception {
    addPackageGroupWithIncludes("fruit", "tropical", "citrus");
    addPackageGroupWithPackages("fruit", "citrus", "//fruit/orange");
    addRuleWithVisibility("land", "land", "//fruit:tropical");
    addRuleWithDeps("fruit/apple", "root", "//land:land");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//fruit/apple:root"));
    assertIllegalDependency("//fruit/apple:root", "//land:land");
  }

  @Test
  public void testLongPackageGroupIncludeChain() throws Exception {
    addPackageGroupWithIncludes("geo", "europe", ":germany");
    addPackageGroupWithIncludes("geo", "germany", ":bavaria");
    addPackageGroupWithIncludes("geo", "bavaria", ":munchen");
    addPackageGroupWithPackages("geo", "munchen", "//devtools/blaze");
    addRuleWithVisibility("something", "important", "//geo:europe");
    addRuleWithDeps("devtools/blaze", "blaze", "//something:important");
    outputVisibilityTestRules();

    update("//devtools/blaze:blaze");
  }

  @Test
  public void testPackageGroupCanIncludeFromOtherPackage() throws Exception {
    addPackageGroupWithIncludes("apple", "apple", "//peach:peach");
    addPackageGroupWithPackages("peach", "peach", "//plum");
    addRuleWithVisibility("pear", "pear", "//apple:apple");
    addRuleWithDeps("plum", "plum", "//pear:pear");
    outputVisibilityTestRules();

    update("//plum:plum");
  }

  @Test
  public void testAllBeneathPackageSpecificationCanAllow() throws Exception {
    addPackageGroupWithPackages("fruits", "tropical", "//fruits/tropical/...");
    addRuleWithVisibility("fruits", "sun", "//fruits:tropical");
    addRuleWithDeps("fruits/tropical", "coconut", "//fruits:sun");
    addRuleWithDeps("fruits/tropical/stinky", "durian", "//fruits:sun");
    addRuleWithDeps("fruits/tropical/asia/tasty", "lychee", "//fruits:sun");
    outputVisibilityTestRules();

    update(
        "//fruits/tropical:coconut",
        "//fruits/tropical/stinky:durian",
        "//fruits/tropical/asia/tasty:lychee");
  }

  @Test
  public void testAllBeneathPackageSpecificationCanDeny() throws Exception {
    addPackageGroupWithPackages("fruits", "tropical", "//fruits/tropical/...");
    addRuleWithVisibility("fruits", "sun", "//fruits:tropical");
    addRuleWithDeps("fruits/temperate", "apple", "//fruits:sun");
    addRuleWithDeps("fruits/tropicality", "durianity", "//fruits:sun");
    addRuleWithDeps("fruits/tropic", "lych", "//fruits:sun");
    addRuleWithDeps("fruits/tropic/photo", "sunflower", "//fruits:sun");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//fruits/temperate:apple"));
    assertThrows(ViewCreationFailedException.class, () -> update("//fruits/tropicality:durianity"));
    assertThrows(ViewCreationFailedException.class, () -> update("//fruits/tropic:lych"));
    assertThrows(
        ViewCreationFailedException.class, () -> update("//fruits/tropic/photo:sunflower"));

    assertIllegalDependency("//fruits/temperate:apple", "//fruits:sun");
    assertIllegalDependency("//fruits/tropicality:durianity", "//fruits:sun");
    assertIllegalDependency("//fruits/tropic:lych", "//fruits:sun");
    assertIllegalDependency("//fruits/tropic/photo:sunflower", "//fruits:sun");
  }

  // Regression test for bug #2674950:
  // "NullPointerException in RuleConfiguredTarget.lookupMakeVariable()"
  @Test
  public void testDefaultVisibilityHasNonPackageGroup() throws Exception {
    scratch.file(
        "plum/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        package(default_visibility = ["//pear"])

        foo_library(
            name = "sh",
            srcs = ["sh.sh"],
        )

        foo_library(
            name = "dep",
            srcs = ["dep.sh"],
        )

        foo_library(
            name = "target",
            deps = [":dep"],
        )
        """);

    scratch.file(
        "pear/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name='pear', srcs=['pear.sh'])");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//plum:target"));
    assertContainsEvent("does not refer to a package group");
  }

  @Test
  public void testDefaultVisibilityIncludesNonPackageGroup() throws Exception {
    scratch.file(
        "plum/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        package(default_visibility = ["//plum"])

        package_group(
            name = "plum",
            includes = ["//pear:pear"],
        )

        foo_library(
            name = "sh",
            srcs = ["sh.sh"],
        )

        foo_library(
            name = "dep",
            srcs = ["dep.sh"],
        )

        foo_library(
            name = "target",
            deps = [":dep"],
        )
        """);

    scratch.file(
        "pear/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name='pear', srcs=['pear.sh'])");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//plum:target"));
    assertContainsEvent("does not refer to a package group");
  }

  @Test
  public void testDefaultVisibilityIsRuleItself() throws Exception {
    scratch.file(
        "melon/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        package(default_visibility = ["//melon"])

        foo_library(
            name = "melon",
            srcs = ["melon.cc"],
        )
        """);
  }

  @Test
  public void testCanAllowPackageInVisibilityList() throws Exception {
    scratch.file(
        "plum/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name='plum', visibility=['//kiwi:__pkg__'])");

    scratch.file(
        "kiwi/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name='kiwi', deps=['//plum:plum'])");

    update("//kiwi:kiwi");
  }

  @Test
  public void testSubtreeVisibilityCanAllow() throws Exception {
    addRuleWithVisibility("banana", "banana", "//fruit:__subpackages__");
    addRuleWithDeps("fruit", "fruit", "//banana:banana");
    addRuleWithDeps("fruit/apple", "apple", "//banana:banana");
    addRuleWithDeps("fruit/apple/nicoter", "nicoter", "//banana:banana");
    outputVisibilityTestRules();

    update("//fruit:fruit", "//fruit/apple:apple", "//fruit/apple/nicoter:nicoter");
  }

  @Test
  public void testSubtreeVisibilityCanDeny() throws Exception {
    addRuleWithVisibility("peach", "peach", "//fruit/apple:__subpackages__");
    addRuleWithDeps("fruit", "fruit", "//peach:peach");
    addRuleWithDeps("fruit/plum", "plum", "//peach:peach");
    addRuleWithDeps("vegetable/tomato", "tomato", "//peach:peach");
    outputVisibilityTestRules();

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//fruit:fruit"));
    assertThrows(ViewCreationFailedException.class, () -> update("//fruit/plum:plum"));
    assertThrows(ViewCreationFailedException.class, () -> update("//vegetable/tomato:tomato"));

    assertIllegalDependency("//fruit:fruit", "//peach:peach");
    assertIllegalDependency("//fruit/plum:plum", "//peach:peach");
    assertIllegalDependency("//vegetable/tomato:tomato", "//peach:peach");
  }

  @Test
  public void testLocationCueForDefaultVisibilityMissingRule() throws Exception {
    scratch.file(
        "cherry/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        package(default_visibility = ["//cherry:flower"])

        foo_library(
            name = "seed",
            srcs = ["seed.sh"],
        )
        """);
    scratch.file(
        "lemon/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name='lemon', deps=['//cherry:seed'])");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//lemon:lemon"));
    assertContainsEvent(
        "no such target '//cherry:flower': target 'flower' not "
            + "declared in package 'cherry' defined by /workspace/cherry/BUILD");
  }

  private void doVisibilityDisablingTest(String dependencyVisibility) throws Exception {
    addRuleWithVisibility("fruit/cherry", "cherry", dependencyVisibility);
    addRuleWithDeps("drink/cherry", "cherry", "//fruit/cherry:cherry");
    outputVisibilityTestRules();

    // Use update() rather than getConfiguredTarget(), to ensure we go through BuildView#update
    // thereby potentially triggering the warning event.
    update("//drink/cherry:cherry");
  }

  @Test
  public void testVisibilityCheckCanBeDisabled() throws Exception {
    useConfiguration("--nocheck_visibility");
    doVisibilityDisablingTest("//visibility:private");

    assertThat(eventCollector.filtered(EventKind.ERROR)).isEmpty();
    assertContainsEvent(
        "This build has globally disabled target visibility checking (--nocheck_visibility).");
  }

  @Test
  public void testVisibilityNagMessageAppearsEvenWithoutVisibilityViolations() throws Exception {
    useConfiguration("--nocheck_visibility");
    doVisibilityDisablingTest("//visibility:public");

    assertThat(eventCollector.filtered(EventKind.ERROR)).isEmpty();
    assertContainsEvent(
        "This build has globally disabled target visibility checking (--nocheck_visibility).");
  }

  @Test
  public void testVisibilityNagMessageDoesNotAppearWhenNotDisabled() throws Exception {
    doVisibilityDisablingTest("//visibility:public");

    assertNoEvents();
  }

  @Test
  public void testSelectVisibility() throws Exception {
    scratch.file(
        "grape/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        package(default_visibility = ["//visibility:private"])

        cc_library(name = "grape")
        """);
    scratch.file(
        "wine/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        config_setting(
            name = "white_config",
            values = {"define": "color=white"},
        )

        config_setting(
            name = "red_config",
            values = {"define": "color=red"},
        )

        cc_library(
            name = "white_wine",
            deps = select({
                ":white_config": ["//grape"],
                "//conditions:default": [],
            }),
        )

        cc_library(
            name = "red_wine",
            deps = select({
                ":red_config": ["//grape"],
                "//conditions:default": [],
            }),
        )
        """);

    useConfiguration("--define", "color=red");
    update("//wine:white_wine");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//wine:red_wine"));
    assertContainsEvent("target '//grape:grape' is not visible from\ntarget '//wine:red_wine'");
  }

  // Regression test for bug #4073933:
  // "IllegalArgumentException in OutputFileConfiguredTarget.getGeenratingRule()"
  @Test
  public void testGetGeneratingRuleForOutputFileWithDefaultVisibility() throws Exception {
    scratch.file(
        "mango/BUILD",
        """
        package(default_visibility = ["//mango:skin"])

        package_group(name = "skin")

        genrule(
            name = "mango",
            srcs = [],
            outs = ["mango_out"],
            cmd = "",
        )
        """);

    update("//mango:mango_out", "//mango:mango");
    OutputFileConfiguredTarget mangoOut =
        (OutputFileConfiguredTarget) getConfiguredTarget("//mango:mango_out");
    ConfiguredTarget out = getConfiguredTarget("//mango:mango");

    assertThat(mangoOut.getGeneratingRule().getLabel()).isEqualTo(out.getLabel());
  }

  // Regression test for bug #4471264:
  // --nocheck_visibility_in_experimental doesn't propagate to exec configuration
  @Test
  public void testExecDepsWithNoCheckVisibility() throws Exception {
    useConfiguration("--nocheck_visibility");

    scratch.file(
        "kiwi/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        package(default_visibility = ["//visibility:private"])

        foo_library(
            name = "kiwi",
            srcs = ["kiwi.sh"],
        )
        """);
    scratch.file(
        "mango/BUILD",
        """
        load("//test_defs:foo_binary.bzl", "foo_binary")

        foo_binary(
            name = "mango",
            srcs = ["mango.sh"],
            deps = ["//kiwi"],
        )

        genrule(
            name = "genmango",
            outs = ["mango.out"],
            cmd = "",
            tools = [":mango"],
        )
        """);

    update("//mango:genmango");
  }

  // This is some infrastructure to make visibility test cases more legible.
  // add* methods should be called first, and when you are done, call
  // outputVisibilityTestRules(), which will output the actual BUILD files.

  Map<String, StringBuilder> packageToLines = new HashMap<>();

  private void addPackageLine(String pkg, String line) {
    if (!packageToLines.containsKey(pkg)) {
      packageToLines.put(pkg, new StringBuilder());
    }

    packageToLines.get(pkg).append(line);
    packageToLines.get(pkg).append("\n");
  }

  private void addPackageGroupWithIncludes(String pkg, String name, String... includes) {
    String includesList = "'" + Joiner.on("', '").join(includes) + "'";

    String line = String.format("package_group(name='%s', includes=[%s])", name, includesList);
    addPackageLine(pkg, line);
  }

  private void addPackageGroupWithPackages(String pkg, String name, String... entries) {
    String entriesList = "'" + Joiner.on("', '").join(entries) + "'";

    String line = String.format("package_group(name='%s', packages=[%s])", name, entriesList);
    addPackageLine(pkg, line);
  }

  private void addDefaultVisibility(String pkg, String... entries) {
    String visList = "'" + Joiner.on("', '").join(entries) + "'";
    String line = String.format("package(default_visibility=[%s])", visList);

    addPackageLine(pkg, line);
  }

  private void addRule(String pkg, String name) {
    String line = String.format("foo_library(name='%s')", name);
    addPackageLine(pkg, line);
  }

  private void addRuleWithVisibility(String pkg, String name, String... visibilities) {
    String visList = "'" + Joiner.on("', '").join(visibilities) + "'";
    String line = String.format("foo_library(name='%s', visibility=[%s])", name, visList);

    addPackageLine(pkg, line);
  }

  private void addRuleWithDeps(String pkg, String name, String... deps) {
    String depList = "'" + Joiner.on("', '").join(deps) + "'";
    String line = String.format("foo_library(name='%s', deps=[%s])", name, depList);

    addPackageLine(pkg, line);
  }

  private void outputVisibilityTestRules() throws Exception {
    for (String pkg : packageToLines.keySet()) {
      scratch.file(
          pkg + "/BUILD",
          "load('//test_defs:foo_library.bzl', 'foo_library')",
          packageToLines.get(pkg).toString());
    }
  }

  private void assertIllegalDependency(String from, String to) {
    assertContainsEvent(
        String.format("Visibility error:\ntarget '%s' is not visible from\ntarget '%s'", to, from));
  }
}
