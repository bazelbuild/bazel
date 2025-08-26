// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertEventCount;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.skyframe.BzlLoadCycleReporter;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.BzlmodRepoCycleReporter;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for module extension resolution. */
@RunWith(JUnit4.class)
public class ModuleExtensionResolutionTest extends BuildViewTestCase {
  private final CyclesReporter cyclesReporter =
      new CyclesReporter(new BzlLoadCycleReporter(), new BzlmodRepoCycleReporter());

  @Before
  public void setUpSimpleRepoRule() throws Exception {
    // Set up a simple repo rule.
    registry.addModule(
        createModuleKey("data_repo", "1.0"), "module(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("data_repo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("data_repo+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("data_repo+1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', 'data = '+json.encode(ctx.attr.data))",
        "  ctx.file(",
        "    'names.bzl',",
        "    'names='+json.encode({",
        "      'name': ctx.name,",
        "      'original_name': ctx.original_name,",
        "    })",
        "  )",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl,",
        "  attrs={'data':attr.string()})");
  }

  @Test
  public void simpleExtension() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "ext.tag(name='foo', data='fu')",
        "ext.tag(name='bar', data='ba')",
        "use_repo(ext, 'foo', 'bar')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {'name':attr.string(),'data':attr.string()})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_repo(name=tag.name,data=tag.data)",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag}, "
            + "os_dependent=True, arch_dependent=True)");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@foo//:names.bzl', foo_names='names')",
        "load('@bar//:data.bzl', bar_data='data')",
        "load('@bar//:names.bzl', bar_names='names')",
        "data = 'foo:'+foo_data+' bar:'+bar_data",
        "names = 'foo:'+foo_names['name']+' bar:'+bar_names['name']",
        "original_names = 'foo:'+foo_names['original_name']+' bar:'+bar_names['original_name']");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("foo:fu bar:ba");
    assertThat(result.get(skyKey).getModule().getGlobal("names"))
        .isEqualTo("foo:+ext+foo bar:+ext+bar");
    assertThat(result.get(skyKey).getModule().getGlobal("original_names"))
        .isEqualTo("foo:foo bar:bar");
  }

  @Test
  public void simpleExtension_nonCanonicalLabel() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='my_module', version = '1.0')",
        "bazel_dep(name='data_repo', version='1.0')",
        "ext1 = use_extension('//:defs.bzl', 'ext')",
        "ext1.tag(name='foo', data='fu')",
        "use_repo(ext1, 'foo')",
        "ext2 = use_extension('@my_module//:defs.bzl', 'ext')",
        "ext2.tag(name='bar', data='ba')",
        "use_repo(ext2, 'bar')",
        "ext3 = use_extension('@//:defs.bzl', 'ext')",
        "ext3.tag(name='quz', data='qu')",
        "use_repo(ext3, 'quz')",
        "ext4 = use_extension('defs.bzl', 'ext')",
        "ext4.tag(name='qor', data='qo')",
        "use_repo(ext4, 'qor')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {'name':attr.string(),'data':attr.string()})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_repo(name=tag.name,data=tag.data)",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@bar//:data.bzl', bar_data='data')",
        "load('@quz//:data.bzl', quz_data='data')",
        "load('@qor//:data.bzl', qor_data='data')",
        "data = 'foo:'+foo_data+' bar:'+bar_data+' quz:'+quz_data+' qor:'+qor_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("foo:fu bar:ba quz:qu qor:qo");
  }

  @Test
  public void simpleExtension_nonCanonicalLabel_repoName() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='my_module', version = '1.0', repo_name='my_name')",
        "bazel_dep(name='data_repo', version='1.0')",
        "ext1 = use_extension('//:defs.bzl', 'ext')",
        "ext1.tag(name='foo', data='fu')",
        "use_repo(ext1, 'foo')",
        "ext2 = use_extension('@my_name//:defs.bzl', 'ext')",
        "ext2.tag(name='bar', data='ba')",
        "use_repo(ext2, 'bar')",
        "ext3 = use_extension('@//:defs.bzl', 'ext')",
        "ext3.tag(name='quz', data='qu')",
        "use_repo(ext3, 'quz')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {'name':attr.string(),'data':attr.string()})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_repo(name=tag.name,data=tag.data)",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@bar//:data.bzl', bar_data='data')",
        "load('@quz//:data.bzl', quz_data='data')",
        "data = 'foo:'+foo_data+' bar:'+bar_data+' quz:'+quz_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("foo:fu bar:ba quz:qu");
  }

  @Test
  public void multipleExtensions_sameName() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo', version='1.0')",
        "first_ext = use_extension('//first_ext:defs.bzl', 'ext')",
        "first_ext.tag(name='foo', data='first_fu')",
        "first_ext.tag(name='bar', data='first_ba')",
        "use_repo(first_ext, first_foo='foo', first_bar='bar')",
        "second_ext = use_extension('//second_ext:defs.bzl', 'ext')",
        "second_ext.tag(name='foo', data='second_fu')",
        "second_ext.tag(name='bar', data='second_ba')",
        "use_repo(second_ext, second_foo='foo', second_bar='bar')");
    scratch.file("first_ext/BUILD");
    scratch.file(
        "first_ext/defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {'name':attr.string(),'data':attr.string()})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_repo(name=tag.name,data=tag.data)",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.file("second_ext/BUILD");
    scratch.file("second_ext/defs.bzl", "load('//first_ext:defs.bzl', _ext = 'ext')", "ext = _ext");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@first_foo//:data.bzl', first_foo_data='data')",
        "load('@first_bar//:data.bzl', first_bar_data='data')",
        "load('@second_foo//:data.bzl', second_foo_data='data')",
        "load('@second_bar//:data.bzl', second_bar_data='data')",
        "data = 'first_foo:'+first_foo_data+' first_bar:'+first_bar_data"
            + "+' second_foo:'+second_foo_data+' second_bar:'+second_bar_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo(
            "first_foo:first_fu first_bar:first_ba second_foo:second_fu " + "second_bar:second_ba");
  }

  @Test
  public void multipleModules() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='root',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='root')",
        "use_repo(ext,'ext_repo')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext_repo//:data.bzl', ext_data='data')", "data=ext_data");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='quux',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='foo@1.0')");
    registry.addModule(
        createModuleKey("bar", "2.0"),
        "module(name='bar',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='quux',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='bar@2.0')");
    registry.addModule(
        createModuleKey("quux", "1.0"),
        "module(name='quux',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='quux@1.0')");
    registry.addModule(
        createModuleKey("quux", "2.0"),
        "module(name='quux',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='quux@2.0')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = ''",
        "  for mod in ctx.modules:",
        "    data_str += mod.name + '@' + mod.version + (' (root): ' if mod.is_root else ': ')",
        "    for tag in mod.tags.tag:",
        "      data_str += tag.data",
        "    data_str += '\\n'",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'data':attr.string()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo(
            "root@1.0 (root): root\nfoo@1.0: foo@1.0\nbar@2.0: bar@2.0\nquux@2.0: quux@2.0\n");
  }

  @Test
  public void multipleModules_devDependency() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='root')",
        "use_repo(ext,'ext_repo')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext_repo//:data.bzl', ext_data='data')", "data=ext_data");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='foo@1.0')");
    registry.addModule(
        createModuleKey("bar", "2.0"),
        "module(name='bar',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='bar@2.0')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = 'modules:'",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + tag.data + ' ' + str(ctx.is_dev_dependency(tag))",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'data':attr.string()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("modules: root True bar@2.0 False");
  }

  @Test
  public void multipleModules_ignoreDevDependency() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='root')",
        "use_repo(ext,'ext_repo')");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='foo@1.0')");
    registry.addModule(
        createModuleKey("bar", "2.0"),
        "module(name='bar',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='bar@2.0')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = 'modules:'",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + tag.data + ' ' + str(ctx.is_dev_dependency(tag))",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'data':attr.string()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(PrecomputedValue.injected(ModuleFileFunction.IGNORE_DEV_DEPS, true)));
    invalidatePackages(false);

    SkyKey skyKey =
        BzlLoadValue.keyForBuild(Label.parseCanonical("@@ext++ext+ext_repo//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("modules: bar@2.0 False");
  }

  @Test
  public void multipleModules_isolatedUsages() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='root',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='root',expect_isolated=False)",
        "use_repo(ext,'ext_repo')",
        "isolated_ext = use_extension('@ext//:defs.bzl','ext',isolate=True)",
        "isolated_ext.tag(data='root_isolated',expect_isolated=True)",
        "use_repo(isolated_ext,isolated_ext_repo='ext_repo')",
        "isolated_dev_ext ="
            + " use_extension('@ext//:defs.bzl','ext',isolate=True,dev_dependency=True)",
        "isolated_dev_ext.tag(data='root_isolated_dev',expect_isolated=True)",
        "use_repo(isolated_dev_ext,isolated_dev_ext_repo='ext_repo')",
        "ext2 = use_extension('@ext//:defs.bzl','ext')",
        "ext2.tag(data='root_2',expect_isolated=False)");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "load('@isolated_ext_repo//:data.bzl', isolated_ext_data='data')",
        "load('@isolated_dev_ext_repo//:data.bzl', isolated_dev_ext_data='data')",
        "data=ext_data",
        "isolated_data=isolated_ext_data",
        "isolated_dev_data=isolated_dev_ext_data");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "isolated_ext = use_extension('@ext//:defs.bzl','ext',isolate=True)",
        "isolated_ext.tag(data='foo@1.0_isolated',expect_isolated=True)",
        "use_repo(isolated_ext,isolated_ext_repo='ext_repo')",
        "isolated_dev_ext ="
            + " use_extension('@ext//:defs.bzl','ext',isolate=True,dev_dependency=True)",
        "isolated_dev_ext.tag(data='foo@1.0_isolated_dev',expect_isolated=True)",
        "use_repo(isolated_dev_ext,isolated_dev_ext_repo='ext_repo')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='foo@1.0',expect_isolated=False)",
        "use_repo(ext,'ext_repo')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("foo+1.0/data.bzl").getPathString(),
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "load('@isolated_ext_repo//:data.bzl', isolated_ext_data='data')",
        "data=ext_data",
        "isolated_data=isolated_ext_data");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = ''",
        "  for mod in ctx.modules:",
        "    data_str += mod.name + '@' + mod.version + (' (root): ' if mod.is_root else ': ')",
        "    for tag in mod.tags.tag:",
        "      data_str += tag.data",
        "      if tag.expect_isolated != ctx.is_isolated:",
        "        fail()",
        "    data_str += '\\n'",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'data':attr.string(),'expect_isolated':attr.bool()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    setBuildLanguageOptions("--experimental_isolated_extension_usages");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("root@1.0 (root): rootroot_2\nfoo@1.0: foo@1.0\n");
    assertThat(result.get(skyKey).getModule().getGlobal("isolated_data"))
        .isEqualTo("root@1.0 (root): root_isolated\n");
    assertThat(result.get(skyKey).getModule().getGlobal("isolated_dev_data"))
        .isEqualTo("root@1.0 (root): root_isolated_dev\n");

    skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("@foo+//:data.bzl"));
    result = SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("root@1.0 (root): rootroot_2\nfoo@1.0: foo@1.0\n");
    assertThat(result.get(skyKey).getModule().getGlobal("isolated_data"))
        .isEqualTo("foo@1.0: foo@1.0_isolated\n");
  }

  @Test
  public void labels_readInModuleExtension() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='//:requirements.txt')",
        "use_repo(ext,'ext_repo')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext_repo//:data.bzl', ext_data='data')", "data=ext_data");
    scratch.file("requirements.txt", "get up at 6am.");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='@bar//:requirements.txt')");
    registry.addModule(createModuleKey("bar", "2.0"), "module(name='bar',version='2.0')");
    scratch.file(moduleRoot.getRelative("bar+2.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("bar+2.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("bar+2.0/requirements.txt").getPathString(), "go to bed at 11pm.");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = 'requirements:'",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + ctx.read(tag.file).strip()",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("requirements: get up at 6am. go to bed at 11pm.");
  }

  @Test
  public void labels_passedOnToRepoRule() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='//:requirements.txt')",
        "use_repo(ext,'ext_repo')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext_repo//:data.bzl', ext_data='data')", "data=ext_data");
    scratch.file("requirements.txt", "get up at 6am.");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='@bar//:requirements.txt')");
    registry.addModule(createModuleKey("bar", "2.0"), "module(name='bar',version='2.0')");
    scratch.file(moduleRoot.getRelative("bar+2.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("bar+2.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("bar+2.0/requirements.txt").getPathString(), "go to bed at 11pm.");

    registry.addModule(createModuleKey("ext", "1.0"), "module(name='ext',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  content = ' '.join([ctx.read(l).strip() for l in ctx.attr.files])",
        "  ctx.file('data.bzl', 'data='+json.encode(content))",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl, attrs={'files':attr.label_list()})",
        "",
        "def _ext_impl(ctx):",
        "  data_files = []",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_files.append(tag.file)",
        "  data_repo(name='ext_repo',files=data_files)",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("get up at 6am. go to bed at 11pm.");

    SkyKey extensionSkyKey =
        SingleExtensionValue.key(
            ModuleExtensionId.create(
                Label.parseCanonicalUnchecked("@@ext+//:defs.bzl"), "ext", Optional.empty()));
    EvaluationResult<SingleExtensionValue> extensionResult =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, extensionSkyKey, false, reporter);
    if (extensionResult.hasError()) {
      throw extensionResult.getError().getException();
    }
    assertThat(
            extensionResult
                .get(extensionSkyKey)
                .lockFileInfo()
                .get()
                .moduleExtension()
                .getRecordedRepoMappingEntries())
        .containsCell(RepositoryName.create("foo+"), "bar", RepositoryName.create("bar+"));
  }

  @Test
  public void labels_fromExtensionGeneratedRepo() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext',version='1.0')",
        "myext = use_extension('//:defs.bzl','myext')",
        "use_repo(myext,'myrepo')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='@myrepo//:requirements.txt')",
        "use_repo(ext,'ext_repo')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext_repo//:data.bzl', ext_data='data')", "data=ext_data");
    scratch.file(
        "defs.bzl",
        "def _myrepo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('requirements.txt', 'get up at 6am.')",
        "myrepo = repository_rule(implementation=_myrepo_impl)",
        "",
        "def _myext_impl(ctx):",
        "  myrepo(name='myrepo')",
        "myext=module_extension(implementation=_myext_impl)");
    scratch.file("requirements.txt", "get up at 6am.");

    registry.addModule(createModuleKey("ext", "1.0"), "module(name='ext',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  content = ' '.join([ctx.read(l).strip() for l in ctx.attr.files])",
        "  ctx.file('data.bzl', 'data='+json.encode(content))",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl, attrs={'files':attr.label_list()})",
        "",
        "def _ext_impl(ctx):",
        "  data_files = []",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_files.append(tag.file)",
        "  data_repo(name='ext_repo',files=data_files)",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("get up at 6am.");
  }

  @Test
  public void labels_constructedInModuleExtension_readInModuleExtension() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag()",
        "use_repo(ext,'ext_repo')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext_repo//:data.bzl', ext_data='data')", "data=ext_data");

    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("foo+1.0/requirements.txt").getPathString(), "get up at 6am.");
    registry.addModule(createModuleKey("bar", "2.0"), "module(name='bar',version='2.0')");
    scratch.file(moduleRoot.getRelative("bar+2.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("bar+2.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("bar+2.0/requirements.txt").getPathString(), "go to bed at 11pm.");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        // The Label() call on the following line should work, using ext.1.0's repo mapping.
        "  data_str = 'requirements: ' + ctx.read(Label('@foo//:requirements.txt')).strip()",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + ctx.read(tag.file).strip()",
        "  data_repo(name='ext_repo',data=data_str)",
        // So should the attr.label default value on the following line.
        "tag=tag_class(attrs={'file':attr.label(default='@bar//:requirements.txt')})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("requirements: get up at 6am. go to bed at 11pm.");
  }

  @Test
  public void labels_constructedInModuleExtensionAsString_passedOnToRepoRule() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "use_repo(ext,'ext_repo')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext_repo//:data.bzl', ext_data='data')", "data=ext_data");

    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("foo+1.0/requirements.txt").getPathString(), "get up at 6am.");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  content = ctx.read(ctx.attr.file).strip()",
        "  ctx.file('data.bzl', 'data='+json.encode(content))",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl, attrs={'file':attr.label()})",
        "",
        "def _ext_impl(ctx):",
        // The label literal on the following line should be interpreted using ext.1.0's repo
        // mapping.
        "  data_repo(name='ext_repo',file='@foo//:requirements.txt')",
        "ext=module_extension(implementation=_ext_impl)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("get up at 6am.");
  }

  /** Tests that a complex-typed attribute (here, string_list_dict) behaves well on a tag. */
  @Test
  public void complexTypedAttribute() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "ext.tag(data={'foo':['val1','val2'],'bar':['val3','val4']})",
        "use_repo(ext, 'foo', 'bar')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {'data':attr.string_list_dict()})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      for key in tag.data:",
        "        data_repo(name=key,data=','.join(tag.data[key]))",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@bar//:data.bzl', bar_data='data')",
        "data = 'foo:'+foo_data+' bar:'+bar_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("foo:val1,val2 bar:val3,val4");
  }

  /**
   * Tests that a complex-typed attribute (here, string_list_dict) behaves well when it has a
   * default value and is omitted in a tag.
   */
  @Test
  public void complexTypedAttribute_default() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "ext.tag()",
        "use_repo(ext, 'foo', 'bar')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {",
        "  'data': attr.string_list_dict(",
        "    default = {'foo':['val1','val2'],'bar':['val3','val4']},",
        ")})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      for key in tag.data:",
        "        data_repo(name=key,data=','.join(tag.data[key]))",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@bar//:data.bzl', bar_data='data')",
        "data = 'foo:'+foo_data+' bar:'+bar_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("foo:val1,val2 bar:val3,val4");
  }

  @Test
  public void generatedReposHaveCorrectMappings() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('//:defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    scratch.file(
        "defs.bzl",
        "def _ext_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', \"\"\"load('@foo//:data.bzl', foo_data='data')",
        "load('@internal//:data.bzl', internal_data='data')",
        "data = 'foo: '+foo_data+' internal: '+internal_data",
        "\"\"\")",
        "ext_repo = repository_rule(implementation=_ext_repo_impl)",
        "",
        "def _internal_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', 'data='+json.encode('internal-stuff'))",
        "internal_repo = repository_rule(implementation=_internal_repo_impl)",
        "",
        "def _ext_impl(ctx):",
        "  internal_repo(name='internal')",
        "  ext_repo(name='ext')",
        "ext=module_extension(implementation=_ext_impl)");

    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/data.bzl").getPathString(), "data = 'foo-stuff'");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("foo: foo-stuff internal: internal-stuff");
  }

  @Test
  public void generatedReposHaveCorrectMappings_moduleOwnRepoName() throws Exception {
    // tests that things work correctly when the module specifies its own repo name (via
    // `module(repo_name=...)`).
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='foo',version='1.0',repo_name='bar')",
        "ext = use_extension('//:defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "data='hello world'");
    scratch.file(
        "defs.bzl",
        "def _ext_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', \"\"\"load('@bar//:data.bzl', bar_data='data')",
        "data = 'bar: '+bar_data",
        "\"\"\")",
        "ext_repo = repository_rule(implementation=_ext_repo_impl)",
        "",
        "ext=module_extension(implementation=lambda ctx: ext_repo(name='ext'))");
    scratch.file(
        "ext_data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data='ext: ' + ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:ext_data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("ext: bar: hello world");
  }

  @Test
  public void generatedReposHaveCorrectMappings_internalRepoWins() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('//:defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    scratch.file(
        "defs.bzl",
        "def _ext_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', \"\"\"load('@foo//:data.bzl', foo_data='data')",
        "data = 'the foo I see is '+foo_data",
        "\"\"\")",
        "ext_repo = repository_rule(implementation=_ext_repo_impl)",
        "",
        "def _internal_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', 'data='+json.encode('inner-foo'))",
        "internal_repo = repository_rule(implementation=_internal_repo_impl)",
        "",
        "def _ext_impl(ctx):",
        "  internal_repo(name='foo')",
        "  ext_repo(name='ext')",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/data.bzl").getPathString(), "data = 'outer-foo'");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("the foo I see is inner-foo");
  }

  @Test
  public void generatedReposHaveCorrectMappings_strictDepsViolation() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel", "ext = use_extension('//:defs.bzl','ext')", "use_repo(ext,'ext')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    scratch.file(
        "defs.bzl",
        "def _ext_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', \"\"\"load('@foo//:data.bzl', 'data')\"\"\")",
        "ext_repo = repository_rule(implementation=_ext_repo_impl)",
        "",
        "def _ext_impl(ctx):",
        "  ext_repo(name='ext')",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains("No repository visible as '@foo' from repository '@@+ext+ext'");
  }

  @Test
  public void wrongModuleExtensionLabel() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel", "ext = use_extension('//foo/defs.bzl','ext')", "use_repo(ext,'ext')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains(
            "Label '//foo/defs.bzl:defs.bzl' is invalid because 'foo/defs.bzl' is not a package");
  }

  @Test
  public void importNonExistentRepo() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "ext = use_extension('//:defs.bzl','ext')",
        "bazel_dep(name='data_repo', version='1.0')",
        "use_repo(ext,my_repo='missing_repo')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='ext',data='void')",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@@+ext+ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains(
            "module extension @@//:defs.bzl%ext does not generate repository"
                + " \"missing_repo\", yet it is imported as \"my_repo\" in the usage at"
                + " /workspace/MODULE.bazel:1:20");
  }

  @Test
  public void invalidAttributeValue() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "ext = use_extension('//:defs.bzl','ext')",
        "bazel_dep(name='data_repo', version='1.0')",
        "use_repo(ext,'ext')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='ext',data=42)",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        """
        ERROR /workspace/defs.bzl:3:12: Traceback (most recent call last):
        \tFile "/workspace/defs.bzl", line 3, column 12, in _ext_impl
        \t\tdata_repo(name='ext',data=42)
        Error: in call to 'data_repo' repo rule with name 'ext', expected value of type 'string' \
        for attribute 'data', but got 42 (int)\
        """);
  }

  @Test
  public void badRepoNameInExtensionImplFunction() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "ext = use_extension('//:defs.bzl','ext')",
        "bazel_dep(name='data_repo', version='1.0')",
        "use_repo(ext,'ext')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='_something',data='void')",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertContainsEvent("invalid user-provided repo name '_something'");
  }

  @Test
  public void nonVisibleLabelInLabelAttr() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel", "ext = use_extension('//:defs.bzl','ext')", "use_repo(ext,'ext')");
    scratch.file(
        "defs.bzl",
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl,",
        "  attrs={'data':attr.label()})",
        "def _ext_impl(ctx):",
        "  data_repo(name='other_repo')",
        "  data_repo(name='ext',data='@not_other_repo//:foo')",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertContainsEvent(
        """
        ERROR /workspace/defs.bzl:8:12: Traceback (most recent call last):
        \tFile "/workspace/defs.bzl", line 8, column 12, in _ext_impl
        \t\tdata_repo(name='ext',data='@not_other_repo//:foo')
        Error: in call to 'data_repo' repo rule with name 'ext', no repository visible as \
        '@not_other_repo' in the extension '@@//:defs.bzl%ext', but referenced by label \
        '@not_other_repo//:foo' in attribute 'data'\
        """);
  }

  @Test
  public void nonVisibleLabelInLabelAttrNonRootModule() throws Exception {
    registry.addModule(
        createModuleKey("ext_module", "1.0"), "module(name='ext_module',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext_module+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext_module+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext_module+1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl,",
        "  attrs={'data':attr.label()})",
        "def _ext_impl(ctx):",
        "  data_repo(name='other_repo')",
        "  data_repo(name='ext',data='@not_other_repo//:foo')",
        "ext = module_extension(implementation=_ext_impl)");

    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name = 'ext_module', version = '1.0')",
        "ext = use_extension('@ext_module//:defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertContainsEvent(
        """
        ERROR /usr/local/google/_blaze_jrluser/FAKEMD5/external/ext_module+/defs.bzl:8:12: \
        Traceback (most recent call last):
        \tFile "/usr/local/google/_blaze_jrluser/FAKEMD5/external/ext_module+/defs.bzl", \
        line 8, column 12, in _ext_impl
        \t\tdata_repo(name='ext',data='@not_other_repo//:foo')
        Error: in call to 'data_repo' repo rule with name 'ext', no repository visible as \
        '@not_other_repo' in the extension '@@ext_module+//:defs.bzl%ext', but referenced by label \
        '@not_other_repo//:foo' in attribute 'data'\
        """);
  }

  @Test
  public void nonVisibleLabelInLabelAttrForwardedFromTag() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "ext = use_extension('//:defs.bzl','ext')",
        "ext.label(label = '@other_repo//:foo')",
        "use_repo(ext,'ext')");
    scratch.file(
        "defs.bzl",
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl,",
        "  attrs={'data':attr.label()})",
        "def _ext_impl(ctx):",
        "  data_repo(name='other_repo')",
        "  data_repo(name='ext',data=ctx.modules[0].tags.label[0].label)",
        "label=tag_class(attrs={'label':attr.label()})",
        "ext = module_extension(",
        "  implementation=_ext_impl,",
        "  tag_classes={'label':label},",
        ")");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    var result = SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .isEqualTo(
            """
            Traceback (most recent call last):
            \tFile "/workspace/MODULE.bazel", line 2, column 10, in <toplevel>
            \t\text.label(label = '@other_repo//:foo')
            Error: in 'label' tag, no repository visible as '@other_repo' \
            to the root module, but referenced by label '@other_repo//:foo' in attribute 'label'\
            """);
  }

  @Test
  public void nonVisibleLabelInLabelListAttr() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel", "ext = use_extension('//:defs.bzl','ext')", "use_repo(ext,'ext')");
    scratch.file(
        "defs.bzl",
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl,",
        "  attrs={'data':attr.label_list()})",
        "def _ext_impl(ctx):",
        "  data_repo(name='other_repo')",
        "  data_repo(name='ext',data=['@not_other_repo//:foo'])",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertContainsEvent(
        """
        ERROR /workspace/defs.bzl:8:12: Traceback (most recent call last):
        \tFile "/workspace/defs.bzl", line 8, column 12, in _ext_impl
        \t\tdata_repo(name='ext',data=['@not_other_repo//:foo'])
        Error: in call to 'data_repo' repo rule with name 'ext', no repository visible as \
        '@not_other_repo' in the extension '@@//:defs.bzl%ext', but referenced by label \
        '@not_other_repo//:foo' in attribute 'data'\
        """);
  }

  @Test
  public void nonVisibleLabelInLabelKeyedStringDictAttr() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel", "ext = use_extension('//:defs.bzl','ext')", "use_repo(ext,'ext')");
    scratch.file(
        "defs.bzl",
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD')",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl,",
        "  attrs={'data':attr.label_keyed_string_dict()})",
        "def _ext_impl(ctx):",
        "  data_repo(name='other_repo')",
        "  data_repo(name='ext',data={'@not_other_repo//:foo':'bar'})",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data=ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertContainsEvent(
        """
        ERROR /workspace/defs.bzl:8:12: Traceback (most recent call last):
        \tFile "/workspace/defs.bzl", line 8, column 12, in _ext_impl
        \t\tdata_repo(name='ext',data={'@not_other_repo//:foo':'bar'})
        Error: in call to 'data_repo' repo rule with name 'ext', no repository visible as \
        '@not_other_repo' in the extension '@@//:defs.bzl%ext', but referenced by label \
        '@not_other_repo//:foo' in attribute 'data'\
        """);
  }

  @Test
  public void nativeExistingRuleIsEmpty() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'ext')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  if not native.existing_rules():",
        "    data_repo(name='ext',data='haha')",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@ext//:data.bzl', ext_data='data')", "data = ext_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("haha");
  }

  @Test
  public void extensionLoadsRepoFromAnotherExtension() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "my_ext = use_extension('@//:defs.bzl', 'my_ext')",
        "use_repo(my_ext, 'summarized_candy')",
        "ext = use_extension('@ext//:defs.bzl', 'ext')",
        "use_repo(ext, 'exposed_candy')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "load('@@ext++ext+candy//:data.bzl', candy='data')",
        "load('@exposed_candy//:data.bzl', exposed_candy='data')",
        "def _ext_impl(ctx):",
        "  data_str = exposed_candy + ' (and ' + candy + ')'",
        "  data_repo(name='summarized_candy', data=data_str)",
        "my_ext=module_extension(implementation=_ext_impl)");

    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@summarized_candy//:data.bzl', data='data')",
        "candy_data = 'candy: ' + data");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='candy', data='cotton candy')",
        "  data_repo(name='exposed_candy', data='lollipops')",
        "ext = module_extension(implementation=_ext_impl)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("candy_data"))
        .isEqualTo("candy: lollipops (and cotton candy)");
  }

  @Test
  public void extensionRepoCtxReadsFromAnotherExtensionRepo() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo',version='1.0')",
        "my_ext = use_extension('@//:defs.bzl', 'my_ext')",
        "use_repo(my_ext, 'candy1')",
        // Repos from this extension (i.e. my_ext2) can still be used if their canonical name is
        // somehow known
        "my_ext2 = use_extension('@//:defs.bzl', 'my_ext2')");

    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_file = ctx.read(Label('@@+my_ext2+candy2//:data.bzl'))",
        "  data_repo(name='candy1',data=data_file)",
        "my_ext=module_extension(implementation=_ext_impl)",
        "def _ext_impl2(ctx):",
        "  data_repo(name='candy2',data='lollipops')",
        "my_ext2=module_extension(implementation=_ext_impl2)");

    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@candy1//:data.bzl', data='data')", "candy_data_file = data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw Objects.requireNonNull(result.getError().getException());
    }
    assertThat(result.get(skyKey).getModule().getGlobal("candy_data_file"))
        .isEqualTo("data = \"lollipops\"");
  }

  @Test
  public void testReportRepoAndBzlCycles_circularExtReposCtxRead() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo',version='1.0')",
        "my_ext = use_extension('@//:defs.bzl', 'my_ext')",
        "use_repo(my_ext, 'candy1')",
        "my_ext2 = use_extension('@//:defs.bzl', 'my_ext2')",
        "use_repo(my_ext2, 'candy2')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  ctx.read(Label('@candy2//:data.bzl'))",
        "  data_repo(name='candy1',data='lollipops')",
        "my_ext=module_extension(implementation=_ext_impl)",
        "def _ext_impl2(ctx):",
        "  ctx.read(Label('@candy1//:data.bzl'))",
        "  data_repo(name='candy2',data='lollipops')",
        "my_ext2=module_extension(implementation=_ext_impl2)");
    scratch.overwriteFile("BUILD");
    invalidatePackages(false);

    SkyKey skyKey =
        PackageIdentifier.create(
            RepositoryName.createUnvalidated("+my_ext+candy1"), PathFragment.EMPTY_FRAGMENT);
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getCycleInfo()).isNotEmpty();
    reporter.removeHandler(failFastHandler);
    cyclesReporter.reportCycles(result.getError().getCycleInfo(), skyKey, reporter);
    assertContainsEvent(
        """
        ERROR <no location>: Circular definition of repositories generated by module extensions\
         and/or .bzl files:
        .-> @@+my_ext+candy1
        |   module extension @@//:defs.bzl%my_ext
        |   @@+my_ext2+candy2
        |   module extension @@//:defs.bzl%my_ext2
        `-- @@+my_ext+candy1\
        """);
  }

  @Test
  public void testReportRepoAndBzlCycles_circularExtReposLoadInDefFile() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo',version='1.0')",
        "my_ext = use_extension('@//:defs.bzl', 'my_ext')",
        "use_repo(my_ext, 'candy1')",
        "my_ext2 = use_extension('@//:defs2.bzl', 'my_ext2')",
        "use_repo(my_ext2, 'candy2')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  ctx.read(Label('@candy2//:data.bzl'))",
        "  data_repo(name='candy1',data='lollipops')",
        "my_ext=module_extension(implementation=_ext_impl)");
    scratch.file(
        "defs2.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "load('@candy1//:data.bzl','data')",
        "def _ext_impl(ctx):",
        "  data_repo(name='candy2',data='lollipops')",
        "my_ext2=module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    invalidatePackages(false);

    SkyKey skyKey =
        PackageIdentifier.create(
            RepositoryName.createUnvalidated("+my_ext+candy1"), PathFragment.create("data.bzl"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getCycleInfo()).isNotEmpty();
    reporter.removeHandler(failFastHandler);
    cyclesReporter.reportCycles(result.getError().getCycleInfo(), skyKey, reporter);
    assertContainsEvent(
        """
        ERROR <no location>: Circular definition of repositories generated by module extensions\
         and/or .bzl files:
        .-> @@+my_ext+candy1
        |   module extension @@//:defs.bzl%my_ext
        |   @@+my_ext2+candy2
        |   module extension @@//:defs2.bzl%my_ext2
        |   //:defs2.bzl
        |   @@+my_ext+candy1//:data.bzl
        `-- @@+my_ext+candy1\
        """);
  }

  @Test
  public void testReportRepoAndBzlCycles_extRepoLoadSelfCycle() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='data_repo',version='1.0')",
        "my_ext = use_extension('@//:defs.bzl', 'my_ext')",
        "use_repo(my_ext, 'candy1')");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "load('@candy1//:data.bzl','data')",
        "def _ext_impl(ctx):",
        "  data_repo(name='candy1',data='lollipops')",
        "my_ext=module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    invalidatePackages(false);

    SkyKey skyKey =
        PackageIdentifier.create(
            RepositoryName.createUnvalidated("+my_ext+candy1"), PathFragment.create("data.bzl"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getCycleInfo()).isNotEmpty();
    reporter.removeHandler(failFastHandler);
    cyclesReporter.reportCycles(result.getError().getCycleInfo(), skyKey, reporter);
    assertContainsEvent(
        """
        ERROR <no location>: Circular definition of repositories generated by module extensions\
         and/or .bzl files:
        .-> @@+my_ext+candy1
        |   module extension @@//:defs.bzl%my_ext
        |   //:defs.bzl
        |   @@+my_ext+candy1//:data.bzl
        `-- @@+my_ext+candy1\
        """);
  }

  @Test
  public void extensionMetadata_exactlyOneArgIsNone() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return ctx.extension_metadata(root_module_direct_deps=['foo'])");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_deps and root_module_direct_dev_deps must both be specified or both be"
            + " unspecified");
  }

  @Test
  public void extensionMetadata_exactlyOneArgIsNoneDev() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return ctx.extension_metadata(root_module_direct_dev_deps=['foo'])");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_deps and root_module_direct_dev_deps must both be specified or both be"
            + " unspecified");
  }

  @Test
  public void extensionMetadata_allUsedTwice() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps='all',root_module_direct_dev_deps='all')");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "if one of root_module_direct_deps and root_module_direct_dev_deps is \"all\", the other"
            + " must be an empty list");
  }

  @Test
  public void extensionMetadata_allAndNone() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return ctx.extension_metadata(root_module_direct_deps='all')");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "if one of root_module_direct_deps and root_module_direct_dev_deps is \"all\", the other"
            + " must be an empty list");
  }

  @Test
  public void extensionMetadata_unsupportedString() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return ctx.extension_metadata(root_module_direct_deps='not_all')");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_deps and root_module_direct_dev_deps must be None, \"all\", or a list"
            + " of strings");
  }

  @Test
  public void extensionMetadata_unsupportedStringDev() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return ctx.extension_metadata(root_module_direct_dev_deps='not_all')");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_deps and root_module_direct_dev_deps must be None, \"all\", or a list"
            + " of strings");
  }

  @Test
  public void extensionMetadata_invalidRepoName() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=['+invalid'],root_module_direct_dev_deps=[])");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "in root_module_direct_deps: invalid user-provided repo name '+invalid': valid names may"
            + " contain only A-Z, a-z, 0-9, '-', '_', '.', and must start with a letter");
  }

  @Test
  public void extensionMetadata_invalidDevRepoName() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_dev_deps=['+invalid'],root_module_direct_deps=[])");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "in root_module_direct_dev_deps: invalid user-provided repo name '+invalid': valid names"
            + " may contain only A-Z, a-z, 0-9, '-', '_', '.', and must start with a letter");
  }

  @Test
  public void extensionMetadata_duplicateRepo() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=['dep','dep'],root_module_direct_dev_deps=[])");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent("in root_module_direct_deps: duplicate entry 'dep'");
  }

  @Test
  public void extensionMetadata_duplicateDevRepo() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=[],root_module_direct_dev_deps=['dep','dep'])");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent("in root_module_direct_dev_deps: duplicate entry 'dep'");
  }

  @Test
  public void extensionMetadata_duplicateRepoAcrossTypes() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=['dep'],root_module_direct_dev_deps=['dep'])");

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "in root_module_direct_dev_deps: entry 'dep' is also in root_module_direct_deps");
  }

  @Test
  public void extensionMetadata_devUsageWithAllDirectNonDevDeps() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=\"all\","
                + "root_module_direct_dev_deps=[])",
            /* devDependency= */ true);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_deps must be empty if the root module contains no usages with "
            + "dev_dependency = False");
  }

  @Test
  public void extensionMetadata_nonDevUsageWithAllDirectDevDeps() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=[],"
                + "root_module_direct_dev_deps=\"all\")",
            /* devDependency= */ false);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_dev_deps must be empty if the root module contains no usages with "
            + "dev_dependency = True");
  }

  @Test
  public void extensionMetadata_devUsageWithDirectNonDevDeps() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=['dep1'],"
                + "root_module_direct_dev_deps=['dep2'])",
            /* devDependency= */ true);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_deps must be empty if the root module contains no usages with "
            + "dev_dependency = False");
  }

  @Test
  public void extensionMetadata_nonDevUsageWithDirectDevDeps() throws Exception {
    var result =
        evaluateSimpleModuleExtension(
            "return"
                + " ctx.extension_metadata(root_module_direct_deps=['dep1'],"
                + "root_module_direct_dev_deps=['dep2'])",
            /* devDependency= */ false);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "root_module_direct_dev_deps must be empty if the root module contains no usages with "
            + "dev_dependency = True");
  }

  @Test
  public void extensionMetadata() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl', 'ext')",
        "use_repo(",
        "  ext,",
        "  'indirect_dep',",
        "  'invalid_dep',",
        "  'dev_as_non_dev_dep',",
        "  my_direct_dep = 'direct_dep',",
        ")",
        "inject_repo(ext, my_data_repo = 'data_repo')",
        "ext_dev = use_extension('@ext//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(",
        "  ext_dev,",
        "  'indirect_dev_dep',",
        "  'invalid_dev_dep',",
        "  'non_dev_as_dev_dep',",
        "  my_direct_dev_dep = 'direct_dev_dep',",
        ")");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@my_direct_dep//:data.bzl', direct_dep_data='data')",
        "data = direct_dep_data");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'indirect_dep')",
        "ext_dev = use_extension('//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'indirect_dev_dep')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='direct_dep')",
        "  data_repo(name='direct_dev_dep')",
        "  data_repo(name='missing_direct_dep')",
        "  data_repo(name='missing_direct_dev_dep')",
        "  data_repo(name='indirect_dep')",
        "  data_repo(name='indirect_dev_dep')",
        "  data_repo(name='dev_as_non_dev_dep')",
        "  data_repo(name='non_dev_as_dev_dep')",
        "  return ctx.extension_metadata(",
        "    root_module_direct_deps=['direct_dep', 'missing_direct_dep', 'non_dev_as_dev_dep'],",
        "    root_module_direct_dev_deps=['direct_dev_dep', 'missing_direct_dev_dep',"
            + " 'dev_as_non_dev_dep'],",
        "  )",
        "ext=module_extension(implementation=_ext_impl)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    // Evaluation fails due to the import of a repository not generated by the extension, but we
    // only want to assert that the warning is emitted.
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();

    assertEventCount(1, eventCollector);
    assertContainsEvent(
        "WARNING /workspace/MODULE.bazel:3:20: The module extension ext defined in @ext//:defs.bzl"
            + " reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Imported, but not created by the extension (will cause the build to fail):\n"
            + "    invalid_dep, invalid_dev_dep\n"
            + "\n"
            + "Not imported, but reported as direct dependencies by the extension (may cause the"
            + " build to fail):\n"
            + "    missing_direct_dep, missing_direct_dev_dep\n"
            + "\n"
            + "Imported as a regular dependency, but reported as a dev dependency by the"
            + " extension (may cause the build to fail when used by other modules):\n"
            + "    dev_as_non_dev_dep\n"
            + "\n"
            + "Imported as a dev dependency, but reported as a regular dependency by the"
            + " extension (may cause the build to fail when used by other modules):\n"
            + "    non_dev_as_dev_dep\n"
            + "\n"
            + "Imported, but reported as indirect dependencies by the extension:\n"
            + "    indirect_dep, indirect_dev_dep\n"
            + "\n"
            + "Fix the use_repo calls by running 'bazel mod tidy'.",
        ImmutableSet.of(EventKind.WARNING));
    SingleExtensionValue evalValue =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"), "ext", Optional.empty())));
    assertThat(evalValue.fixup()).isPresent();
    assertThat(evalValue.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("MODULE.bazel"),
            "use_repo_add ext missing_direct_dep non_dev_as_dev_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext dev_as_non_dev_dep indirect_dep invalid_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_add ext_dev dev_as_non_dev_dep missing_direct_dev_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext_dev indirect_dev_dep invalid_dev_dep non_dev_as_dev_dep");
    assertThat(evalValue.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for @ext//:defs.bzl%ext");
  }

  @Test
  public void extensionMetadata_includes() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "include('//:firstProd.MODULE.bazel')",
        "include('//:second.MODULE.bazel')");
    scratch.file(
        "firstProd.MODULE.bazel",
        "ext = use_extension('@ext//:defs.bzl', 'ext')",
        "use_repo(",
        "  ext,",
        "  'indirect_dep',",
        "  'invalid_dep',",
        "  'dev_as_non_dev_dep',",
        "  my_direct_dep = 'direct_dep',",
        ")",
        "include('//:firstDev.MODULE.bazel')");
    scratch.file(
        "firstDev.MODULE.bazel",
        "ext_dev = use_extension('@ext//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(",
        "  ext_dev,",
        "  'indirect_dev_dep',",
        "  'invalid_dev_dep',",
        "  'non_dev_as_dev_dep',",
        "  my_direct_dev_dep = 'direct_dev_dep',",
        ")");
    scratch.file(
        "second.MODULE.bazel",
        "ext = use_extension('@ext//:defs.bzl', 'ext')",
        "use_repo(ext, 'invalid_dep2')",
        "ext_dev = use_extension('@ext//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'invalid_dev_dep2')");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@my_direct_dep//:data.bzl', direct_dep_data='data')",
        "data = direct_dep_data");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'indirect_dep')",
        "ext_dev = use_extension('//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'indirect_dev_dep')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='direct_dep')",
        "  data_repo(name='direct_dev_dep')",
        "  data_repo(name='missing_direct_dep')",
        "  data_repo(name='missing_direct_dev_dep')",
        "  data_repo(name='indirect_dep')",
        "  data_repo(name='indirect_dev_dep')",
        "  data_repo(name='dev_as_non_dev_dep')",
        "  data_repo(name='non_dev_as_dev_dep')",
        "  return ctx.extension_metadata(",
        "    root_module_direct_deps=['direct_dep', 'missing_direct_dep', 'non_dev_as_dev_dep'],",
        "    root_module_direct_dev_deps=['direct_dev_dep', 'missing_direct_dev_dep',"
            + " 'dev_as_non_dev_dep'],",
        "  )",
        "ext=module_extension(implementation=_ext_impl)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    // Evaluation fails due to the import of a repository not generated by the extension, but we
    // only want to assert that the warning is emitted.
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();

    assertEventCount(1, eventCollector);
    assertContainsEvent(
        """
        WARNING /workspace/firstProd.MODULE.bazel:1:20: The module extension ext defined in \
        @ext//:defs.bzl reported incorrect imports of repositories via use_repo():

        Imported, but not created by the extension (will cause the build to fail):
            invalid_dep, invalid_dep2, invalid_dev_dep, invalid_dev_dep2

        Not imported, but reported as direct dependencies by the extension (may cause the\
         build to fail):
            missing_direct_dep, missing_direct_dev_dep

        Imported as a regular dependency, but reported as a dev dependency by the\
         extension (may cause the build to fail when used by other modules):
            dev_as_non_dev_dep

        Imported as a dev dependency, but reported as a regular dependency by the\
         extension (may cause the build to fail when used by other modules):
            non_dev_as_dev_dep

        Imported, but reported as indirect dependencies by the extension:
            indirect_dep, indirect_dev_dep

        Fix the use_repo calls by running 'bazel mod tidy'.\
        """,
        ImmutableSet.of(EventKind.WARNING));
    SingleExtensionValue evalValue =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"), "ext", Optional.empty())));
    assertThat(evalValue.fixup()).isPresent();
    assertThat(evalValue.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("firstProd.MODULE.bazel"),
            "use_repo_add ext missing_direct_dep non_dev_as_dev_dep",
            PathFragment.create("firstProd.MODULE.bazel"),
            "use_repo_remove ext dev_as_non_dev_dep indirect_dep invalid_dep",
            PathFragment.create("second.MODULE.bazel"),
            "use_repo_remove ext invalid_dep2",
            PathFragment.create("firstDev.MODULE.bazel"),
            "use_repo_add ext_dev dev_as_non_dev_dep missing_direct_dev_dep",
            PathFragment.create("firstDev.MODULE.bazel"),
            "use_repo_remove ext_dev indirect_dev_dep invalid_dev_dep non_dev_as_dev_dep",
            PathFragment.create("second.MODULE.bazel"),
            "use_repo_remove ext_dev invalid_dev_dep2");
    assertThat(evalValue.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for @ext//:defs.bzl%ext");
  }

  @Test
  public void extensionMetadata_all() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl', 'ext')",
        "use_repo(ext, 'direct_dep', 'indirect_dep', 'invalid_dep')",
        "ext_dev = use_extension('@ext//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'direct_dev_dep', 'indirect_dev_dep', 'invalid_dev_dep')");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@direct_dep//:data.bzl', direct_dep_data='data')",
        "data = direct_dep_data");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'indirect_dep')",
        "ext_dev = use_extension('//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'indirect_dev_dep')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='direct_dep')",
        "  data_repo(name='direct_dev_dep')",
        "  data_repo(name='missing_direct_dep')",
        "  data_repo(name='missing_direct_dev_dep')",
        "  data_repo(name='indirect_dep')",
        "  data_repo(name='indirect_dev_dep')",
        "  return ctx.extension_metadata(",
        "    root_module_direct_deps='all',",
        "    root_module_direct_dev_deps=[],",
        "  )",
        "ext=module_extension(implementation=_ext_impl)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .isEqualTo(
            "module extension @@ext+//:defs.bzl%ext does not generate repository "
                + "\"invalid_dep\", yet it is imported as \"invalid_dep\" in the usage at "
                + "/workspace/MODULE.bazel:3:20");

    assertEventCount(1, eventCollector);
    assertContainsEvent(
        "WARNING /workspace/MODULE.bazel:3:20: The module extension ext defined in @ext//:defs.bzl"
            + " reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Imported, but not created by the extension (will cause the build to fail):\n"
            + "    invalid_dep, invalid_dev_dep\n"
            + "\n"
            + "Not imported, but reported as direct dependencies by the extension (may cause the"
            + " build to fail):\n"
            + "    missing_direct_dep, missing_direct_dev_dep\n"
            + "\n"
            + "Imported as a dev dependency, but reported as a regular dependency by the"
            + " extension (may cause the build to fail when used by other modules):\n"
            + "    direct_dev_dep, indirect_dev_dep\n"
            + "\n"
            + "Fix the use_repo calls by running 'bazel mod tidy'.",
        ImmutableSet.of(EventKind.WARNING));
    SingleExtensionValue evalValue =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"), "ext", Optional.empty())));
    assertThat(evalValue.fixup()).isPresent();
    assertThat(evalValue.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("MODULE.bazel"),
            "use_repo_add ext direct_dev_dep indirect_dev_dep missing_direct_dep"
                + " missing_direct_dev_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext invalid_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext_dev direct_dev_dep indirect_dev_dep invalid_dev_dep");
    assertThat(evalValue.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for @ext//:defs.bzl%ext");
  }

  @Test
  public void extensionMetadata_allDev() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl', 'ext')",
        "use_repo(ext, 'direct_dep', 'indirect_dep', 'invalid_dep')",
        "ext_dev = use_extension('@ext//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'direct_dev_dep', 'indirect_dev_dep', 'invalid_dev_dep')");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@direct_dep//:data.bzl', direct_dep_data='data')",
        "data = direct_dep_data");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'indirect_dep')",
        "ext_dev = use_extension('//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'indirect_dev_dep')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='direct_dep')",
        "  data_repo(name='direct_dev_dep')",
        "  data_repo(name='missing_direct_dep')",
        "  data_repo(name='missing_direct_dev_dep')",
        "  data_repo(name='indirect_dep')",
        "  data_repo(name='indirect_dev_dep')",
        "  return ctx.extension_metadata(",
        "    root_module_direct_deps=[],",
        "    root_module_direct_dev_deps='all',",
        "  )",
        "ext=module_extension(implementation=_ext_impl)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    // Evaluation fails due to the import of a repository not generated by the extension, but we
    // only want to assert that the warning is emitted.
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .isEqualTo(
            "module extension @@ext+//:defs.bzl%ext does not generate repository "
                + "\"invalid_dep\", yet it is imported as \"invalid_dep\" in the usage at "
                + "/workspace/MODULE.bazel:3:20");

    assertEventCount(1, eventCollector);
    assertContainsEvent(
        "WARNING /workspace/MODULE.bazel:3:20: The module extension ext defined in @ext//:defs.bzl"
            + " reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Imported, but not created by the extension (will cause the build to fail):\n"
            + "    invalid_dep, invalid_dev_dep\n"
            + "\n"
            + "Not imported, but reported as direct dependencies by the extension (may cause the"
            + " build to fail):\n"
            + "    missing_direct_dep, missing_direct_dev_dep\n"
            + "\n"
            + "Imported as a regular dependency, but reported as a dev dependency by the"
            + " extension (may cause the build to fail when used by other modules):\n"
            + "    direct_dep, indirect_dep\n"
            + "\n"
            + "Fix the use_repo calls by running 'bazel mod tidy'.",
        ImmutableSet.of(EventKind.WARNING));
    SingleExtensionValue evalValue =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"), "ext", Optional.empty())));
    assertThat(evalValue.fixup()).isPresent();
    assertThat(evalValue.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext direct_dep indirect_dep invalid_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_add ext_dev direct_dep indirect_dep missing_direct_dep"
                + " missing_direct_dev_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext_dev invalid_dev_dep");
    assertThat(evalValue.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for @ext//:defs.bzl%ext");
  }

  @Test
  public void extensionMetadata_noRootUsage() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.overwriteFile("BUILD");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'indirect_dep')",
        "ext_dev = use_extension('//:defs.bzl', 'ext', dev_dependency = True)",
        "use_repo(ext_dev, 'indirect_dev_dep')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='direct_dep')",
        "  data_repo(name='direct_dev_dep')",
        "  data_repo(name='missing_direct_dep')",
        "  data_repo(name='missing_direct_dev_dep')",
        "  data_repo(name='indirect_dep', data='indirect_dep_data')",
        "  data_repo(name='indirect_dev_dep')",
        "  return ctx.extension_metadata(",
        "    root_module_direct_deps='all',",
        "    root_module_direct_dev_deps=[],",
        "  )",
        "ext=module_extension(implementation=_ext_impl)");
    scratch.file(
        moduleRoot.getRelative("ext+1.0/data.bzl").getPathString(),
        "load('@indirect_dep//:data.bzl', indirect_dep_data='data')",
        "data = indirect_dep_data");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("@ext+//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("indirect_dep_data");

    assertEventCount(0, eventCollector);
    SingleExtensionValue evalValue =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"), "ext", Optional.empty())));
    assertThat(evalValue.fixup()).isEmpty();
  }

  @Test
  public void extensionMetadata_isolated() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext1 = use_extension('@ext//:defs.bzl', 'ext', isolate = True)",
        "use_repo(",
        "  ext1,",
        "  'indirect_dep',",
        ")",
        "ext2 = use_extension('@ext//:defs.bzl', 'ext', isolate = True)",
        "use_repo(",
        "  ext2,",
        "  'direct_dep',",
        ")");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@direct_dep//:data.bzl', data_1='data')",
        "load('@indirect_dep//:data.bzl', data_2='data')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'indirect_dep')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='direct_dep')",
        "  data_repo(name='missing_direct_dep')",
        "  data_repo(name='indirect_dep')",
        "  return ctx.extension_metadata(",
        "    root_module_direct_deps=['direct_dep', 'missing_direct_dep'],",
        "    root_module_direct_dev_deps=[],",
        "  )",
        "ext=module_extension(implementation=_ext_impl)");
    setBuildLanguageOptions("--experimental_isolated_extension_usages");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    // Evaluation fails due to the import of a repository not generated by the extension, but we
    // only want to assert that the warning is emitted.
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }

    assertEventCount(2, eventCollector);
    assertContainsEvent(
        """
        WARNING /workspace/MODULE.bazel:3:21: The module extension ext defined in @ext//:defs.bzl\
         reported incorrect imports of repositories via use_repo():

        Not imported, but reported as direct dependencies by the extension (may cause the\
         build to fail):
            direct_dep, missing_direct_dep

        Imported, but reported as indirect dependencies by the extension:
            indirect_dep

        Fix the use_repo calls by running 'bazel mod tidy'.\
        """,
        ImmutableSet.of(EventKind.WARNING));
    assertContainsEvent(
        """
        WARNING /workspace/MODULE.bazel:8:21: The module extension ext defined in @ext//:defs.bzl\
         reported incorrect imports of repositories via use_repo():

        Not imported, but reported as direct dependencies by the extension (may cause the\
         build to fail):
            missing_direct_dep

        Fix the use_repo calls by running 'bazel mod tidy'.\
        """,
        ImmutableSet.of(EventKind.WARNING));
    SingleExtensionValue ext1Value =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"),
                            "ext",
                            Optional.of(
                                ModuleExtensionId.IsolationKey.create(ModuleKey.ROOT, "ext1")))));
    assertThat(ext1Value.fixup()).isPresent();
    assertThat(ext1Value.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("MODULE.bazel"),
            "use_repo_add ext1 direct_dep missing_direct_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext1 indirect_dep");
    assertThat(ext1Value.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for isolated usage 'ext1' of @ext//:defs.bzl%ext");
    SingleExtensionValue ext2Value =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"),
                            "ext",
                            Optional.of(
                                ModuleExtensionId.IsolationKey.create(ModuleKey.ROOT, "ext2")))));
    assertThat(ext2Value.fixup()).isPresent();
    assertThat(ext2Value.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("MODULE.bazel"), "use_repo_add ext2 missing_direct_dep");
    assertThat(ext2Value.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for isolated usage 'ext2' of @ext//:defs.bzl%ext");
  }

  @Test
  public void extensionMetadata_isolatedDev() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='ext', version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext1 = use_extension('@ext//:defs.bzl', 'ext', isolate = True, dev_dependency = True)",
        "use_repo(",
        "  ext1,",
        "  'indirect_dep',",
        ")",
        "ext2 = use_extension('@ext//:defs.bzl', 'ext', isolate = True, dev_dependency = True)",
        "use_repo(",
        "  ext2,",
        "  'direct_dep',",
        ")");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@direct_dep//:data.bzl', data_1='data')",
        "load('@indirect_dep//:data.bzl', data_2='data')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'indirect_dep')");
    scratch.file(moduleRoot.getRelative("ext+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("ext+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("ext+1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_repo(name='direct_dep')",
        "  data_repo(name='missing_direct_dep')",
        "  data_repo(name='indirect_dep')",
        "  return ctx.extension_metadata(",
        "    root_module_direct_deps=[],",
        "    root_module_direct_dev_deps=['direct_dep', 'missing_direct_dep'],",
        "  )",
        "ext=module_extension(implementation=_ext_impl)");
    setBuildLanguageOptions("--experimental_isolated_extension_usages");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    // Evaluation fails due to the import of a repository not generated by the extension, but we
    // only want to assert that the warning is emitted.
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }

    assertEventCount(2, eventCollector);
    assertContainsEvent(
        "WARNING /workspace/MODULE.bazel:3:21: The module extension ext defined in @ext//:defs.bzl"
            + " reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Not imported, but reported as direct dependencies by the extension (may cause the"
            + " build to fail):\n"
            + "    direct_dep, missing_direct_dep\n"
            + "\n"
            + "Imported, but reported as indirect dependencies by the extension:\n"
            + "    indirect_dep\n"
            + "\n"
            + "Fix the use_repo calls by running 'bazel mod tidy'.",
        ImmutableSet.of(EventKind.WARNING));
    assertContainsEvent(
        "WARNING /workspace/MODULE.bazel:8:21: The module extension ext defined in @ext//:defs.bzl"
            + " reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Not imported, but reported as direct dependencies by the extension (may cause the"
            + " build to fail):\n"
            + "    missing_direct_dep\n"
            + "\n"
            + "Fix the use_repo calls by running 'bazel mod tidy'.",
        ImmutableSet.of(EventKind.WARNING));
    SingleExtensionValue ext1Value =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"),
                            "ext",
                            Optional.of(
                                ModuleExtensionId.IsolationKey.create(ModuleKey.ROOT, "ext1")))));
    assertThat(ext1Value.fixup()).isPresent();
    assertThat(ext1Value.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("MODULE.bazel"),
            "use_repo_add ext1 direct_dep missing_direct_dep",
            PathFragment.create("MODULE.bazel"),
            "use_repo_remove ext1 indirect_dep");
    assertThat(ext1Value.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for isolated usage 'ext1' of @ext//:defs.bzl%ext");
    SingleExtensionValue ext2Value =
        (SingleExtensionValue)
            skyframeExecutor
                .getEvaluator()
                .getDoneValues()
                .get(
                    SingleExtensionValue.evalKey(
                        ModuleExtensionId.create(
                            Label.parseCanonical("@@ext+//:defs.bzl"),
                            "ext",
                            Optional.of(
                                ModuleExtensionId.IsolationKey.create(ModuleKey.ROOT, "ext2")))));
    assertThat(ext2Value.fixup()).isPresent();
    assertThat(ext2Value.fixup().get().moduleFilePathToBuildozerCommands())
        .containsExactly(
            PathFragment.create("MODULE.bazel"), "use_repo_add ext2 missing_direct_dep");
    assertThat(ext2Value.fixup().get().getSuccessMessage())
        .isEqualTo("Updated use_repo calls for isolated usage 'ext2' of @ext//:defs.bzl%ext");
  }

  private EvaluationResult<SingleExtensionValue> evaluateSimpleModuleExtension(
      String returnStatement) throws Exception {
    return evaluateSimpleModuleExtension(returnStatement, /* devDependency= */ false);
  }

  private EvaluationResult<SingleExtensionValue> evaluateSimpleModuleExtension(
      String returnStatement, boolean devDependency) throws Exception {
    String devDependencyStr = devDependency ? "True" : "False";
    scratch.overwriteFile(
        "MODULE.bazel",
        String.format(
            "ext = use_extension('//:defs.bzl', 'ext', dev_dependency = %s)", devDependencyStr));
    scratch.file(
        "defs.bzl",
        "repo = repository_rule(lambda ctx: True)",
        "def _ext_impl(ctx):",
        "  repo(name = 'dep1')",
        "  repo(name = 'dep2')",
        "  " + returnStatement,
        "ext = module_extension(implementation=_ext_impl)");
    scratch.overwriteFile("BUILD");
    invalidatePackages(false);

    ModuleExtensionId extensionId =
        ModuleExtensionId.create(Label.parseCanonical("//:defs.bzl"), "ext", Optional.empty());
    reporter.removeHandler(failFastHandler);
    invalidatePackages(false);
    return SkyframeExecutorTestUtils.evaluate(
        skyframeExecutor, SingleExtensionValue.key(extensionId), false, reporter);
  }

  @Test
  public void isDevDependency_usages() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='root',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')",
        "ext1 = use_extension('//:defs.bzl','ext1')",
        "use_repo(ext1,ext1_repo='ext_repo')",
        "ext2 = use_extension('//:defs.bzl','ext2',dev_dependency=True)",
        "use_repo(ext2,ext2_repo='ext_repo')",
        "ext3a = use_extension('//:defs.bzl','ext3')",
        "use_repo(ext3a,ext3_repo='ext_repo')",
        "ext3b = use_extension('//:defs.bzl','ext3',dev_dependency=True)");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@ext1_repo//:data.bzl', _ext1_data='data')",
        "load('@ext2_repo//:data.bzl', _ext2_data='data')",
        "load('@ext3_repo//:data.bzl', _ext3_data='data')",
        "ext1_data=_ext1_data",
        "ext2_data=_ext2_data",
        "ext3_data=_ext3_data");
    scratch.file(
        "defs.bzl",
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(id,ctx):",
        "  data_str = id + ': ' + str(ctx.root_module_has_non_dev_dependency)",
        "  data_repo(name='ext_repo',data=data_str)",
        "ext1=module_extension(implementation=lambda ctx: _ext_impl('ext1', ctx))",
        "ext2=module_extension(implementation=lambda ctx: _ext_impl('ext2', ctx))",
        "ext3=module_extension(implementation=lambda ctx: _ext_impl('ext3', ctx))");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("ext1_data")).isEqualTo("ext1: True");
    assertThat(result.get(skyKey).getModule().getGlobal("ext2_data")).isEqualTo("ext2: False");
    assertThat(result.get(skyKey).getModule().getGlobal("ext3_data")).isEqualTo("ext3: True");
  }

  @Test
  public void printAndFailOnTag() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel", "ext = use_extension('//:defs.bzl', 'ext')", "ext.foo()", "ext.foo()");
    scratch.file(
        "defs.bzl",
        "repo = repository_rule(lambda ctx: True)",
        "def _ext_impl(ctx):",
        "  tag1 = ctx.modules[0].tags.foo[0]",
        "  tag2 = ctx.modules[0].tags.foo[1]",
        "  print('Conflict between', tag1, 'and', tag2)",
        "  fail('Fatal conflict between', tag1, 'and', tag2)",
        "foo = tag_class()",
        "ext = module_extension(implementation=_ext_impl,tag_classes={'foo':foo})");
    scratch.overwriteFile("BUILD");
    invalidatePackages(false);

    ModuleExtensionId extensionId =
        ModuleExtensionId.create(Label.parseCanonical("//:defs.bzl"), "ext", Optional.empty());
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        SkyframeExecutorTestUtils.evaluate(
            skyframeExecutor, SingleExtensionValue.key(extensionId), false, reporter);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "Fatal conflict between 'foo' tag at /workspace/MODULE.bazel:2:8 and 'foo' tag at "
            + "/workspace/MODULE.bazel:3:8",
        ImmutableSet.of(EventKind.ERROR));
    assertContainsEvent(
        "Conflict between 'foo' tag at /workspace/MODULE.bazel:2:8 and 'foo' tag at"
            + " /workspace/MODULE.bazel:3:8",
        ImmutableSet.of(EventKind.DEBUG));
  }

  @Test
  public void innate() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='foo',version='1.0')",
        "data_repo = use_repo_rule('@foo//:repo.bzl', 'data_repo')",
        "data_repo(name='data1', data='get up at 6am.')");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@data1//:data.bzl', self_data='data')",
        "load('@data1//:names.bzl', self_names='names')",
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@foo//:names.bzl', foo_names='names')",
        "data=self_data+' '+foo_data",
        "names=self_names['name']+' '+foo_names['name']",
        "original_names=self_names['original_name']+' '+foo_names['original_name']");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "data_repo = use_repo_rule('//:repo.bzl', 'data_repo')",
        "data_repo(name='data2', data='go to bed at 11pm.')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("foo+1.0/data.bzl").getPathString(),
        "load('@data2//:data.bzl',repo_data='data')",
        "data=repo_data");
    scratch.file(
        moduleRoot.getRelative("foo+1.0/names.bzl").getPathString(),
        "load('@data2//:names.bzl',repo_names='names')",
        "names=repo_names");
    scratch.file(
        moduleRoot.getRelative("foo+1.0/repo.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD.bazel')",
        "  ctx.file('data.bzl', 'data='+json.encode(ctx.attr.data))",
        "  ctx.file(",
        "    'names.bzl',",
        "    'names='+json.encode({",
        "      'name': ctx.name,",
        "      'original_name': ctx.original_name,",
        "    })",
        "  )",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl, attrs={'data':attr.string()})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("get up at 6am. go to bed at 11pm.");
    assertThat(result.get(skyKey).getModule().getGlobal("names"))
        .isEqualTo("+data_repo+data1 foo++data_repo+data2");
    assertThat(result.get(skyKey).getModule().getGlobal("original_names")).isEqualTo("data1 data2");
  }

  @Test
  public void innate_repoRuleDependencies() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='foo',version='1.0')",
        "gen_data_repo = use_repo_rule('@foo//:repo.bzl', 'gen_data_repo')",
        "gen_data_repo(name='gen_data_repo')",
        "data_repo = use_repo_rule('@gen_data_repo//:repo.bzl', 'data_repo')",
        "data_repo(name='data', data='get up at 6am.')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@data//:data.bzl', _data='data')", "data=_data");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='data_repo', version='1.0')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("foo+1.0/repo.bzl").getPathString(),
        "def _gen_data_repo_impl(ctx):",
        "  ctx.file('BUILD.bazel')",
        "  ctx.file('repo.bzl', '''",
        "load('{data_repo_defs}', _data_repo='data_repo')",
        "data_repo=_data_repo",
        "'''.format(data_repo_defs = Label('@data_repo//:defs.bzl')))",
        "gen_data_repo = repository_rule(implementation=_gen_data_repo_impl)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      if (result.getError().getException() != null) {
        throw result.getError().getException();
      }
      throw new IllegalStateException("Cycle: " + result.getError().getCycleInfo());
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("get up at 6am.");
  }

  @Test
  public void innate_noSuchRepoRule() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "data_repo = use_repo_rule('//:repo.bzl', 'data_repo')",
        "data_repo(name='data', data='get up at 6am.')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@data//:data.bzl', self_data='data')", "data=self_data");
    scratch.file("repo.bzl", "# not a repo rule", "def data_repo(name):", "    pass");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains(
            "//:repo.bzl exports a value called data_repo of type function, yet a repository_rule"
                + " is requested at /workspace/MODULE.bazel");
  }

  @Test
  public void innate_noSuchValue() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "data_repo = use_repo_rule('//:repo.bzl', 'data_repo')",
        "data_repo(name='data', data='get up at 6am.')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@data//:data.bzl', self_data='data')", "data=self_data");
    scratch.file("repo.bzl", "");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains(
            "//:repo.bzl does not export a repository_rule called data_repo, yet its use is"
                + " requested at /workspace/MODULE.bazel");
  }

  @Test
  public void innate_noSuchValueIfPrivate() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "data_repo = use_repo_rule('//:repo.bzl', '_data_repo')",
        "data_repo(name='data', data='get up at 6am.')");
    scratch.overwriteFile("BUILD");
    scratch.file("data.bzl", "load('@data//:data.bzl', self_data='data')", "data=self_data");
    scratch.file("repo.bzl", "_data_repo = repository_rule(lambda _: None)");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains(
            "//:repo.bzl does not export a repository_rule called _data_repo, yet its use is"
                + " requested at /workspace/MODULE.bazel");
  }

  @Test
  public void innate_invalidAttributeValue() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "bazel_dep(name='foo',version='1.0')",
        "data_repo = use_repo_rule('@foo//:repo.bzl', 'data_repo')",
        "data_repo(name='data', data=5)");
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        "load('@data//:data.bzl', self_data='data')",
        "load('@foo//:data.bzl', foo_data='data')",
        "data=self_data+' '+foo_data");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "data_repo = use_repo_rule('//:repo.bzl', 'data_repo')",
        "data_repo(name='data', data='go to bed at 11pm.')");
    scratch.file(moduleRoot.getRelative("foo+1.0/REPO.bazel").getPathString());
    scratch.file(moduleRoot.getRelative("foo+1.0/BUILD").getPathString());
    scratch.file(
        moduleRoot.getRelative("foo+1.0/data.bzl").getPathString(),
        "load('@data//:data.bzl',repo_data='data')",
        "data=repo_data");
    scratch.file(
        moduleRoot.getRelative("foo+1.0/repo.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('BUILD.bazel')",
        "  ctx.file('data.bzl', 'data='+json.encode(ctx.attr.data))",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl, attrs={'data':attr.string()})");
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        """
        ERROR /workspace/MODULE.bazel:3:10: Traceback (most recent call last):
        \tFile "/workspace/MODULE.bazel", line 3, column 10, in <toplevel>
        \t\tdata_repo(name='data', data=5)
        Error: in call to 'data_repo' repo rule with name 'data', expected value of type 'string' \
        for attribute 'data', but got 5 (int)\
        """);
  }

  @Test
  public void extensionRepoMapping() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        """
        bazel_dep(name = "data_repo", version = "1.0")
        ext = use_extension("//:defs.bzl","ext")
        use_repo(ext, real_foo = "foo", real_bar = "bar")
        other_ext = use_extension("//:defs.bzl", "other_ext")
        use_repo(other_ext, foo = "other_foo", bar = "other_bar")
        """);
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        """
        load("@real_foo//:list.bzl", _foo_list = "list")
        load("@real_bar//:list.bzl", _bar_list = "list")
        foo_list = _foo_list
        bar_list = _bar_list
        """);
    scratch.file(
        "defs.bzl",
        """
        load("@data_repo//:defs.bzl", "data_repo")
        def _list_repo_impl(ctx):
          ctx.file("BUILD")
          labels = [str(Label(l)) for l in ctx.attr.labels]
          names = [str(Label(n)) for n in ctx.attr.names]
          ctx.file("list.bzl", "list = " + repr(labels + names))
        list_repo = repository_rule(
          implementation = _list_repo_impl,
          attrs = {
            "names": attr.string_list(),
            "labels": attr.label_list(),
          },
        )
        def _ext_impl(ctx):
          labels = [
            "@foo//:target1",
            "@bar//:target2",
            Label("@foo//:target3"),
            Label("@bar//:target4"),
          ]
          list_repo(
            name = "foo",
            labels = labels,
            names = [
              "@foo",
              "@bar",
            ],
          )

          # Modify the list passed to "foo" to verify that it is not retained by
          # reference.
          labels[0] = "@foo//:target5"
          labels[1] = "@bar//:target6"
          labels[2] = Label("@foo//:target7")
          labels[3] = Label("@bar//:target8")
          list_repo(
            name = "bar",
            labels = labels,
            names = [
              "@foo",
              "@bar",
            ],
          )
        ext = module_extension(implementation = _ext_impl)
        def _other_ext_impl(ctx):
          data_repo(name="other_foo",data="other_foo_data")
          data_repo(name="other_bar",data="other_bar_data")
        other_ext=module_extension(implementation=_other_ext_impl)
        """);
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat((List<?>) result.get(skyKey).getModule().getGlobal("foo_list"))
        .containsExactly(
            "@@+ext+foo//:target1",
            "@@+ext+bar//:target2",
            "@@+other_ext+other_foo//:target3",
            "@@+other_ext+other_bar//:target4",
            "@@+other_ext+other_foo//:foo",
            "@@+other_ext+other_bar//:bar")
        .inOrder();
    assertThat((List<?>) result.get(skyKey).getModule().getGlobal("bar_list"))
        .containsExactly(
            "@@+ext+foo//:target5",
            "@@+ext+bar//:target6",
            "@@+other_ext+other_foo//:target7",
            "@@+other_ext+other_bar//:target8",
            "@@+other_ext+other_foo//:foo",
            "@@+other_ext+other_bar//:bar")
        .inOrder();
  }

  @Test
  public void overrideRepo_override() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        """
        bazel_dep(name = "data_repo", version = "1.0")
        ext = use_extension("//:defs.bzl","ext")
        use_repo(ext, "bar", module_foo = "foo")
        data_repo = use_repo_rule("@data_repo//:defs.bzl", "data_repo")
        data_repo(name = "override", data = "overridden_data")
        override_repo(ext, foo = "override")
        """);
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        """
        load("@bar//:list.bzl", _bar_list = "list")
        load("@override//:data.bzl", _override_data = "data")
        load("@module_foo//:data.bzl", _foo_data = "data")
        bar_list = _bar_list
        foo_data = _foo_data
        override_data = _override_data
        """);
    scratch.file(
        "defs.bzl",
        """
        load("@data_repo//:defs.bzl", "data_repo")
        def _list_repo_impl(ctx):
          ctx.file("BUILD")
          labels = [str(Label(l)) for l in ctx.attr.labels]
          labels += [str(Label("@module_foo//:target3"))]
          ctx.file("list.bzl", "list = " + repr(labels) + " + [str(Label('@foo//:target4'))]")
        list_repo = repository_rule(
          implementation = _list_repo_impl,
          attrs = {
            "labels": attr.label_list(),
          },
        )
        def _fail_repo_impl(ctx):
          fail("This rule should not be evaluated")
        fail_repo = repository_rule(implementation = _fail_repo_impl)
        def _ext_impl(ctx):
          fail_repo(name = "foo")
          list_repo(
            name = "bar",
            labels = [
              # lazy extension implementation function repository mapping
              "@foo//:target1",
              # module repo repository mapping
              "@module_foo//:target2",
            ],
          )
        ext = module_extension(implementation = _ext_impl)
        """);
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat((List<?>) result.get(skyKey).getModule().getGlobal("bar_list"))
        .containsExactly(
            "@@+data_repo+override//:target1",
            "@@+data_repo+override//:target2",
            "@@+data_repo+override//:target3",
            "@@+data_repo+override//:target4")
        .inOrder();
    Object overrideData = result.get(skyKey).getModule().getGlobal("override_data");
    assertThat(overrideData).isInstanceOf(String.class);
    assertThat(overrideData).isEqualTo("overridden_data");
    Object fooData = result.get(skyKey).getModule().getGlobal("foo_data");
    assertThat(fooData).isSameInstanceAs(overrideData);
  }

  @Test
  public void overrideRepo_override_onNonExistentRepoFails() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        """
        bazel_dep(name = "data_repo", version = "1.0")
        ext = use_extension("//:defs.bzl","ext")
        use_repo(ext, "bar", module_foo = "foo")
        data_repo = use_repo_rule("@data_repo//:defs.bzl", "data_repo")
        data_repo(name = "foo", data = "overridden_data")
        override_repo(ext, "foo")
        """);
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        """
        load("@bar//:list.bzl", _bar_list = "list")
        load("@foo//:data.bzl", _foo_data = "data")
        bar_list = _bar_list
        foo_data = _foo_data
        """);
    scratch.file(
        "defs.bzl",
        """
        load("@data_repo//:defs.bzl", "data_repo")
        def _list_repo_impl(ctx):
          ctx.file("BUILD")
          labels = [str(Label(l)) for l in ctx.attr.labels]
          labels += [str(Label("@foo//:target3"))]
          ctx.file("list.bzl", "list = " + repr(labels) + " + [str(Label('@foo//:target4'))]")
        list_repo = repository_rule(
          implementation = _list_repo_impl,
          attrs = {
            "labels": attr.label_list(),
          },
        )
        def _ext_impl(ctx):
          list_repo(
            name = "bar",
            labels = [
              # lazy extension implementation function repository mapping
              "@foo//:target1",
              # module repo repository mapping
              Label("@foo//:target2"),
            ],
          )
        ext = module_extension(implementation = _ext_impl)
        """);
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .isEqualTo(
            "module extension @@//:defs.bzl%ext does not generate repository \"foo\","
                + " yet it is overridden via override_repo() at /workspace/MODULE.bazel:6:14. Use"
                + " inject_repo() instead to inject a new repository.");
  }

  @Test
  public void overrideRepo_inject() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        """
        bazel_dep(name = "data_repo", version = "1.0")
        ext = use_extension("//:defs.bzl","ext")
        use_repo(ext, "bar")
        data_repo = use_repo_rule("@data_repo//:defs.bzl", "data_repo")
        data_repo(name = "foo", data = "overridden_data")
        inject_repo(ext, "foo")
        """);
    scratch.overwriteFile("BUILD");
    scratch.file(
        "data.bzl",
        """
        load("@bar//:list.bzl", _bar_list = "list")
        load("@foo//:data.bzl", _foo_data = "data")
        bar_list = _bar_list
        foo_data = _foo_data
        """);
    scratch.file(
        "defs.bzl",
        """
        load("@data_repo//:defs.bzl", "data_repo")
        def _list_repo_impl(ctx):
          ctx.file("BUILD")
          labels = [str(Label(l)) for l in ctx.attr.labels]
          labels += [str(Label("@foo//:target3"))]
          ctx.file("list.bzl", "list = " + repr(labels) + " + [str(Label('@foo//:target4'))]")
        list_repo = repository_rule(
          implementation = _list_repo_impl,
          attrs = {
            "labels": attr.label_list(),
          },
        )
        def _ext_impl(ctx):
          list_repo(
            name = "bar",
            labels = [
              # lazy extension implementation function repository mapping
              "@foo//:target1",
              # module repo repository mapping
              Label("@foo//:target2"),
            ],
          )
        ext = module_extension(implementation = _ext_impl)
        """);
    invalidatePackages(false);

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseCanonical("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, skyKey, false, reporter);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat((List<?>) result.get(skyKey).getModule().getGlobal("bar_list"))
        .containsExactly(
            "@@+data_repo+foo//:target1",
            "@@+data_repo+foo//:target2",
            "@@+data_repo+foo//:target3",
            "@@+data_repo+foo//:target4")
        .inOrder();
    Object fooData = result.get(skyKey).getModule().getGlobal("foo_data");
    assertThat(fooData).isInstanceOf(String.class);
    assertThat(fooData).isEqualTo("overridden_data");
  }
}
