// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link CompileOneDependencyTransformer}.
 */
@RunWith(JUnit4.class)
public class CompileOneDependencyTransformerTest extends PackageLoadingTestCase {

  private static Set<Label> targetsToLabels(Iterable<Target> targets) {
    return AbstractTargetPatternEvaluatorTest.targetsToLabels(targets);
  }

  private TargetPatternEvaluator parser;
  private CompileOneDependencyTransformer transformer;

  @Before
  public final void createTransformer() throws Exception {
    parser = skyframeExecutor.newTargetPatternEvaluator();
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE,
                Optional.<RootedPath>absent())));
    transformer = new CompileOneDependencyTransformer(getPackageManager());
  }

  private void writeSimpleExample() throws IOException {
    scratch.file(
        "foo/rule.bzl",
        "def _impl(ctx):",
        "  ctx.actions.do_nothing(mnemonic='Mnemonic')",
        "  return struct()",
        "crule_without_srcs = rule(",
        "  _impl,",
        "  attrs = { ",
        "    'hdrs': attr.label_list(flags = ['DIRECT_COMPILE_TIME_INPUT']),",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    scratch.file(
        "foo/BUILD",
        "load(':rule.bzl', 'crule_without_srcs')",
        "cc_library(name = 'foo1', srcs = [ 'foo1.cc' ], hdrs = [ 'foo1.h' ])",
        "crule_without_srcs(name = 'foo2', hdrs = [ 'foo2.h' ])",
        "exports_files(['baz/bang'])");
    scratch.file("foo/bar/BUILD",
                "cc_library(name = 'bar1', alwayslink = 1)",
                "cc_library(name = 'bar2')",
                "exports_files(['wiz/bang', 'wiz/all', 'baz', 'baz/bang', 'undeclared.h'])");
  }

  private static Set<Label> labels(String... labelStrings) throws LabelSyntaxException {
    Set<Label> labels = new HashSet<>();
    for (String labelString : labelStrings) {
      labels.add(Label.parseAbsolute(labelString, ImmutableMap.of()));
    }
    return labels;
  }

  private static ResolvedTargets<Target> parseTargetPatternList(
      TargetPatternEvaluator parser, Reporter reporter,
      List<String> targetPatterns, FilteringPolicy policy,
      boolean keepGoing) throws Exception {
    return parser.parseTargetPatternList(
        PathFragment.EMPTY_FRAGMENT, reporter, targetPatterns, policy, keepGoing);
  }

  private ResolvedTargets<Target> parseCompileOneDep(String... patterns) throws Exception {
    ResolvedTargets<Target> result = parseTargetPatternList(parser, reporter,
        Arrays.asList(patterns), FilteringPolicies.NO_FILTER, false);
    return transformer.transformCompileOneDependency(reporter, result);
  }

  private Set<Label> parseListCompileOneDep(String... patterns) throws Exception {
    return targetsToLabels(getFailFast(parseCompileOneDep(patterns)));
  }

  private Set<Label> parseListCompileOneDepRelative(String... patterns)
      throws TargetParsingException, IOException, InterruptedException {
    Path foo = scratch.dir("foo");
    TargetPatternEvaluator fooOffsetParser = skyframeExecutor.newTargetPatternEvaluator();
    ResolvedTargets<Target> result;
    try {
      result = fooOffsetParser.parseTargetPatternList(
          foo.relativeTo(rootDirectory),
          reporter,
          Arrays.asList(patterns),
          FilteringPolicies.NO_FILTER, false);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
    result = transformer.transformCompileOneDependency(reporter, result);
    return targetsToLabels(getFailFast(result));
  }

  private static Set<Target> getFailFast(ResolvedTargets<Target> result) {
    assertThat(result.hasError()).isFalse();
    return result.getTargets();
  }

  @Test
  public void testCompileOneDep() throws Exception {
    writeSimpleExample();
    assertThat(parseListCompileOneDep("foo/foo1.cc"))
        .containsExactlyElementsIn(labels("@//foo:foo1"));
    assertThat(parseListCompileOneDep("foo/foo1.h"))
        .containsExactlyElementsIn(labels("@//foo:foo1"));
    assertThat(parseListCompileOneDep("foo:foo1.cc"))
        .containsExactlyElementsIn(labels("@//foo:foo1"));
    assertThat(parseListCompileOneDep("//foo:foo1.cc"))
        .containsExactlyElementsIn(labels("@//foo:foo1"));
    assertThat(parseListCompileOneDepRelative("//foo:foo1.cc"))
        .containsExactlyElementsIn(labels("@//foo:foo1"));
    assertThat(parseListCompileOneDepRelative(":foo1.cc"))
        .containsExactlyElementsIn(labels("@//foo:foo1"));
    assertThat(parseListCompileOneDepRelative("foo1.cc"))
        .containsExactlyElementsIn(labels("@//foo:foo1"));
    assertThat(parseListCompileOneDep("foo/foo2.h"))
        .containsExactlyElementsIn(labels("@//foo:foo2"));
  }

  /**
   * Regression test for bug:
   * "--compile_one_dependency should report error for missing input".
   */
  @Test
  public void testCompileOneDepOnMissingFile() throws Exception {
    writeSimpleExample();
    try {
      parseCompileOneDep("//foo:missing.cc");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "no such target '//foo:missing.cc': target 'missing.cc' not declared in package "
                  + "'foo' defined by /workspace/foo/BUILD");
    }

    // Also, try a valid input file which has no dependent rules in its package.
    try {
      parseCompileOneDep("//foo:baz/bang");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("Couldn't find dependency on target '//foo:baz/bang'");
    }

    // Try a header that is in a package but where no cc_library explicitly lists it.
    try {
      parseCompileOneDep("//foo/bar:undeclared.h");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("Couldn't find dependency on target '//foo/bar:undeclared.h'");
    }
  }

  @Test
  public void testCompileOneDepOnNonSourceTarget() throws Exception {
    writeSimpleExample();
    try {
      parseCompileOneDep("//foo:foo1");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("--compile_one_dependency target '//foo:foo1' must be a file");
    }
  }

  @Test
  public void testCompileOneDepOnTwoTargets() throws Exception {
    scratch.file(
        "recursive/BUILD",
        "cc_library(name = 'x', srcs = ['foox.cc'])",
        "cc_library(name = 'y', srcs = ['fooy.cc'])");
    assertThat(parseListCompileOneDep("//recursive:foox.cc", "//recursive:fooy.cc"))
        .containsExactlyElementsIn(labels("//recursive:x", "//recursive:y"));

  }

  /**
   * Regression test for bug:
   * "--compile_one_dependency should not crash in the presence of mutually recursive targets"
   */
  @Test
  public void testCompileOneDepOnRecursiveTarget() throws Exception {
    scratch.file(
        "recursive/BUILD",
        "filegroup(name = 'x', srcs = ['foo.cc', ':y'])",
        "filegroup(name = 'y', srcs = [':x'])",
        "cc_library(name = 'foo', srcs = [':y'])");
    assertThat(parseListCompileOneDep("//recursive:foo.cc"))
        .containsExactlyElementsIn(labels("//recursive:foo"));
  }

  @Test
  public void testCompileOneDepOnRecursiveNotFoundTarget() throws Exception {
    scratch.file("recursive/BUILD",
        "filegroup(name = 'x', srcs = [':y'])",
        "filegroup(name = 'y', srcs = [':x'])",
        "exports_files(['foo'])");

    try {
      parseCompileOneDep("//recursive:foo");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("Couldn't find dependency on target '//recursive:foo'");
    }
  }

  @Test
  public void testCompileOneDepOnDeepRecursiveTarget() throws Exception {
    scratch.file(
        "recursive/BUILD",
        "filegroup(name = 'x', srcs = ['foox.cc', ':y'])",
        "filegroup(name = 'y', srcs = ['fooy.cc', ':z'])",
        "filegroup(name = 'z', srcs = ['fooz.cc', ':x'])",
        "cc_library(name = 'cc', srcs = [':x'])");

    Set<Label> result =
        parseListCompileOneDep("//recursive:foox.cc", "//recursive:fooy.cc", "//recursive:fooy.cc");
    assertThat(result).containsExactlyElementsIn(labels("//recursive:cc"));
  }

  @Test
  public void testCompileOneDepOnCrossPackageRecursiveTarget() throws Exception {
    scratch.file(
        "recursive/BUILD",
        "filegroup(name = 'x', srcs = ['foo.cc', '//recursivetoo:x'])",
        "cc_library(name = 'cc', srcs = [':x'])");

    scratch.file(
        "recursivetoo/BUILD",
        "filegroup(name = 'x', srcs = ['foo.cc', '//recursive:x'])",
        "cc_library(name = 'cc', srcs = [':x'])");
    assertThat(parseListCompileOneDep("//recursive:foo.cc", "//recursivetoo:foo.cc"))
        .containsExactlyElementsIn(labels("//recursive:cc", "//recursivetoo:cc"));

  }

  /**
   * Tests that when multiple rules match the target, the one that appears first in the BUILD
   * file is chosen.
   */
  @Test
  public void testRuleChoiceOrdering() throws Exception {
    scratch.file("a/BUILD",
                "cc_library(name = 'foo_lib', srcs = [ 'file.cc' ])",
                "cc_library(name = 'bar_lib', srcs = [ 'file.cc' ])");
    scratch.file("b/BUILD",
                "cc_library(name = 'bar_lib', srcs = [ 'file.cc' ])",
                "cc_library(name = 'foo_lib', srcs = [ 'file.cc' ])");

    assertThat(parseListCompileOneDep("a/file.cc"))
        .containsExactlyElementsIn(labels("//a:foo_lib"));
    assertThat(parseListCompileOneDep("b/file.cc"))
        .containsExactlyElementsIn(labels("//b:bar_lib"));
  }

  /**
   * Tests that when multiple rule match a target, language-specific rules take precedence.
   */
  @Test
  public void testRuleChoiceLanguagePreferences() throws Exception {
    String srcs = "srcs = [ 'a.cc', 'a.c', 'a.h', 'a.java', 'a.py', 'a.txt' ])";
    scratch.file("a/BUILD",
                "genrule(name = 'gen_rule', cmd = '', outs = [ 'out' ], " + srcs,
                "cc_library(name = 'cc_rule', " + srcs,
                "java_library(name = 'java_rule', " + srcs,
                "py_library(name = 'py_rule', " + srcs);

    assertThat(parseListCompileOneDep("a/a.cc")).containsExactlyElementsIn(labels("//a:cc_rule"));
    assertThat(parseListCompileOneDep("a/a.c")).containsExactlyElementsIn(labels("//a:cc_rule"));
    assertThat(parseListCompileOneDep("a/a.h")).containsExactlyElementsIn(labels("//a:cc_rule"));
    assertThat(parseListCompileOneDep("a/a.java"))
        .containsExactlyElementsIn(labels("//a:java_rule"));
    assertThat(parseListCompileOneDep("a/a.py")).containsExactlyElementsIn(labels("//a:py_rule"));
    assertThat(parseListCompileOneDep("a/a.txt")).containsExactlyElementsIn(labels("//a:gen_rule"));
  }

  @Test
  public void testGeneratedFile() throws Exception {
    scratch.file("a/BUILD",
                "genrule(name = 'gen_rule', cmd = '', outs = [ 'out.cc' ])",
                "cc_library(name = 'cc', srcs = ['out.cc'])");
    assertThat(parseListCompileOneDep("a/out.cc")).containsExactlyElementsIn(labels("//a:cc"));
  }

  @Test
  public void testGeneratedFileDepOnGenerator() throws Exception {
    scratch.file(
        "a/BUILD",
        "genrule(name = 'gen_rule', cmd = '', outs = [ 'out.cc' ])",
        "cc_library(name = 'cc', srcs = [':gen_rule'])");
    assertThat(parseListCompileOneDep("a/out.cc")).containsExactlyElementsIn(labels("//a:cc"));
  }

  @Test
  public void testHdrsFilegroup() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(name = 'headers', srcs = ['a.h'])",
        "cc_library(name = 'cc', hdrs = [':headers'], srcs = ['a.cc'])");
    assertThat(parseListCompileOneDep("a/a.h")).containsExactlyElementsIn(labels("//a:cc"));
  }

  @Test
  public void testConfigurableSrcs() throws Exception {
    // TODO(djasper): We currently flatten the contents of configurable attributes, which might not
    // always do the right thing. In this situation it is actually good as compiling "foo_select"
    // at least has the chance to actually be a correct --compile_one_dependency choice for both
    // "b.cc" and "c.cc". However, if it also contained "a.cc" it might be better to still always
    // choose "foo_always".
    scratch.file(
        "a/BUILD",
        "config_setting(name = 'a', values = {'define': 'foo=a'})",
        "cc_library(name = 'foo_select', srcs = select({':a': ['b.cc'], ':b': ['c.cc']}))",
        "cc_library(name = 'foo_always', srcs = ['a.cc'])");
    assertThat(parseListCompileOneDep("a/a.cc"))
        .containsExactlyElementsIn(labels("//a:foo_always"));
    assertThat(parseListCompileOneDep("a/b.cc"))
        .containsExactlyElementsIn(labels("//a:foo_select"));
    assertThat(parseListCompileOneDep("a/c.cc"))
        .containsExactlyElementsIn(labels("//a:foo_select"));
  }

  @Test
  public void testConfigurableCopts() throws Exception {
    // This configurable attribute doesn't preclude accurately knowing the srcs.
    scratch.file("a/BUILD",
                "config_setting(name = 'a', values = {'define': 'foo=a'})",
                "cc_library(name = 'foo_select', srcs = ['a.cc'],",
                "    copts = select({':a': ['-DA'], ':b': ['-DB']}))",
                "cc_library(name = 'foo_always', srcs = ['a.cc'])");
    assertThat(parseListCompileOneDep("a/a.cc"))
        .containsExactlyElementsIn(labels("//a:foo_select"));
  }

  @Test
  public void testFallBackToHeaderOnlyLibrary() throws Exception {
    scratch.file(
        "a/BUILD",
        "cc_library(name = 'h', hdrs = ['a.h'], features = ['parse_headers'])");
    assertThat(parseListCompileOneDep("a/a.h")).containsExactlyElementsIn(labels("//a:h"));
  }
}
