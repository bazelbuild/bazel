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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertDoesNotContainEvent;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.GcFinalization;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.util.MockGenruleSupport;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.lang.ref.WeakReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Miscellaneous tests of the analysis phase.  (Sometimes it's easier to
 * express these in terms of the BuildTool than of the BuildView because the
 * latter's class interface is quite complex.)
 */
@RunWith(JUnit4.class)
public class MiscAnalysisTest extends GoogleBuildIntegrationTestCase {

  // Regression test for bug #1324794, "Replay of errors in --cache_analysis
  // mode is not working".
  // Note that the cache_analysis flag has been deleted, as it is now standard app behavior.
  @Test
  public void testWarningsAreReplayedEvenWithAnalysisCaching() throws Exception {
    AnalysisMock.get().pySupport().setup(mockToolsConfig);
    write("y/BUILD",
        "py_library(name='y', srcs=[':c'])",
        "genrule(name='c', outs=['c.out'], cmd=':')");
    addOptions("--nobuild");

    buildTarget("//y");
    events.assertContainsWarning("in srcs attribute of py_library rule //y:y: rule '//y:c' " +
        "does not produce any Python source files");

    events.clear();

    buildTarget("//y");
    events.assertContainsWarning("in srcs attribute of py_library rule //y:y: rule '//y:c' " +
        "does not produce any Python source files");
  }

  @Test
  public void testDeprecatedTargetOnCommandLine() throws Exception {
    write("raspberry/BUILD",
        "sh_library(name='raspberry', srcs=['raspberry.sh'], deprecation='rotten')");
    addOptions("--nobuild");
    buildTarget("//raspberry:raspberry");
    events.assertContainsWarning("target '//raspberry:raspberry' is deprecated: rotten");
  }

  // Regression test for http://b/12465751: "IllegalStateException in ParallelEvaluator".
  @Test
  public void testShBinaryTwoSrcs() throws Exception {
    write("sh/BUILD", "sh_test(name = 'double', srcs = ['a','b'])");
    addOptions("--nobuild");

    assertThrows(Exception.class, () -> buildTarget("//sh:double"));
    events.assertContainsError("you must specify exactly one file in 'srcs'");
  }

  // Note that the cache_analysis flag has been deleted, as it is now standard app behavior.
  @Test
  public void testAnalysisCachingAndKeepGoing() throws Exception {
    write(
        "fruit/BUILD",
        "cc_library(name='apple', deps=[':banana'])",
        "cc_library(name='banana', deps=[':cherry'])",
        "cc_library(name='cherry', deps=[':durian__hdrs__'])",
        "genrule(name='durian', outs=['durian.out'], cmd=':')");
    addOptions("--nobuild", "--keep_going");

    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//fruit:apple"));
    assertThat(e).hasMessageThat().contains("command succeeded");
    events.assertContainsError(
        "in deps attribute of cc_library rule //fruit:cherry: "
            + "target '//fruit:durian__hdrs__' does not exist");

    events.clear();
    e = assertThrows(BuildFailedException.class, () -> buildTarget("//fruit:apple"));
    assertThat(e).hasMessageThat().contains("command succeeded");
    events.assertContainsError(
        "in deps attribute of cc_library rule //fruit:cherry: "
            + "target '//fruit:durian__hdrs__' does not exist");
  }

  // Note that the cache_analysis flag has been deleted, as it is now standard app behavior.
  @Test
  public void testErrorsAreReplayedEvenWithAnalysisCaching() throws Exception {
    write(
        "fruit/BUILD",
        "cc_library(name='apple', deps=[':banana__hdrs__'])",
        "genrule(name='banana', outs=['banana.out'], cmd=':')");
    addOptions("--nobuild");

    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//fruit:apple"));
    events.assertContainsError(
        "in deps attribute of cc_library rule //fruit:apple: "
            + "target '//fruit:banana__hdrs__' does not exist");

    events.clear();
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//fruit:apple"));
    events.assertContainsError(
        "in deps attribute of cc_library rule //fruit:apple: "
            + "target '//fruit:banana__hdrs__' does not exist");
  }

  // Regression test for bug #1332987, "--experimental_deps_ok switch doesn't
  // propagate transitively".
  @Test
  public void testExperimentalDepsOkInheritedByHostConfiguration() throws Exception {
    MockGenruleSupport.setup(mockToolsConfig);

    write("x/BUILD",
          "genrule(name='x', outs=['x.out'], tools=[':y'], cmd=':')",
          "genrule(name='y', srcs=['//experimental/x'], outs=['y.out'], cmd=':')");
    write("experimental/x/BUILD",
          "exports_files(['x'])");
    addOptions("--nobuild", "--experimental_deps_ok");

    buildTarget("//x"); // no error, just a warning.
    events.assertContainsWarning("non-experimental target '//x:y' depends "
                        + "on experimental target '//experimental/x:x' "
                        + "(ignored due to --experimental_deps_ok; do not submit)");
  }

  /**
   * Regression Test for bug 3071861. Checks that the {@code --define} command
   * line option is also applied to the host configuration.
   */
  @Test
  public void testHostDefine() throws Exception {
    MockGenruleSupport.setup(mockToolsConfig);

    write("x/BUILD",
        "vardef('CMD', 'false');",
        "genrule(name='foo', outs=['foo.out'], tools=[':bar'], cmd='touch $@ && ' + varref('CMD'))",
        "genrule(name='bar', outs=['bar.out'], cmd='touch $@ && ' + varref('CMD'))");

    addOptions("--define=CMD=true");
    buildTarget("//x:foo");
  }

  @Test
  public void testBuildAllAndParsingError() throws Exception {
    write("pkg/BUILD",
          "java_binary(",
          "name = \"foo\",",
          "  syntax error here",
          ")");

    addOptions("--nobuild");

    Exception e = assertThrows(Exception.class, () -> buildTarget("//pkg:all"));
    events.assertContainsError("syntax error at 'error'");
    assertPkgErrorMsg(e);
  }

  @Test
  public void testDiscardAnalysisCache() throws Exception {
    write("sh/BUILD",
        "sh_library(name = 'sh', srcs = [], deps = [':dep'])",
        "sh_library(name = 'dep', srcs = [])"
        );
    buildTarget("//sh:sh");
    // We test with dep because target completion middleman actions keep references to the
    // top-level configured targets.
    ConfiguredTarget ct = getConfiguredTarget("//sh:dep");
    addOptions("--discard_analysis_cache");
    buildTarget("//sh:sh");
    addOptions("--nodiscard_analysis_cache");
    buildTarget("//sh:sh");
    // Configured target was replaced.
    ConfiguredTarget newCt = getConfiguredTarget("//sh:dep");
    assertThat(newCt).isNotSameInstanceAs(ct);
    WeakReference<ConfiguredTarget> ref = new WeakReference<>(newCt);
    newCt = null;
    addOptions("--discard_analysis_cache");
    buildTarget("//sh:sh");
    GcFinalization.awaitClear(ref);
  }

  @Test
  public void testDiscardAnalysisCacheWithError() throws Exception {
    write("x/BUILD",
          "genrule(name='x', outs=['x.out'], srcs=['//experimental/x'], cmd=':')");
    write("experimental/x/BUILD", "exports_files(['x'])");
    write("y/BUILD", "sh_library(name='y')");
    addOptions("--discard_analysis_cache", "--keep_going");
    EventCollector collector = new EventCollector(EventKind.STDERR);
    events.addHandler(collector);
    assertThrows(BuildFailedException.class, () -> buildTarget("//x:x", "//y:y"));
    events.assertContainsError("depends on experimental target");
    MoreAsserts.assertContainsEvent(collector, "Target //y:y up-to-date", EventKind.STDERR);
  }

  @Test
  public void testBuildAllAndEvaluationError() throws Exception {
    write("pkg/BUILD",
          "java_binary(",
          "    name = \"foo\",",
          "    srcs = unknown_value,",
          ")");

    addOptions("--nobuild");

    Exception e = assertThrows(Exception.class, () -> buildTarget("//pkg:all"));
    events.assertContainsError("name 'unknown_value' is not defined");
    assertPkgErrorMsg(e);
  }

  private static void assertPkgErrorMsg(Exception e) {
    assertThat(e).hasMessageThat().containsMatch("[pP]ackage.*contains errors");
  }

  @Test
  public void testNoTestTargetsFoundMessageForBuildCommand() throws Exception {
    write("pkg/BUILD");
    for (String option : ImmutableList.of("", "--nobuild", "--noanalyze")) {
      resetOptions();
      addOptions(option);
      buildTarget("//pkg:all");
      assertDoesNotContainEvent(events.infos(), "test target");
    }
  }
}
