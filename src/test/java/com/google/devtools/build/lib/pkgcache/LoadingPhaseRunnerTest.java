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

import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.MoreCollectors;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PatternExpandingError;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SkyframeExecutor#loadTargetPatterns}. */
@RunWith(JUnit4.class)
public class LoadingPhaseRunnerTest {

  private static final ImmutableList<Logger> loggers = ImmutableList.of(
      Logger.getLogger(BuildView.class.getName()));
  static {
    for (Logger logger : loggers) {
      logger.setLevel(Level.OFF);
    }
  }

  private LoadingPhaseTester tester;

  @Before
  public final void createLoadingPhaseTester() throws Exception  {
    tester = new LoadingPhaseTester();
  }

  private List<Label> getLabels(String... labels) throws Exception {
    List<Label> result = new ArrayList<>();
    for (String label : labels) {
      result.add(Label.parseAbsoluteUnchecked(label));
    }
    return result;
  }

  private void assertCircularSymlinksDuringTargetParsing(String targetPattern) throws Exception {
    try {
      tester.load(targetPattern);
      fail();
    } catch (TargetParsingException e) {
      // Expected.
      tester.assertContainsError("circular symlinks detected");
    }
  }

  private TargetPatternPhaseValue assertNoErrors(TargetPatternPhaseValue loadingResult) {
    assertThat(loadingResult.hasError()).isFalse();
    assertThat(loadingResult.hasPostExpansionError()).isFalse();
    tester.assertNoEvents();
    return loadingResult;
  }

  @Test
  public void testSmoke() throws Exception {
    tester.addFile("base/BUILD",
        "filegroup(name = 'hello', srcs = ['foo.txt'])");
    TargetPatternPhaseValue loadingResult = assertNoErrors(tester.load("//base:hello"));
    assertThat(loadingResult.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//base:hello"));
    assertThat(loadingResult.getTestsToRunLabels()).isNull();
  }

  @Test
  public void testNonExistentPackage() throws Exception {
    TargetPatternPhaseValue loadingResult = tester.loadKeepGoing("//base:missing");
    assertThat(loadingResult.hasError()).isTrue();
    assertThat(loadingResult.hasPostExpansionError()).isFalse();
    assertThat(loadingResult.getTargetLabels()).isEmpty();
    assertThat(loadingResult.getTestsToRunLabels()).isNull();
    tester.assertContainsError("Skipping '//base:missing': no such package 'base'");
    tester.assertContainsWarning("Target pattern parsing failed.");
    PatternExpandingError err = tester.findPostOnce(PatternExpandingError.class);
    assertThat(err.getPattern()).containsExactly("//base:missing");
  }

  @Test
  public void testNonExistentPackageWithKeepGoing() throws Exception {
    tester.addFile("base/BUILD",
        "filegroup(name = 'hello', srcs = ['foo.txt'])");
    tester.loadKeepGoing("//base:hello", "//base:missing");
    PatternExpandingError err = tester.findPostOnce(PatternExpandingError.class);
    assertThat(err.getPattern()).containsExactly("//base:missing");
    TargetParsingCompleteEvent event = tester.findPostOnce(TargetParsingCompleteEvent.class);
    assertThat(event.getOriginalTargetPattern()).containsExactly("//base:hello", "//base:missing");
    assertThat(event.getFailedTargetPatterns()).containsExactly("//base:missing");
  }

  @Test
  public void testNonExistentPackageWithoutKeepGoing() throws Exception {
    try {
      tester.load("//does/not/exist");
      fail();
    } catch (TargetParsingException expected) {
    }
    PatternExpandingError err = tester.findPostOnce(PatternExpandingError.class);
    assertThat(err.getPattern()).containsExactly("//does/not/exist");
  }

  @Test
  public void testNonExistentTarget() throws Exception {
    tester.addFile("base/BUILD");
    TargetPatternPhaseValue loadingResult = tester.loadKeepGoing("//base:missing");
    assertThat(loadingResult.hasError()).isTrue();
    assertThat(loadingResult.hasPostExpansionError()).isFalse();
    assertThat(loadingResult.getTargetLabels()).isEmpty();
    assertThat(loadingResult.getTestsToRunLabels()).isNull();
    tester.assertContainsError("Skipping '//base:missing': no such target '//base:missing'");
    tester.assertContainsWarning("Target pattern parsing failed.");
    PatternExpandingError err = tester.findPostOnce(PatternExpandingError.class);
    assertThat(err.getPattern()).containsExactly("//base:missing");
  }

  @Test
  public void testMistypedTarget() throws Exception {
    try {
      tester.load("foo//bar:missing");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e).hasMessageThat().contains(
          "invalid target format 'foo//bar:missing': "
          + "invalid package name 'foo//bar': "
          + "package names may not contain '//' path separators");
    }
    ParsingFailedEvent err = tester.findPostOnce(ParsingFailedEvent.class);
    assertThat(err.getPattern()).isEqualTo("foo//bar:missing");
  }

  @Test
  public void testEmptyTarget() throws Exception {
    try {
      tester.load("");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e).hasMessageThat().contains("the empty string is not a valid target");
    }
  }

  @Test
  public void testMistypedTargetKeepGoing() throws Exception {
    TargetPatternPhaseValue result = tester.loadKeepGoing("foo//bar:missing");
    assertThat(result.hasError()).isTrue();
    tester.assertContainsError(
          "invalid target format 'foo//bar:missing': "
          + "invalid package name 'foo//bar': "
          + "package names may not contain '//' path separators");
    ParsingFailedEvent err = tester.findPostOnce(ParsingFailedEvent.class);
    assertThat(err.getPattern()).isEqualTo("foo//bar:missing");
  }

  @Test
  public void testBadTargetPatternWithTest() throws Exception {
    tester.addFile("base/BUILD");
    TargetPatternPhaseValue loadingResult = tester.loadTestsKeepGoing("//base:missing");
    assertThat(loadingResult.hasError()).isTrue();
    assertThat(loadingResult.hasPostExpansionError()).isFalse();
    assertThat(loadingResult.getTargetLabels()).isEmpty();
    assertThat(loadingResult.getTestsToRunLabels()).isEmpty();
    tester.assertContainsError("Skipping '//base:missing': no such target '//base:missing'");
    tester.assertContainsWarning("Target pattern parsing failed.");
  }

  @Test
  public void testManualTarget() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD", "cc_library(name = 'my_lib', srcs = ['lib.cc'], tags = ['manual'])");
    TargetPatternPhaseValue loadingResult = assertNoErrors(tester.load("//cc:all"));
    assertThat(loadingResult.getTargetLabels()).containsExactlyElementsIn(getLabels());

    // Explicitly specified on the command line.
    loadingResult = assertNoErrors(tester.load("//cc:my_lib"));
    assertThat(loadingResult.getTargetLabels()).containsExactlyElementsIn(getLabels("//cc:my_lib"));
  }

  @Test
  public void testConfigSettingTarget() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("config/BUILD",
        "cc_library(name = 'somelib', srcs = [ 'somelib.cc' ], hdrs = [ 'somelib.h' ])",
        "config_setting(name = 'configa', values = { 'define': 'foo=a' })",
        "config_setting(name = 'configb', values = { 'define': 'foo=b' })");
    TargetPatternPhaseValue result = assertNoErrors(tester.load("//config:all"));
    assertThat(result.getTargetLabels()).containsExactlyElementsIn(getLabels("//config:somelib"));

    // Explicitly specified on the command line.
    result = assertNoErrors(tester.load("//config:configa"));
    assertThat(result.getTargetLabels()).containsExactlyElementsIn(getLabels("//config:configa"));
  }

  @Test
  public void testNegativeTestDoesNotShowUpAtAll() throws Exception {
    tester.addFile("my_test/BUILD",
        "sh_test(name = 'my_test', srcs = ['test.cc'])");
    assertNoErrors(tester.loadTests("-//my_test"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).isEmpty();
  }

  @Test
  public void testNegativeTargetDoesNotShowUpAtAll() throws Exception {
    tester.addFile("my_library/BUILD",
        "cc_library(name = 'my_library', srcs = ['test.cc'])");
    assertNoErrors(tester.loadTests("-//my_library"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).isEmpty();
  }

  private void writeBuildFilesForTestFiltering() throws Exception {
    tester.addFile("tests/BUILD",
        "sh_test(name = 't1', srcs = ['pass.sh'], size= 'small', local=1)",
        "sh_test(name = 't2', srcs = ['pass.sh'], size = 'medium')",
        "sh_test(name = 't3', srcs = ['pass.sh'], tags = ['manual', 'local'])");
  }

  @Test
  public void testTestFiltering() throws Exception {
    writeBuildFilesForTestFiltering();
    TargetPatternPhaseValue loadingResult = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(loadingResult.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t2"));
    assertThat(loadingResult.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t2"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).isEmpty();
  }

  @Test
  public void testTestFilteringIncludingManual() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--build_manual_tests");
    TargetPatternPhaseValue loadingResult = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(loadingResult.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t2", "//tests:t3"));
    assertThat(loadingResult.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t2"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).isEmpty();
  }

  @Test
  public void testTestFilteringBuildTestsOnly() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--build_tests_only");
    TargetPatternPhaseValue result = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(result.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t2"));
    assertThat(result.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t2"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).isEmpty();
  }

  @Test
  public void testTestFilteringSize() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--test_size_filters=small");
    TargetPatternPhaseValue result = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(result.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t2"));
    assertThat(result.getTestsToRunLabels()).containsExactlyElementsIn(getLabels("//tests:t1"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).isEmpty();
  }

  @Test
  public void testTestFilteringSizeAndBuildTestsOnly() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--test_size_filters=small", "--build_tests_only");
    TargetPatternPhaseValue result = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(result.getTargetLabels()).containsExactlyElementsIn(getLabels("//tests:t1"));
    assertThat(result.getTestsToRunLabels()).containsExactlyElementsIn(getLabels("//tests:t1"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getLabels("//tests:t2"));
  }

  @Test
  public void testTestFilteringLocalAndBuildTestsOnly() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--test_tag_filters=local", "--build_tests_only");
    TargetPatternPhaseValue result = assertNoErrors(tester.loadTests("//tests:all", "//tests:t3"));
    assertThat(result.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t3"));
    assertThat(result.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//tests:t1", "//tests:t3"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getLabels("//tests:t2"));
  }

  @Test
  public void testTestSuiteExpansion() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD",
        "cc_test(name = 'my_test', srcs = ['test.cc'])",
        "test_suite(name = 'tests', tests = [':my_test'])");
    TargetPatternPhaseValue loadingResult = assertNoErrors(tester.loadTests("//cc:tests"));
    assertThat(loadingResult.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//cc:my_test"));
    assertThat(loadingResult.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//cc:my_test"));
    assertThat(tester.getOriginalTargets())
        .containsExactlyElementsIn(getLabels("//cc:tests", "//cc:my_test"));
    assertThat(tester.getTestSuiteTargets())
        .containsExactly(Label.parseAbsoluteUnchecked("//cc:tests"));
  }

  @Test
  public void testTestSuiteExpansionFails() throws Exception {
    tester.addFile("ts/BUILD",
        "test_suite(name = 'tests', tests = ['//nonexistent:my_test'])");
    tester.useLoadingOptions("--build_tests_only");
    TargetPatternPhaseValue loadingResult = tester.loadTestsKeepGoing("//ts:tests");
    assertThat(loadingResult.hasError()).isTrue();
    assertThat(loadingResult.hasPostExpansionError()).isFalse();
    tester.assertContainsError("no such package 'nonexistent'");
  }

  @Test
  public void testTestSuiteExpansionFailsForBuild() throws Exception {
    tester.addFile("ts/BUILD",
        "test_suite(name = 'tests', tests = [':nonexistent_test'])");
    TargetPatternPhaseValue loadingResult = tester.loadKeepGoing("//ts:tests");
    assertThat(loadingResult.hasError()).isFalse();
    assertThat(loadingResult.hasPostExpansionError()).isTrue();
    tester.assertContainsError(
        "expecting a test or a test_suite rule but '//ts:nonexistent_test' is not one");
  }

  @Test
  public void testTestSuiteExpansionFailsMissingTarget() throws Exception {
    tester.addFile("other/BUILD", "");
    tester.addFile("ts/BUILD",
        "test_suite(name = 'tests', tests = ['//other:no_such_test'])");
    TargetPatternPhaseValue result = tester.loadTestsKeepGoing("//ts:tests");
    assertThat(result.hasError()).isTrue();
    assertThat(result.hasPostExpansionError()).isTrue();
    tester.assertContainsError("no such target '//other:no_such_test'");
  }

  @Test
  public void testTestSuiteExpansionFailsMultipleSuites() throws Exception {
    tester.addFile("other/BUILD", "");
    tester.addFile("ts/BUILD",
        "test_suite(name = 'a', tests = ['//other:no_such_test'])",
        "test_suite(name = 'b', tests = [])");
    TargetPatternPhaseValue result = tester.loadTestsKeepGoing("//ts:all");
    assertThat(result.hasError()).isTrue();
    assertThat(result.hasPostExpansionError()).isTrue();
    tester.assertContainsError("no such target '//other:no_such_test'");
  }

  @Test
  public void testTestSuiteOverridesManualWithBuildTestsOnly() throws Exception {
    tester.addFile("foo/BUILD",
        "sh_test(name = 'foo', srcs = ['foo.sh'], tags = ['manual'])",
        "sh_test(name = 'bar', srcs = ['bar.sh'], tags = ['manual'])",
        "sh_test(name = 'baz', srcs = ['baz.sh'])",
        "test_suite(name = 'foo_suite', tests = [':foo', ':baz'])");
    tester.useLoadingOptions("--build_tests_only");
    TargetPatternPhaseValue result = assertNoErrors(tester.loadTests("//foo:all"));
    assertThat(result.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//foo:foo", "//foo:baz"));
    assertThat(result.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//foo:foo", "//foo:baz"));
    assertThat(tester.getFilteredTargets()).isEmpty();
    assertThat(tester.getTestFilteredTargets())
        .containsExactlyElementsIn(getLabels("//foo:foo_suite"));
  }

  /** Regression test for bug: "subtracting tests from test doesn't work" */
  @Test
  public void testFilterNegativeTestFromTestSuite() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD",
        "cc_test(name = 'my_test', srcs = ['test.cc'])",
        "cc_test(name = 'my_other_test', srcs = ['other_test.cc'])",
        "test_suite(name = 'tests', tests = [':my_test', ':my_other_test'])");
    TargetPatternPhaseValue result =
        assertNoErrors(tester.loadTests("//cc:tests", "-//cc:my_test"));
    assertThat(result.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//cc:my_other_test", "//cc:my_test"));
    assertThat(result.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//cc:my_other_test"));
  }

  /** Regression test for bug: "blaze doesn't seem to respect target subtractions" */
  @Test
  public void testNegativeTestSuiteExpanded() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD",
        "cc_test(name = 'my_test', srcs = ['test.cc'])",
        "cc_test(name = 'my_other_test', srcs = ['other_test.cc'])",
        "test_suite(name = 'tests', tests = [':my_test'])",
        "test_suite(name = 'all_tests', tests = ['my_other_test'])");
    TargetPatternPhaseValue result =
        assertNoErrors(tester.loadTests("//cc:all_tests", "-//cc:tests"));
    assertThat(result.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//cc:my_other_test"));
    assertThat(result.getTestsToRunLabels())
        .containsExactlyElementsIn(getLabels("//cc:my_other_test"));
  }

  @Test
  public void testComplexTestSuite() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD",
        "cc_test(name = 'test1', srcs = ['test.cc'])",
        "cc_test(name = 'test2', srcs = ['test.cc'])",
        "test_suite(name = 'empty', tags = ['impossible'], tests = [])",
        "test_suite(name = 'suite1', tests = ['empty', 'test1'])",
        "test_suite(name = 'suite2', tests = ['test2'])",
        "test_suite(name = 'all_tests', tests = ['suite1', 'suite2'])");
    TargetPatternPhaseValue result = assertNoErrors(tester.loadTests("//cc:all_tests"));
    assertThat(result.getTargetLabels())
        .containsExactlyElementsIn(getLabels("//cc:test1", "//cc:test2"));
  }

  /**
   * Regression test for bug: "blaze is lying to me about what tests exist (have been specified)"
   */
  @Test
  public void testTotalNegationEmitsWarning() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD",
        "cc_test(name = 'my_test', srcs = ['test.cc'])",
        "test_suite(name = 'tests', tests = [':my_test'])");
    TargetPatternPhaseValue result = tester.loadTests("//cc:tests", "-//cc:my_test");
    tester.assertContainsWarning("All specified test targets were excluded by filters");
    assertThat(result.getTestsToRunLabels()).containsExactlyElementsIn(getLabels());
  }

  @Test
  public void testRepeatedSameLoad() throws Exception {
    tester.addFile("base/BUILD",
        "filegroup(name = 'hello', srcs = ['foo.txt'])");
    TargetPatternPhaseValue firstResult = assertNoErrors(tester.load("//base:hello"));
    TargetPatternPhaseValue secondResult = assertNoErrors(tester.load("//base:hello"));
    assertThat(secondResult.getTargetLabels()).isEqualTo(firstResult.getTargetLabels());
    assertThat(secondResult.getTestsToRunLabels()).isEqualTo(firstResult.getTestsToRunLabels());
  }

  /**
   * Tests whether globs can update correctly when a new file is added.
   *
   * <p>The usage of {@link LoadingPhaseTester#sync()} triggers this via
   * {@link SkyframeExecutor#invalidateFilesUnderPathForTesting}.
   */
  @Test
  public void testGlobPicksUpNewFile() throws Exception {
    tester.addFile("foo/BUILD", "filegroup(name='x', srcs=glob(['*.y']))");
    tester.addFile("foo/a.y");
    Label label =
        Iterables.getOnlyElement(assertNoErrors(tester.load("//foo:x")).getTargetLabels());
    Target result = tester.getTarget(label.toString());
    assertThat(
        Iterables.transform(result.getAssociatedRule().getLabels(), Functions.toStringFunction()))
        .containsExactly("//foo:a.y");

    tester.addFile("foo/b.y");
    tester.sync();
    label = Iterables.getOnlyElement(assertNoErrors(tester.load("//foo:x")).getTargetLabels());
    result = tester.getTarget(label.toString());
    assertThat(
        Iterables.transform(result.getAssociatedRule().getLabels(), Functions.toStringFunction()))
        .containsExactly("//foo:a.y", "//foo:b.y");
  }

  /** Regression test: handle symlink cycles gracefully. */
  @Test
  public void testCycleReporting_SymlinkCycleDuringTargetParsing() throws Exception {
    tester.addFile("hello/BUILD", "cc_library(name = 'a', srcs = glob(['*.cc']))");
    Path buildFilePath = tester.getWorkspace().getRelative("hello/BUILD");
    Path dirPath = buildFilePath.getParentDirectory();
    Path fooFilePath = dirPath.getRelative("foo.cc");
    Path barFilePath = dirPath.getRelative("bar.cc");
    Path bazFilePath = dirPath.getRelative("baz.cc");
    fooFilePath.createSymbolicLink(barFilePath);
    barFilePath.createSymbolicLink(bazFilePath);
    bazFilePath.createSymbolicLink(fooFilePath);
    assertCircularSymlinksDuringTargetParsing("//hello:a");
  }

  @Test
  public void testRecursivePatternWithCircularSymlink() throws Exception {
    tester.getWorkspace().getChild("broken").createDirectory();

    // Create a circular symlink.
    tester.getWorkspace().getRelative(PathFragment.create("broken/BUILD"))
        .createSymbolicLink(PathFragment.create("BUILD"));

    assertCircularSymlinksDuringTargetParsing("//broken/...");
  }

  @Test
  public void testRecursivePatternWithTwoCircularSymlinks() throws Exception {
    tester.getWorkspace().getChild("broken").createDirectory();

    // Create a circular symlink.
    tester.getWorkspace().getRelative(PathFragment.create("broken/BUILD"))
        .createSymbolicLink(PathFragment.create("x"));
    tester.getWorkspace().getRelative(PathFragment.create("broken/x"))
        .createSymbolicLink(PathFragment.create("BUILD"));

    assertCircularSymlinksDuringTargetParsing("//broken/...");
  }

  @Test
  public void testSuiteInSuite() throws Exception {
    tester.addFile("suite/BUILD",
        "test_suite(name = 'a', tests = [':b'])",
        "test_suite(name = 'b', tests = [':c'])",
        "sh_test(name = 'c', srcs = ['test.cc'])");
    TargetPatternPhaseValue result = assertNoErrors(tester.load("//suite:a"));
    assertThat(result.getTargetLabels()).containsExactlyElementsIn(getLabels("//suite:c"));
  }

  @Test
  public void testTopLevelTargetErrorsPrintedExactlyOnce_NoKeepGoing() throws Exception {
    tester.addFile("bad/BUILD",
        "sh_binary(name = 'bad', srcs = ['bad.sh'])",
        "undefined_symbol");
    try {
      tester.load("//bad");
      fail();
    } catch (TargetParsingException expected) {
    }
    tester.assertContainsEventWithFrequency("name 'undefined_symbol' is not defined", 1);
    PatternExpandingError err = tester.findPostOnce(PatternExpandingError.class);
    assertThat(err.getPattern()).containsExactly("//bad");
  }

  @Test
  public void testTopLevelTargetErrorsPrintedExactlyOnce_KeepGoing() throws Exception {
    tester.addFile("bad/BUILD",
        "sh_binary(name = 'bad', srcs = ['bad.sh'])",
        "undefined_symbol");
    TargetPatternPhaseValue result = tester.loadKeepGoing("//bad");
    assertThat(result.hasError()).isTrue();
    tester.assertContainsEventWithFrequency("name 'undefined_symbol' is not defined", 1);
  }

  @Test
  public void testCompileOneDependency() throws Exception {
    tester.addFile("base/BUILD",
        "cc_library(name = 'hello', srcs = ['hello.cc'])");
    tester.useLoadingOptions("--compile_one_dependency");
    TargetPatternPhaseValue result = assertNoErrors(tester.load("base/hello.cc"));
    assertThat(result.getTargetLabels()).containsExactlyElementsIn(getLabels("//base:hello"));
  }

  @Test
  public void testCompileOneDependencyNonExistentSource() throws Exception {
    tester.addFile("base/BUILD",
        "cc_library(name = 'hello', srcs = ['hello.cc', '//bad:bad.cc'])");
    tester.useLoadingOptions("--compile_one_dependency");
    try {
      TargetPatternPhaseValue loadingResult = tester.load("base/hello.cc");
      assertThat(loadingResult.hasPostExpansionError()).isFalse();
    } catch (LoadingFailedException expected) {
      tester.assertContainsError("no such package 'bad'");
    }
  }

  @Test
  public void testCompileOneDependencyNonExistentSourceKeepGoing() throws Exception {
    tester.addFile("base/BUILD",
        "cc_library(name = 'hello', srcs = ['hello.cc', '//bad:bad.cc'])");
    tester.useLoadingOptions("--compile_one_dependency");
    TargetPatternPhaseValue loadingResult = tester.loadKeepGoing("base/hello.cc");
    assertThat(loadingResult.hasPostExpansionError()).isFalse();
  }

  @Test
  public void testCompileOneDependencyReferencesFile() throws Exception {
    tester.addFile("base/BUILD",
        "cc_library(name = 'hello', srcs = ['hello.cc', '//bad:bad.cc'])");
    tester.useLoadingOptions("--compile_one_dependency");
    try {
      tester.load("//base:hello");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e).hasMessageThat()
          .contains("--compile_one_dependency target '//base:hello' must be a file");
    }
  }

  @Test
  public void testParsingFailureReported() throws Exception {
    TargetPatternPhaseValue loadingResult = tester.loadKeepGoing("//does_not_exist");
    assertThat(loadingResult.hasError()).isTrue();
    ParsingFailedEvent event = tester.findPostOnce(ParsingFailedEvent.class);
    assertThat(event.getPattern()).isEqualTo("//does_not_exist");
    assertThat(event.getMessage()).contains("BUILD file not found on package path");
  }

  @Test
  public void testCyclesKeepGoing() throws Exception {
    tester.addFile("test/BUILD", "load(':cycle1.bzl', 'make_cycle')");
    tester.addFile("test/cycle1.bzl", "load(':cycle2.bzl', 'make_cycle')");
    tester.addFile("test/cycle2.bzl", "load(':cycle1.bzl', 'make_cycle')");
    // The skyframe target pattern evaluator isn't able to provide partial results in the presence
    // of cycles, so it simply raises an exception rather than returning an empty result.
    try {
      tester.load("//test:cycle1");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e).hasMessageThat().contains("cycles detected");
    }
    tester.assertContainsEventWithFrequency("cycle detected in extension", 1);
    PatternExpandingError err = tester.findPostOnce(PatternExpandingError.class);
    assertThat(err.getPattern()).containsExactly("//test:cycle1");
  }

  @Test
  public void testCyclesNoKeepGoing() throws Exception {
    tester.addFile("test/BUILD", "load(':cycle1.bzl', 'make_cycle')");
    tester.addFile("test/cycle1.bzl", "load(':cycle2.bzl', 'make_cycle')");
    tester.addFile("test/cycle2.bzl", "load(':cycle1.bzl', 'make_cycle')");
    try {
      tester.load("//test:cycle1");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e).hasMessageThat().contains("cycles detected");
    }
    tester.assertContainsEventWithFrequency("cycle detected in extension", 1);
    PatternExpandingError err = tester.findPostOnce(PatternExpandingError.class);
    assertThat(err.getPattern()).containsExactly("//test:cycle1");
  }

  private static class LoadingPhaseTester {
    private final ManualClock clock = new ManualClock();
    private final Path workspace;

    private final AnalysisMock analysisMock;
    private final SkyframeExecutor skyframeExecutor;

    private final List<Path> changes = new ArrayList<>();
    private final BlazeDirectories directories;
    private final ActionKeyContext actionKeyContext = new ActionKeyContext();

    private LoadingOptions options;
    private final StoredEventHandler storedErrors;

    private TargetParsingCompleteEvent targetParsingCompleteEvent;
    private LoadingPhaseCompleteEvent loadingPhaseCompleteEvent;

    private MockToolsConfig mockToolsConfig;

    public LoadingPhaseTester() throws IOException {
      FileSystem fs = new InMemoryFileSystem(clock);
      this.workspace = fs.getPath("/workspace");
      workspace.createDirectory();
      mockToolsConfig = new MockToolsConfig(workspace);
      analysisMock = AnalysisMock.get();
      analysisMock.setupMockClient(mockToolsConfig);
      directories =
          new BlazeDirectories(
              new ServerDirectories(
                  fs.getPath("/install"), fs.getPath("/output"), fs.getPath("/userRoot")),
              workspace,
              /* defaultSystemJavabase= */ null,
              analysisMock.getProductName());
      FileSystemUtils.deleteTree(workspace.getRelative("base"));

      ConfiguredRuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
      PackageFactory pkgFactory =
          analysisMock.getPackageFactoryBuilderForTesting(directories).build(ruleClassProvider);
      PackageCacheOptions options = Options.getDefaults(PackageCacheOptions.class);
      storedErrors = new StoredEventHandler();
      BuildOptions defaultBuildOptions;
      try {
        defaultBuildOptions = BuildOptions.of(ImmutableList.of());
      } catch (OptionsParsingException e) {
        throw new RuntimeException(e);
      }
      skyframeExecutor =
          SequencedSkyframeExecutor.create(
              pkgFactory,
              fs,
              directories,
              actionKeyContext,
              null, /* workspaceStatusActionFactory -- not used */
              ruleClassProvider.getBuildInfoFactories(),
              ImmutableList.<DiffAwareness.Factory>of(),
              analysisMock.getSkyFunctions(directories),
              ImmutableList.<SkyValueDirtinessChecker>of(),
              BazelSkyframeExecutorConstants.HARDCODED_BLACKLISTED_PACKAGE_PREFIXES,
              BazelSkyframeExecutorConstants.ADDITIONAL_BLACKLISTED_PACKAGE_PREFIXES_FILE,
              BazelSkyframeExecutorConstants.CROSS_REPOSITORY_LABEL_VIOLATION_STRATEGY,
              BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
              BazelSkyframeExecutorConstants.ACTION_ON_IO_EXCEPTION_READING_BUILD_FILE,
              defaultBuildOptions);
      TestConstants.processSkyframeExecutorForTesting(skyframeExecutor);
      PathPackageLocator pkgLocator =
          PathPackageLocator.create(
              null,
              options.packagePath,
              storedErrors,
              workspace,
              workspace,
              BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
      PackageCacheOptions packageCacheOptions = Options.getDefaults(PackageCacheOptions.class);
      packageCacheOptions.defaultVisibility = ConstantRuleVisibility.PRIVATE;
      packageCacheOptions.showLoadingProgress = true;
      packageCacheOptions.globbingThreads = 7;
      skyframeExecutor.preparePackageLoading(
          pkgLocator,
          packageCacheOptions,
          Options.getDefaults(SkylarkSemanticsOptions.class),
          analysisMock.getDefaultsPackageContent(),
          UUID.randomUUID(),
          ImmutableMap.<String, String>of(),
          new TimestampGranularityMonitor(clock));
      skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
      this.options = Options.getDefaults(LoadingOptions.class);
    }

    public void useLoadingOptions(String... options) throws OptionsParsingException {
      OptionsParser parser = OptionsParser.newOptionsParser(LoadingOptions.class);
      parser.parse(ImmutableList.copyOf(options));
      this.options = parser.getOptions(LoadingOptions.class);
    }

    public TargetPatternPhaseValue load(String... patterns) throws Exception {
      return loadWithFlags(/*keepGoing=*/false, /*determineTests=*/false, patterns);
    }

    public TargetPatternPhaseValue loadKeepGoing(String... patterns) throws Exception {
      return loadWithFlags(/*keepGoing=*/true, /*determineTests=*/false, patterns);
    }

    public TargetPatternPhaseValue loadTests(String... patterns) throws Exception {
      return loadWithFlags(/*keepGoing=*/false, /*determineTests=*/true, patterns);
    }

    public TargetPatternPhaseValue loadTestsKeepGoing(String... patterns) throws Exception {
      return loadWithFlags(/*keepGoing=*/true, /*determineTests=*/true, patterns);
    }

    public TargetPatternPhaseValue loadWithFlags(
        boolean keepGoing, boolean determineTests, String... patterns) throws Exception {
      sync();
      storedErrors.clear();
      TargetPatternPhaseValue result =
          skyframeExecutor.loadTargetPatterns(
              storedErrors,
              ImmutableList.copyOf(patterns),
              PathFragment.EMPTY_FRAGMENT,
              options,
              // We load very few packages, and everything is in memory; two should be plenty.
              /* threadCount= */ 2,
              keepGoing,
              determineTests);
      this.targetParsingCompleteEvent = findPost(TargetParsingCompleteEvent.class);
      this.loadingPhaseCompleteEvent = findPost(LoadingPhaseCompleteEvent.class);
      if (!keepGoing) {
        assertThat(storedErrors.hasErrors()).isFalse();
      }
      return result;
    }

    public Path getWorkspace() {
      return workspace;
    }

    public void addFile(String fileName, String... content) throws IOException {
      Path buildFile = workspace.getRelative(fileName);
      Preconditions.checkState(!buildFile.exists());
      Path currentPath = buildFile;

      // Add the new file and all the directories that will be created by
      // createDirectoryAndParents()
      while (!currentPath.exists()) {
        changes.add(currentPath);
        currentPath = currentPath.getParentDirectory();
      }

      buildFile.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContentAsLatin1(buildFile, Joiner.on('\n').join(content));
    }

    private void sync() throws InterruptedException {
      String pkgContents = analysisMock.getDefaultsPackageContent();
      skyframeExecutor.setupDefaultPackage(pkgContents);
      clock.advanceMillis(1);
      ModifiedFileSet.Builder builder = ModifiedFileSet.builder();
      for (Path path : changes) {
        if (!path.startsWith(workspace)) {
          continue;
        }

        PathFragment workspacePath = path.relativeTo(workspace);
        builder.modify(workspacePath);
      }
      ModifiedFileSet modified = builder.build();
      skyframeExecutor.invalidateFilesUnderPathForTesting(
          storedErrors, modified, Root.fromPath(workspace));

      changes.clear();
    }

    public Target getTarget(String targetName) throws Exception {
      StoredEventHandler eventHandler = new StoredEventHandler();
      Target target = getPkgManager().getTarget(
          eventHandler, Label.parseAbsoluteUnchecked(targetName));
      assertThat(eventHandler.hasErrors()).isFalse();
      return target;
    }

    private PackageManager getPkgManager() {
      return skyframeExecutor.getPackageManager();
    }

    public ImmutableSet<Label> getFilteredTargets() {
      return ImmutableSet.copyOf(targetParsingCompleteEvent.getFilteredLabels());
    }

    public ImmutableSet<Label> getTestFilteredTargets() {
      return ImmutableSet.copyOf(targetParsingCompleteEvent.getTestFilteredLabels());
    }

    public ImmutableSet<Label> getOriginalTargets() {
      return ImmutableSet.copyOf(targetParsingCompleteEvent.getLabels());
    }

    public ImmutableSet<Label> getTestSuiteTargets() {
      return loadingPhaseCompleteEvent.getFilteredLabels();
    }

    private Iterable<Event> filteredEvents() {
      return Iterables.filter(storedErrors.getEvents(), new Predicate<Event>() {
        @Override
        public boolean apply(Event event) {
          return event.getKind() != EventKind.PROGRESS;
        }
      });
    }

    public void assertNoEvents() {
      MoreAsserts.assertNoEvents(filteredEvents());
    }

    public Event assertContainsWarning(String expectedMessage) {
      return MoreAsserts.assertContainsEvent(filteredEvents(), expectedMessage, EventKind.WARNING);
    }

    public Event assertContainsError(String expectedMessage) {
      return MoreAsserts.assertContainsEvent(filteredEvents(), expectedMessage, EventKind.ERRORS);
    }

    public void assertContainsEventWithFrequency(String expectedMessage, int expectedFrequency) {
      MoreAsserts.assertContainsEventWithFrequency(
          filteredEvents(), expectedMessage, expectedFrequency);
    }

    public <T extends Postable> T findPost(Class<T> clazz) {
      return Iterators.getNext(
          storedErrors.getPosts().stream().filter(clazz::isInstance).map(clazz::cast).iterator(),
          null);
    }

    public <T extends Postable> T findPostOnce(Class<T> clazz) {
      return storedErrors
          .getPosts()
          .stream()
          .filter(clazz::isInstance)
          .map(clazz::cast)
          .collect(MoreCollectors.onlyElement());
    }
  }
}
