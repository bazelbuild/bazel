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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Tests for {@link LoadingPhaseRunner}. */
@RunWith(JUnit4.class)
public class LoadingPhaseRunnerTest {

  private static final ImmutableList<Logger> loggers = ImmutableList.of(
      Logger.getLogger(LoadingPhaseRunner.class.getName()),
      Logger.getLogger(BuildView.class.getName()));
  static {
    for (Logger logger : loggers) {
      logger.setLevel(Level.OFF);
    }
  }

  private LoadingPhaseTester tester;

  @Before
  public final void createLoadingPhaseTester() throws Exception  {
    tester = new LoadingPhaseTester(!runsLoadingPhase());
  }

  protected List<Target> getTargets(String... targetNames) throws Exception {
    return tester.getTargets(targetNames);
  }

  protected boolean runsLoadingPhase() {
    return true;
  }

  @Test
  public void testSmoke() throws Exception {
    tester.addFile("base/BUILD",
        "filegroup(name = 'hello', srcs = ['foo.txt'])");
    LoadingResult loadingResult = assertNoErrors(tester.load("//base:hello"));
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets("//base:hello"));
    assertNull(loadingResult.getTestsToRun());
    // TODO(ulfjack): We don't collect package roots if we don't run the loading phase.
    if (runsLoadingPhase()) {
      assertEquals(
          ImmutableMap.of(PackageIdentifier.createInDefaultRepo("base"), tester.getWorkspace()),
          loadingResult.getPackageRoots());
    }
  }

  @Test
  public void testSmokeWithCallback() throws Exception {
    tester.addFile("base/BUILD",
        "filegroup(name = 'hello', srcs = ['foo.txt'])");
    final List<Target> targetsNotified = new ArrayList<>();
    tester.setCallback(new LoadingCallback() {
      @Override
      public void notifyTargets(Collection<Target> targets) throws LoadingFailedException {
        targetsNotified.addAll(targets);
      }

      @Override
      public void notifyVisitedPackages(Set<PackageIdentifier> visitedPackages) {
      }
    });
    assertNoErrors(tester.load("//base:hello"));
    assertThat(targetsNotified).containsExactlyElementsIn(getTargets("//base:hello"));
  }

  @Test
  public void testNonExistentPackage() throws Exception {
    LoadingResult loadingResult = tester.loadKeepGoing("//base:missing");
    assertThat(loadingResult.hasTargetPatternError()).isTrue();
    assertThat(loadingResult.hasLoadingError()).isFalse();
    assertThat(loadingResult.getTargets()).isEmpty();
    assertThat(loadingResult.getTestsToRun()).isNull();
    assertThat(loadingResult.getPackageRoots()).isEmpty();
    tester.assertContainsError("Skipping '//base:missing': no such package 'base'");
    tester.assertContainsWarning("Target pattern parsing failed.");
  }

  @Test
  public void testNonExistentTarget() throws Exception {
    tester.addFile("base/BUILD");
    LoadingResult loadingResult = tester.loadKeepGoing("//base:missing");
    assertThat(loadingResult.hasTargetPatternError()).isTrue();
    assertThat(loadingResult.hasLoadingError()).isFalse();
    assertThat(loadingResult.getTargets()).isEmpty();
    assertThat(loadingResult.getTestsToRun()).isNull();
    assertThat(loadingResult.getPackageRoots()).isEmpty();
    tester.assertContainsError("Skipping '//base:missing': no such target '//base:missing'");
    tester.assertContainsWarning("Target pattern parsing failed.");
  }

  @Test
  public void testBadTargetPatternWithTest() throws Exception {
    tester.addFile("base/BUILD");
    LoadingResult loadingResult = tester.loadTestsKeepGoing("//base:missing");
    assertTrue(loadingResult.hasTargetPatternError());
    assertFalse(loadingResult.hasLoadingError());
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(ImmutableList.<Target>of());
    assertThat(loadingResult.getTestsToRun()).containsExactlyElementsIn(ImmutableList.<Target>of());
    tester.assertContainsError("Skipping '//base:missing': no such target '//base:missing'");
    tester.assertContainsWarning("Target pattern parsing failed.");
  }

  @Test
  public void testBadTransitiveClosure() throws Exception {
    if (!runsLoadingPhase()) {
      // TODO(ulfjack): Requires loading phase.
      return;
    }
    tester.addFile("base/BUILD",
        "filegroup(name = 'hello', srcs = ['//nonexistent:missing'])");
    LoadingResult loadingResult = tester.loadKeepGoing("//base:hello");
    assertFalse(loadingResult.hasTargetPatternError());
    assertTrue(loadingResult.hasLoadingError());
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(ImmutableList.<Target>of());
    assertNull(loadingResult.getTestsToRun());
    assertTrue(loadingResult.getPackageRoots().size() <= 1);
    tester.assertContainsError("no such package 'nonexistent': BUILD file not found");
  }

  @Test
  public void testManualTarget() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD", "cc_library(name = 'my_lib', srcs = ['lib.cc'], tags = ['manual'])");
    LoadingResult loadingResult = assertNoErrors(tester.load("//cc:all"));
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets());

    // Explicitly specified on the command line.
    loadingResult = assertNoErrors(tester.load("//cc:my_lib"));
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets("//cc:my_lib"));
  }

  @Test
  public void testConfigSettingTarget() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("config/BUILD",
        "cc_library(name = 'somelib', srcs = [ 'somelib.cc' ], hdrs = [ 'somelib.h' ])",
        "config_setting(name = 'configa', values = { 'define': 'foo=a' })",
        "config_setting(name = 'configb', values = { 'define': 'foo=b' })");
    LoadingResult loadingResult = assertNoErrors(tester.load("//config:all"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//config:somelib"));

    // Explicitly specified on the command line.
    loadingResult = assertNoErrors(tester.load("//config:configa"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//config:configa"));
  }

  @Test
  public void testNegativeTestDoesNotShowUpAtAll() throws Exception {
    tester.addFile("my_test/BUILD",
        "sh_test(name = 'my_test', srcs = ['test.cc'])");
    assertNoErrors(tester.loadTests("-//my_test"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getTargets());
  }

  @Test
  public void testNegativeTargetDoesNotShowUpAtAll() throws Exception {
    tester.addFile("my_library/BUILD",
        "cc_library(name = 'my_library', srcs = ['test.cc'])");
    assertNoErrors(tester.loadTests("-//my_library"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getTargets());
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
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//tests:t1", "//tests:t2"));
    assertThat(loadingResult.getTestsToRun())
        .containsExactlyElementsIn(getTargets("//tests:t1", "//tests:t2"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getTargets());
  }

  @Test
  public void testTestFilteringBuildTestsOnly() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--build_tests_only");
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//tests:t1", "//tests:t2"));
    assertThat(loadingResult.getTestsToRun())
        .containsExactlyElementsIn(getTargets("//tests:t1", "//tests:t2"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getTargets());
  }

  @Test
  public void testTestFilteringSize() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--test_size_filters=small");
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//tests:t1", "//tests:t2"));
    assertThat(loadingResult.getTestsToRun()).containsExactlyElementsIn(getTargets("//tests:t1"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getTargets());
  }

  @Test
  public void testTestFilteringSizeAndBuildTestsOnly() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--test_size_filters=small", "--build_tests_only");
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//tests:all"));
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets("//tests:t1"));
    assertThat(loadingResult.getTestsToRun()).containsExactlyElementsIn(getTargets("//tests:t1"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getTargets("//tests:t2"));
  }

  @Test
  public void testTestFilteringLocalAndBuildTestsOnly() throws Exception {
    writeBuildFilesForTestFiltering();
    tester.useLoadingOptions("--test_tag_filters=local", "--build_tests_only");
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//tests:all", "//tests:t3"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//tests:t1", "//tests:t3"));
    assertThat(loadingResult.getTestsToRun())
        .containsExactlyElementsIn(getTargets("//tests:t1", "//tests:t3"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets()).containsExactlyElementsIn(getTargets("//tests:t2"));
  }

  @Test
  public void testTestSuiteExpansion() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD",
        "cc_test(name = 'my_test', srcs = ['test.cc'])",
        "test_suite(name = 'tests', tests = [':my_test'])");
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//cc:tests"));
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets("//cc:my_test"));
    assertThat(loadingResult.getTestsToRun()).containsExactlyElementsIn(getTargets("//cc:my_test"));
    if (runsLoadingPhase()) {
      assertThat(loadingResult.getPackageRoots().entrySet())
          .contains(entryFor(PackageIdentifier.createInDefaultRepo("cc"), tester.getWorkspace()));
    }
    assertThat(tester.getOriginalTargets())
        .containsExactlyElementsIn(getTargets("//cc:tests", "//cc:my_test"));
    assertThat(tester.getTestSuiteTargets())
        .containsExactlyElementsIn(getTargets("//cc:tests"));
  }

  @Test
  public void testTestSuiteExpansionFails() throws Exception {
    tester.addFile("ts/BUILD",
        "test_suite(name = 'tests', tests = ['//nonexistent:my_test'])");
    tester.useLoadingOptions("--build_tests_only");
    LoadingResult loadingResult = tester.loadTestsKeepGoing("//ts:tests");
    assertTrue(loadingResult.hasTargetPatternError());
    assertFalse(loadingResult.hasLoadingError());
    tester.assertContainsError("no such package 'nonexistent'");
  }

  @Test
  public void testTestSuiteExpansionFailsForBuild() throws Exception {
    tester.addFile("ts/BUILD",
        "test_suite(name = 'tests', tests = [':nonexistent_test'])");
    LoadingResult loadingResult = tester.loadKeepGoing("//ts:tests");
    assertFalse(loadingResult.hasTargetPatternError());
    assertTrue(loadingResult.hasLoadingError());
    tester.assertContainsError(
        "expecting a test or a test_suite rule but '//ts:nonexistent_test' is not one");
  }

  @Test
  public void testTestSuiteExpansionFailsMissingTarget() throws Exception {
    tester.addFile("other/BUILD", "");
    tester.addFile("ts/BUILD",
        "test_suite(name = 'tests', tests = ['//other:no_such_test'])");
    LoadingResult loadingResult = tester.loadTestsKeepGoing("//ts:tests");
    assertTrue(loadingResult.hasTargetPatternError());
    assertTrue(loadingResult.hasLoadingError());
    tester.assertContainsError("no such target '//other:no_such_test'");
  }

  @Test
  public void testTestSuiteExpansionFailsMultipleSuites() throws Exception {
    tester.addFile("other/BUILD", "");
    tester.addFile("ts/BUILD",
        "test_suite(name = 'a', tests = ['//other:no_such_test'])",
        "test_suite(name = 'b', tests = [])");
    LoadingResult loadingResult = tester.loadTestsKeepGoing("//ts:all");
    assertTrue(loadingResult.hasTargetPatternError());
    assertTrue(loadingResult.hasLoadingError());
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
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//foo:all"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//foo:foo", "//foo:baz"));
    assertThat(loadingResult.getTestsToRun())
        .containsExactlyElementsIn(getTargets("//foo:foo", "//foo:baz"));
    assertThat(tester.getFilteredTargets()).containsExactlyElementsIn(getTargets());
    assertThat(tester.getTestFilteredTargets())
        .containsExactlyElementsIn(getTargets("//foo:foo_suite"));
  }

  /** Regression test for bug: "subtracting tests from test doesn't work" */
  @Test
  public void testFilterNegativeTestFromTestSuite() throws Exception {
    AnalysisMock.get().ccSupport().setup(tester.mockToolsConfig);
    tester.addFile("cc/BUILD",
        "cc_test(name = 'my_test', srcs = ['test.cc'])",
        "cc_test(name = 'my_other_test', srcs = ['other_test.cc'])",
        "test_suite(name = 'tests', tests = [':my_test', ':my_other_test'])");
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//cc:tests", "-//cc:my_test"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//cc:my_other_test", "//cc:my_test"));
    assertThat(loadingResult.getTestsToRun())
        .containsExactlyElementsIn(getTargets("//cc:my_other_test"));
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
    LoadingResult loadingResult = assertNoErrors(tester.loadTests("//cc:all_tests", "-//cc:tests"));
    assertThat(loadingResult.getTargets())
        .containsExactlyElementsIn(getTargets("//cc:my_other_test"));
    assertThat(loadingResult.getTestsToRun())
        .containsExactlyElementsIn(getTargets("//cc:my_other_test"));
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
    LoadingResult loadingResult = tester.loadTests("//cc:tests", "-//cc:my_test");
    tester.assertContainsWarning("All specified test targets were excluded by filters");
    if (runsLoadingPhase()) {
      assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets("//cc:my_test"));
    }
    assertThat(loadingResult.getTestsToRun()).containsExactlyElementsIn(getTargets());
  }

  @Test
  public void testRepeatedSameLoad() throws Exception {
    tester.addFile("base/BUILD",
        "filegroup(name = 'hello', srcs = ['foo.txt'])");
    LoadingResult firstResult = assertNoErrors(tester.load("//base:hello"));
    LoadingResult secondResult = assertNoErrors(tester.load("//base:hello"));
    assertEquals(firstResult.getTargets(), secondResult.getTargets());
    assertEquals(firstResult.getTestsToRun(), secondResult.getTestsToRun());
    assertEquals(firstResult.getPackageRoots(), secondResult.getPackageRoots());
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
    Target result = Iterables.getOnlyElement(assertNoErrors(tester.load("//foo:x")).getTargets());
    assertThat(
        Iterables.transform(result.getAssociatedRule().getLabels(), Functions.toStringFunction()))
        .containsExactly("//foo:a.y");

    tester.addFile("foo/b.y");
    tester.sync();
    result = Iterables.getOnlyElement(assertNoErrors(tester.load("//foo:x")).getTargets());
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
    tester.getWorkspace().getRelative(new PathFragment("broken/BUILD"))
        .createSymbolicLink(new PathFragment("BUILD"));

    assertCircularSymlinksDuringTargetParsing("//broken/...");
  }

  @Test
  public void testRecursivePatternWithTwoCircularSymlinks() throws Exception {
    tester.getWorkspace().getChild("broken").createDirectory();

    // Create a circular symlink.
    tester.getWorkspace().getRelative(new PathFragment("broken/BUILD"))
        .createSymbolicLink(new PathFragment("x"));
    tester.getWorkspace().getRelative(new PathFragment("broken/x"))
        .createSymbolicLink(new PathFragment("BUILD"));

    assertCircularSymlinksDuringTargetParsing("//broken/...");
  }

  @Test
  public void testSuiteInSuite() throws Exception {
    tester.addFile("suite/BUILD",
        "test_suite(name = 'a', tests = [':b'])",
        "test_suite(name = 'b', tests = [':c'])",
        "sh_test(name = 'c', srcs = ['test.cc'])");
    LoadingResult loadingResult = assertNoErrors(tester.load("//suite:a"));
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets("//suite:c"));
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
  }

  @Test
  public void testTopLevelTargetErrorsPrintedExactlyOnce_KeepGoing() throws Exception {
    tester.addFile("bad/BUILD",
        "sh_binary(name = 'bad', srcs = ['bad.sh'])",
        "undefined_symbol");
    LoadingResult loadingResult = tester.loadKeepGoing("//bad");
    if (runsLoadingPhase()) {
      // The legacy loading phase runner reports a loading error, but no target pattern error in
      // keep_going mode, even though it's clearly an error in the referenced target itself, rather
      // than in its transitive closure. This happens because the target pattern eval swallows such
      // errors in keep_going mode. We could fix that, but it's a fairly invasive change, and we're
      // planning to migrate to the Skyframe-based implementation anyway.
      assertThat(loadingResult.hasTargetPatternError()).isFalse();
      assertThat(loadingResult.hasLoadingError()).isTrue();
    } else {
      assertThat(loadingResult.hasTargetPatternError()).isTrue();
    }
    tester.assertContainsEventWithFrequency("name 'undefined_symbol' is not defined", 1);
  }

  @Test
  public void testCompileOneDependency() throws Exception {
    tester.addFile("base/BUILD",
        "cc_library(name = 'hello', srcs = ['hello.cc'])");
    tester.useLoadingOptions("--compile_one_dependency");
    LoadingResult loadingResult = assertNoErrors(tester.load("base/hello.cc"));
    assertThat(loadingResult.getTargets()).containsExactlyElementsIn(getTargets("//base:hello"));
  }

  @Test
  public void testCompileOneDependencyNonExistentSource() throws Exception {
    tester.addFile("base/BUILD",
        "cc_library(name = 'hello', srcs = ['hello.cc', '//bad:bad.cc'])");
    tester.useLoadingOptions("--compile_one_dependency");
    try {
      tester.load("base/hello.cc");
      fail();
    } catch (TargetParsingException expected) {
      tester.assertContainsError("no such package 'bad'");
    }
  }

  @Test
  public void testCompileOneDependencyNonExistentSourceKeepGoing() throws Exception {
    tester.addFile("base/BUILD",
        "cc_library(name = 'hello', srcs = ['hello.cc', '//bad:bad.cc'])");
    tester.useLoadingOptions("--compile_one_dependency");
    if (runsLoadingPhase()) {
      // The LegacyLoadingPhaseRunner throws an exception if it can't load any of the sources in the
      // same rule as the source we're looking for even with --keep_going.
      // In general, we probably want --compile_one_dependency to be compatible with --keep_going
      // for consistency, but it's unclear if this is actually a problem for anyone. The most common
      // use case for compile_one_dependency is to iterate quickly on a single file, without
      // --keep_going.
      try {
        tester.load("base/hello.cc");
        fail();
      } catch (TargetParsingException expected) {
        tester.assertContainsError("no such package 'bad'");
      }
    } else {
      LoadingResult loadingResult = tester.loadKeepGoing("base/hello.cc");
      assertThat(loadingResult.hasTargetPatternError()).isTrue();
    }
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

  private LoadingResult assertNoErrors(LoadingResult loadingResult) {
    assertFalse(loadingResult.hasTargetPatternError());
    assertFalse(loadingResult.hasLoadingError());
    tester.assertNoEvents();
    return loadingResult;
  }

  private <K, V> Map.Entry<K, V> entryFor(K key, V value) {
    return new AbstractMap.SimpleImmutableEntry<>(key, value);
  }

  private static class LoadingPhaseTester {
    private final ManualClock clock = new ManualClock();
    private final Path workspace;

    private final SkyframeExecutor skyframeExecutor;

    private final List<Path> changes = new ArrayList<>();
    private final LoadingPhaseRunner loadingPhaseRunner;

    private LoadingOptions options;
    private final StoredEventHandler storedErrors;
    private LoadingCallback loadingCallback;

    private TargetParsingCompleteEvent targetParsingCompleteEvent;
    private LoadingPhaseCompleteEvent loadingPhaseCompleteEvent;

    private MockToolsConfig mockToolsConfig;

    public LoadingPhaseTester(boolean useNewImpl) throws IOException {
      FileSystem fs = new InMemoryFileSystem(clock);
      this.workspace = fs.getPath("/workspace");
      workspace.createDirectory();
      mockToolsConfig = new MockToolsConfig(workspace);
      AnalysisMock.get().setupMockClient(mockToolsConfig);
      FileSystemUtils.deleteTree(workspace.getRelative("base"));

      PackageFactory pkgFactory = new PackageFactory(TestRuleClassProvider.getRuleClassProvider());
      PackageCacheOptions options = Options.getDefaults(PackageCacheOptions.class);
      storedErrors = new StoredEventHandler();
      BlazeDirectories directories =
          new BlazeDirectories(fs.getPath("/install"), fs.getPath("/output"), workspace);
      skyframeExecutor = SequencedSkyframeExecutor.create(pkgFactory,
          new TimestampGranularityMonitor(clock),
          directories,
          null,  /* binTools -- not used */
          null,  /* workspaceStatusActionFactory -- not used */
          TestRuleClassProvider.getRuleClassProvider().getBuildInfoFactories(),
          ImmutableList.<DiffAwareness.Factory>of(),
          Predicates.<PathFragment>alwaysFalse(),
          Preprocessor.Factory.Supplier.NullSupplier.INSTANCE,
          AnalysisMock.get().getSkyFunctions(directories),
          ImmutableList.<PrecomputedValue.Injected>of(),
          ImmutableList.<SkyValueDirtinessChecker>of());
      PathPackageLocator pkgLocator = PathPackageLocator.create(
          null, options.packagePath, storedErrors, workspace, workspace);
      skyframeExecutor.preparePackageLoading(pkgLocator,
          ConstantRuleVisibility.PRIVATE, true,
          7, TestRuleClassProvider.getRuleClassProvider().getDefaultsPackageContent(),
          UUID.randomUUID());
      loadingPhaseRunner = skyframeExecutor.getLoadingPhaseRunner(
          pkgFactory.getRuleClassNames(), useNewImpl);
      this.options = Options.getDefaults(LoadingOptions.class);
    }

    public void setCallback(LoadingCallback loadingCallback) {
      this.loadingCallback = loadingCallback;
    }

    public void useLoadingOptions(String... options) throws OptionsParsingException {
      OptionsParser parser = OptionsParser.newOptionsParser(LoadingOptions.class);
      parser.parse(ImmutableList.copyOf(options));
      this.options = parser.getOptions(LoadingOptions.class);
    }

    public LoadingResult load(String... patterns) throws Exception {
      return load(/*keepGoing=*/false, /*determineTests=*/false, patterns);
    }

    public LoadingResult loadKeepGoing(String... patterns) throws Exception {
      return load(/*keepGoing=*/true, /*determineTests=*/false, patterns);
    }

    public LoadingResult loadTests(String... patterns) throws Exception {
      return load(/*keepGoing=*/false, /*determineTests=*/true, patterns);
    }

    public LoadingResult loadTestsKeepGoing(String... patterns) throws Exception {
      return load(/*keepGoing=*/true, /*determineTests=*/true, patterns);
    }

    public LoadingResult load(boolean keepGoing, boolean determineTests, String... patterns)
        throws Exception {
      sync();
      storedErrors.clear();
      LoadingResult result;
      try {
        EventBus eventBus = new EventBus();
        FilteredTargetListener listener = new FilteredTargetListener();
        eventBus.register(listener);
        result = loadingPhaseRunner.execute(storedErrors, eventBus,
            ImmutableList.copyOf(patterns), options, ImmutableListMultimap.<String, Label>of(),
            keepGoing, /*enableLoading=*/true, determineTests, loadingCallback);
        this.targetParsingCompleteEvent = listener.targetParsingCompleteEvent;
        this.loadingPhaseCompleteEvent = listener.loadingPhaseCompleteEvent;
      } catch (LoadingFailedException e) {
        System.err.println(storedErrors.getEvents());
        throw e;
      }
      if (!keepGoing) {
        assertFalse(storedErrors.hasErrors());
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

      FileSystemUtils.createDirectoryAndParents(buildFile.getParentDirectory());
      FileSystemUtils.writeContentAsLatin1(buildFile, Joiner.on('\n').join(content));
    }

    private void sync() throws InterruptedException {
      String pkgContents = TestRuleClassProvider.getRuleClassProvider().getDefaultsPackageContent();
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
      skyframeExecutor.invalidateFilesUnderPathForTesting(storedErrors, modified, workspace);

      changes.clear();
    }

    public List<Target> getTargets(String... targetNames) throws Exception {
      List<Target> result = new ArrayList<>();
      for (String targetName : targetNames) {
        result.add(getTarget(targetName));
      }
      return result;
    }

    public Target getTarget(String targetName) throws Exception {
      StoredEventHandler eventHandler = new StoredEventHandler();
      Target target = getPkgManager().getTarget(
          eventHandler, Label.parseAbsoluteUnchecked(targetName));
      assertFalse(eventHandler.hasErrors());
      return target;
    }

    private PackageManager getPkgManager() {
      return skyframeExecutor.getPackageManager();
    }

    public ImmutableSet<Target> getFilteredTargets() {
      return targetParsingCompleteEvent.getFilteredTargets();
    }

    public ImmutableSet<Target> getTestFilteredTargets() {
      return targetParsingCompleteEvent.getTestFilteredTargets();
    }

    public ImmutableSet<Target> getOriginalTargets() {
      return targetParsingCompleteEvent.getTargets();
    }

    public ImmutableSet<Target> getTestSuiteTargets() {
      return loadingPhaseCompleteEvent.getFilteredTargets();
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
  }

  public static class FilteredTargetListener {
    private TargetParsingCompleteEvent targetParsingCompleteEvent;
    private LoadingPhaseCompleteEvent loadingPhaseCompleteEvent;

    @Subscribe
    public void targetParsingComplete(TargetParsingCompleteEvent event) {
      this.targetParsingCompleteEvent = event;
    }

    @Subscribe
    public void loadingPhaseComplete(LoadingPhaseCompleteEvent event) {
      this.loadingPhaseCompleteEvent = event;
    }
  }
}
