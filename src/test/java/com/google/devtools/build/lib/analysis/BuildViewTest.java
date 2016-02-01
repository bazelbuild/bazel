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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertEventCount;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.DeterministicInMemoryGraph;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.EventType;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Listener;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Order;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.TrackingAwaiter;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * Tests for the {@link BuildView}.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class BuildViewTest extends BuildViewTestBase {

  @Test
  public void testRuleConfiguredTarget() throws Exception {
    scratch.file("pkg/BUILD",
        "genrule(name='foo', ",
        "        cmd = '',",
        "        srcs=['a.src'],",
        "        outs=['a.out'])");
    update("//pkg:foo");
    Rule ruleTarget = (Rule) getTarget("//pkg:foo");
    assertEquals("genrule", ruleTarget.getRuleClass());

    ConfiguredTarget ruleCT = getConfiguredTarget("//pkg:foo");

    assertSame(ruleTarget, ruleCT.getTarget());
  }

  @Test
  public void testFilterByTargets() throws Exception {
    scratch.file("tests/BUILD",
        "sh_test(name = 'small_test_1',",
        "        srcs = ['small_test_1.sh'],",
        "        data = [':xUnit'],",
        "        size = 'small',",
        "        tags = ['tag1'])",
        "",
        "sh_test(name = 'small_test_2',",
        "        srcs = ['small_test_2.sh'],",
        "        size = 'small',",
        "        tags = ['tag2'])",
        "",
        "",
        "test_suite( name = 'smallTests', tags=['small'])");
    //scratch.file("tests/small_test_1.py");

    update("//tests:smallTests");

    ConfiguredTarget test1 = getConfiguredTarget("//tests:small_test_1");
    ConfiguredTarget test2 = getConfiguredTarget("//tests:small_test_2");
    ConfiguredTarget suite = getConfiguredTarget("//tests:smallTests");
    assertNoEvents(); // start from a clean slate


    Collection<ConfiguredTarget> targets =
        new LinkedHashSet<>(ImmutableList.of(test1, test2, suite));
    targets = Lists.newArrayList(
        BuildView.filterTestsByTargets(targets,
            Sets.newHashSet(test1.getTarget(), suite.getTarget())));
    assertThat(targets).containsExactlyElementsIn(Sets.newHashSet(test1, suite));
  }

  @Test
  public void testSourceArtifact() throws Exception {
    setupDummyRule();
    update("//pkg:a.src");
    InputFileConfiguredTarget inputCT = getInputFileConfiguredTarget("//pkg:a.src");
    Artifact inputArtifact = inputCT.getArtifact();
    assertNull(getGeneratingAction(inputArtifact));
    assertEquals("pkg/a.src", inputArtifact.getExecPathString());
  }

  @Test
  public void testGeneratedArtifact() throws Exception {
    setupDummyRule();
    update("//pkg:a.out");
    OutputFileConfiguredTarget outputCT = (OutputFileConfiguredTarget)
        getConfiguredTarget("//pkg:a.out");
    Artifact outputArtifact = outputCT.getArtifact();
    assertEquals(getTargetConfiguration().getBinDirectory(), outputArtifact.getRoot());
    assertEquals(getTargetConfiguration().getBinFragment().getRelative("pkg/a.out"),
        outputArtifact.getExecPath());
    assertEquals(new PathFragment("pkg/a.out"), outputArtifact.getRootRelativePath());

    Action action = getGeneratingAction(outputArtifact);
    assertSame(FailAction.class, action.getClass());
  }

  @Test
  public void testReportsAnalysisRootCauses() throws Exception {
    scratch.file("private/BUILD",
        "genrule(",
        "    name='private',",
        "    outs=['private.out'],",
        "    cmd='',",
        "    visibility=['//visibility:private'])");
    scratch.file("foo/BUILD",
        "genrule(",
        "    name='foo',",
        "    tools=[':bar'],",
        "    outs=['foo.out'],",
        "    cmd='')",
        "genrule(",
        "    name='bar',",
        "    tools=['//private'],",
        "    outs=['bar.out'],",
        "    cmd='')");

    reporter.removeHandler(failFastHandler);
    EventBus eventBus = new EventBus();
    AnalysisFailureRecorder recorder = new AnalysisFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events).hasSize(1);
    AnalysisFailureEvent event = recorder.events.get(0);
    assertEquals("//foo:bar", event.getFailureReason().toString());
    assertEquals("//foo:foo", event.getFailedTarget().getLabel().toString());
  }

  @Test
  public void testReportsLoadingRootCauses() throws Exception {
    scratch.file("pkg/BUILD",
        "genrule(name='foo',",
        "        tools=['//nopackage:missing'],",
        "        cmd='')");

    reporter.removeHandler(failFastHandler);
    EventBus eventBus = new EventBus();
    LoadingFailureRecorder recorder = new LoadingFailureRecorder();
    eventBus.register(recorder);
    // Note: no need to run analysis for a loading failure.
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//pkg:foo");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events)
        .contains(
            Pair.of(Label.parseAbsolute("//pkg:foo"), Label.parseAbsolute("//nopackage:missing")));
    assertContainsEvent("missing value for mandatory attribute 'outs'");
    assertContainsEvent("no such package 'nopackage'");
    // Skyframe correctly reports the other root cause as the genrule itself (since it is
    // missing attributes).
    assertThat(recorder.events).hasSize(2);
    assertThat(recorder.events)
        .contains(Pair.of(Label.parseAbsolute("//pkg:foo"), Label.parseAbsolute("//pkg:foo")));
  }

  @Test
  public void testConvolutedLoadRootCauseAnalysis() throws Exception {
    // You need license declarations in third_party. We use this constraint to
    // create targets that are loadable, but are in error.
    scratch.file("third_party/first/BUILD",
        "sh_library(name='first', deps=['//third_party/second'], licenses=['notice'])");
    scratch.file("third_party/second/BUILD",
        "sh_library(name='second', deps=['//third_party/third'], licenses=['notice'])");
    scratch.file("third_party/third/BUILD",
        "sh_library(name='third', deps=['//third_party/fourth'], licenses=['notice'])");
    scratch.file("third_party/fourth/BUILD",
        "sh_library(name='fourth', deps=['//third_party/fifth'])");
    scratch.file("third_party/fifth/BUILD",
        "sh_library(name='fifth', licenses=['notice'])");
    reporter.removeHandler(failFastHandler);
    EventBus eventBus = new EventBus();
    LoadingFailureRecorder recorder = new LoadingFailureRecorder();
    eventBus.register(recorder);
    // Note: no need to run analysis for a loading failure.
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING),
        "//third_party/first", "//third_party/third");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events).hasSize(2);
    assertTrue(recorder.events.toString(), recorder.events.contains(
        Pair.of(Label.parseAbsolute("//third_party/first"),
            Label.parseAbsolute("//third_party/fourth"))));
    assertThat(recorder.events)
        .contains(Pair.of(
            Label.parseAbsolute("//third_party/third"),
            Label.parseAbsolute("//third_party/fourth")));
  }

  @Test
  public void testMultipleRootCauseReporting() throws Exception {
    scratch.file("gp/BUILD",
        "sh_library(name = 'gp', deps = ['//p:p'])");
    scratch.file("p/BUILD",
        "sh_library(name = 'p', deps = ['//c1:not', '//c2:not'])");
    scratch.file("c1/BUILD");
    scratch.file("c2/BUILD");
    reporter.removeHandler(failFastHandler);
    EventBus eventBus = new EventBus();
    LoadingFailureRecorder recorder = new LoadingFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//gp");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events).hasSize(2);
    assertTrue(recorder.events.toString(), recorder.events.contains(
        Pair.of(Label.parseAbsolute("//gp"),
            Label.parseAbsolute("//c1:not"))));
    assertThat(recorder.events)
        .contains(Pair.of(Label.parseAbsolute("//gp"), Label.parseAbsolute("//c2:not")));
  }

  /**
   * Regression test for: "Package group includes are broken"
   */
  @Test
  public void testTopLevelPackageGroup() throws Exception {
    scratch.file("tropical/BUILD",
        "package_group(name='guava', includes=[':mango'])",
        "package_group(name='mango')");

    // If the analysis phase results in an error, this will throw an exception
    update("//tropical:guava");

    // Check if the included package group also got analyzed
    assertNotNull(getConfiguredTarget("//tropical:mango", null));
  }

  @Test
  public void testTopLevelInputFile() throws Exception {
    scratch.file("tropical/BUILD",
        "exports_files(['file.txt'])");
    update("//tropical:file.txt");
    assertNotNull(getConfiguredTarget("//tropical:file.txt", null));
  }

  @Test
  public void testGetDirectPrerequisites() throws Exception {
    scratch.file(
        "package/BUILD",
        "filegroup(name='top', srcs=[':inner', 'file'])",
        "sh_binary(name='inner', srcs=['script.sh'])");
    update("//package:top");
    ConfiguredTarget top = getConfiguredTarget("//package:top", getTargetConfiguration());
    Iterable<ConfiguredTarget> targets = getView().getDirectPrerequisitesForTesting(
        reporter, top, getBuildConfigurationCollection());
    Iterable<Label> labels =
        Iterables.transform(
            targets,
            new Function<ConfiguredTarget, Label>() {
              @Override
              public Label apply(ConfiguredTarget target) {
                return target.getLabel();
              }
            });
    assertThat(labels)
        .containsExactly(
            Label.parseAbsolute("//package:inner"), Label.parseAbsolute("//package:file"));
  }

  @Test
  public void testGetDirectPrerequisiteDependencies() throws Exception {
    scratch.file(
        "package/BUILD",
        "filegroup(name='top', srcs=[':inner', 'file'])",
        "sh_binary(name='inner', srcs=['script.sh'])");
    update("//package:top");
    ConfiguredTarget top = getConfiguredTarget("//package:top", getTargetConfiguration());
    Iterable<Dependency> targets = getView().getDirectPrerequisiteDependenciesForTesting(
        reporter, top, getBuildConfigurationCollection()).values();

    Dependency innerDependency;
    Dependency fileDependency;
    if (top.getConfiguration().useDynamicConfigurations()) {
      innerDependency =
          Dependency.withTransitionAndAspects(
              Label.parseAbsolute("//package:inner"),
              Attribute.ConfigurationTransition.NONE,
              ImmutableSet.<Aspect>of());
      fileDependency =
          Dependency.withTransitionAndAspects(
              Label.parseAbsolute("//package:file"),
              Attribute.ConfigurationTransition.NONE,
              ImmutableSet.<Aspect>of());
    } else {
      innerDependency =
          Dependency.withConfiguration(
              Label.parseAbsolute("//package:inner"),
              getTargetConfiguration());
      fileDependency =
          Dependency.withNullConfiguration(
              Label.parseAbsolute("//package:file"));
    }

    assertThat(targets).containsExactly(innerDependency, fileDependency);
  }

  /**
   * Tests that the {@code --configuration short name} option cannot be used on
   * the command line.
   */
  @Test
  public void testConfigurationShortName() throws Exception {
    useConfiguration("--output directory name=foo");
    reporter.removeHandler(failFastHandler);
    try {
      update(defaultFlags());
      fail();
    } catch (InvalidConfigurationException e) {
      assertThat(e).hasMessage("Build options are invalid");
      assertContainsEvent(
          "The internal '--output directory name' option cannot be used on the command line");
    }
  }

  @Test
  public void testFileTranslations() throws Exception {
    scratch.file("foo/file");
    scratch.file("foo/BUILD",
        "exports_files(['file'])");
    useConfiguration("--message_translations=//foo:file");
    scratch.file("bar/BUILD",
        "sh_library(name = 'bar')");
    update("//bar");
  }

  // Regression test: "output_filter broken (but in a different way)"
  @Test
  public void testOutputFilterSeeWarning() throws Exception {
    runAnalysisWithOutputFilter(Pattern.compile(".*"));
    assertContainsEvent("please do not import '//java/a:A.java'");
  }

  // Regression test: "output_filter broken (but in a different way)"
  @Test
  public void testOutputFilter() throws Exception {
    runAnalysisWithOutputFilter(Pattern.compile("^//java/c"));
    assertNoEvents();
  }

  @Test
  public void testAnalysisErrorMessageWithKeepGoing() throws Exception {
    scratch.file("a/BUILD", "sh_binary(name='a', srcs=['a1.sh', 'a2.sh'])");
    reporter.removeHandler(failFastHandler);
    AnalysisResult result = update(defaultFlags().with(Flag.KEEP_GOING), "//a");
    assertThat(result.hasError()).isTrue();
    assertContainsEvent("errors encountered while analyzing target '//a:a'");
  }

  /**
   * Regression test: Exception in ConfiguredTargetGraph.checkForCycles()
   * when multiple top-level targets depend on the same cycle.
   */
  @Test
  public void testCircularDependencyBelowTwoTargets() throws Exception {
    scratch.file("foo/BUILD",
        "sh_library(name = 'top1', srcs = ['top1.sh'], deps = [':rec1'])",
        "sh_library(name = 'top2', srcs = ['top2.sh'], deps = [':rec1'])",
        "sh_library(name = 'rec1', srcs = ['rec1.sh'], deps = [':rec2'])",
        "sh_library(name = 'rec2', srcs = ['rec2.sh'], deps = [':rec1'])"
    );
    reporter.removeHandler(failFastHandler);
    AnalysisResult result =
        update(defaultFlags().with(Flag.KEEP_GOING), "//foo:top1", "//foo:top2");
    assertThat(result.hasError()).isTrue();
    assertContainsEvent("in sh_library rule //foo:rec1: cycle in dependency graph:\n");
    assertContainsEvent("in sh_library rule //foo:top");
  }

  // Regression test: cycle node depends on error.
  // Note that this test can have nondeterministic behavior in Skyframe, depending on if the cycle
  // is detected during the bubbling-up phase.
  @Test
  public void testErrorBelowCycle() throws Exception {
    scratch.file("foo/BUILD",
        "sh_library(name = 'top', deps = ['mid'])",
        "sh_library(name = 'mid', deps = ['bad', 'cycle1'])",
        "sh_library(name = 'bad', srcs = ['//badbuild:isweird'])",
        "sh_library(name = 'cycle1', deps = ['cycle2', 'mid'])",
        "sh_library(name = 'cycle2', deps = ['cycle1'])");
    scratch.file("badbuild/BUILD", "");
    reporter.removeHandler(failFastHandler);
    try {
      update("//foo:top");
      fail();
    } catch (LoadingFailedException | ViewCreationFailedException e) {
      // Expected.
    }
    assertContainsEvent("no such target '//badbuild:isweird': target 'isweird' not declared in "
        + "package 'badbuild'");
    assertContainsEvent("and referenced by '//foo:bad'");
    if (eventCollector.count() > 1) {
      assertContainsEvent("in sh_library rule //foo");
      assertContainsEvent("cycle in dependency graph");
      assertEventCount(3, eventCollector);
    }
  }

  @Test
  public void testAnalysisEntryHasActionsEvenWithError() throws Exception {
    scratch.file("foo/BUILD",
        "cc_binary(name = 'foo', linkshared = 1, srcs = ['foo.cc'])");
    reporter.removeHandler(failFastHandler);
    try {
      update("//foo:foo");
      fail(); // Expected ViewCreationFailedException.
    } catch (ViewCreationFailedException e) {
      // ok.
    }
  }

  @Test
  public void testHelpfulErrorForWrongPackageLabels() throws Exception {
    reporter.removeHandler(failFastHandler);

    scratch.file("x/BUILD",
        "cc_library(name='x', srcs=['x.cc'])");
    scratch.file("y/BUILD",
        "cc_library(name='y', srcs=['y.cc'], deps=['//x:z'])");

    AnalysisResult result = update(defaultFlags().with(Flag.KEEP_GOING), "//y:y");
    assertThat(result.hasError()).isTrue();
    assertContainsEvent("no such target '//x:z': "
        + "target 'z' not declared in package 'x' "
        + "defined by /workspace/x/BUILD and referenced by '//y:y'");
  }

  @Test
  public void testNewActionsAreDifferentAndDontConflict() throws Exception {
    scratch.file("pkg/BUILD",
        "genrule(name='a', ",
        "        cmd='',",
        "        outs=['a.out'])");
    update("//pkg:a.out");
    OutputFileConfiguredTarget outputCT = (OutputFileConfiguredTarget)
        getConfiguredTarget("//pkg:a.out");
    Artifact outputArtifact = outputCT.getArtifact();
    Action action = getGeneratingAction(outputArtifact);
    assertNotNull(action);
    scratch.overwriteFile("pkg/BUILD",
        "genrule(name='a', ",
        "        cmd='false',",
        "        outs=['a.out'])");
    update("//pkg:a.out");
    assertFalse("Actions should not be compatible",
        Actions.canBeShared(action, getGeneratingAction(outputArtifact)));
  }

  /**
   * This test exercises the case where we invalidate (mark dirty) a node in one build command
   * invocation and the revalidation (because it did not change) happens in a subsequent build
   * command call.
   *
   * - In the first update call we construct A.
   *
   * - Then we construct B and we make the glob get invalidated. We do that by deleting a file
   * because it depends on the directory listing. Because of that A gets invalidated.
   *
   * - Then we construct A again. The glob gets revalidated because it is still matching just A.java
   * and A configured target gets revalidated too. At the end of the analysis A java action should
   * be in the action graph.
   */
  @Test
  public void testMultiBuildInvalidationRevalidation() throws Exception {
    scratch.file("java/a/A.java", "bla1");
    scratch.file("java/a/C.java", "bla2");
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = glob(['A*.java']))",
        "java_test(name = 'B',",
        "          srcs = ['B.java'])");
    update("//java/a:A");
    ConfiguredTarget ct = getConfiguredTarget("//java/a:A");
    scratch.deleteFile("java/a/C.java");
    update("//java/a:B");
    update("//java/a:A");
    assertNotNull(getGeneratingAction(
        getBinArtifact("A_deploy.jar", ct)));
  }

  /**
   * Regression test: ClassCastException in SkyframeLabelVisitor.updateRootCauses.
   */
  @Test
  public void testDepOnGoodTargetInBadPkgAndTransitivelyBadTarget() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("parent/BUILD",
        "sh_library(name = 'foo',",
        "           srcs = ['//badpkg1:okay-target', '//okaypkg:transitively-bad-target'])");
    Path badpkg1BuildFile = scratch.file("badpkg1/BUILD",
        "exports_files(['okay-target'])",
        "invalidbuildsyntax");
    scratch.file("okaypkg/BUILD",
        "sh_library(name = 'transitively-bad-target',",
        "           srcs = ['//badpkg2:bad-target'])");
    Path badpkg2BuildFile = scratch.file("badpkg2/BUILD",
        "sh_library(name = 'bad-target')",
        "invalidbuildsyntax");
    update(defaultFlags().with(Flag.KEEP_GOING), "//parent:foo");
    assertEquals(1, getFrequencyOfErrorsWithLocation(
        badpkg1BuildFile.asFragment(), eventCollector));
    assertEquals(1, getFrequencyOfErrorsWithLocation(
        badpkg2BuildFile.asFragment(), eventCollector));
  }

  @Test
  public void testDepOnGoodTargetInBadPkgAndTransitiveCycle_NotIncremental() throws Exception {
    runTestDepOnGoodTargetInBadPkgAndTransitiveCycle(/*incremental=*/false);
  }

  @Test
  public void testDepOnGoodTargetInBadPkgAndTransitiveCycle_Incremental() throws Exception {
    runTestDepOnGoodTargetInBadPkgAndTransitiveCycle(/*incremental=*/true);
  }

  /**
   * Regression test: in keep_going mode, cycles in target graph aren't reported
   * if package is in error.
   */
  @Test
  public void testCycleReporting_TargetCycleWhenPackageInError() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("cycles/BUILD",
        "sh_library(name = 'a', deps = [':b'])",
        "sh_library(name = 'b', deps = [':a'])",
        "notvalidbuildsyntax");
    update(defaultFlags().with(Flag.KEEP_GOING), "//cycles:a");
    assertContainsEvent("'notvalidbuildsyntax'");
    assertContainsEvent("cycle in dependency graph");
  }

  @Test
  public void testTransitiveLoadingDoesntShortCircuitInKeepGoing() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("parent/BUILD",
        "sh_library(name = 'a', deps = ['//child:b'])",
        "parentisbad");
    scratch.file("child/BUILD",
        "sh_library(name = 'b')",
        "childisbad");
    update(defaultFlags().with(Flag.KEEP_GOING), "//parent:a");
    assertContainsEventWithFrequency("parentisbad", 1);
    assertContainsEventWithFrequency("childisbad", 1);
    assertContainsEventWithFrequency("and referenced by '//parent:a'", 1);
  }

  /**
   * Smoke test for the Skyframe code path.
   */
  @Test
  public void testSkyframe() throws Exception {
    setupDummyRule();
    String aoutLabel = "//pkg:a.out";

    update(aoutLabel);

    // However, a ConfiguredTarget was actually produced.
    ConfiguredTarget target = Iterables.getOnlyElement(getAnalysisResult().getTargetsToBuild());
    assertEquals(aoutLabel, target.getLabel().toString());

    Artifact aout = Iterables.getOnlyElement(
        target.getProvider(FileProvider.class).getFilesToBuild());
    Action action = getGeneratingAction(aout);
    assertSame(FailAction.class, action.getClass());
  }

  /**
   * ConfiguredTargetFunction should not register actions in legacy Blaze ActionGraph unless
   * the creation of the node is successful.
   */
  @Test
  public void testActionsNotRegisteredInLegacyWhenError() throws Exception {
    // First find the artifact we want to make sure is not generated by an action with an error.
    // Then update the BUILD file and re-analyze.
    scratch.file("actions_not_registered/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    update("//actions_not_registered:foo");
    Artifact fooOut = Iterables.getOnlyElement(
        getConfiguredTarget("//actions_not_registered:foo")
            .getProvider(FileProvider.class).getFilesToBuild());
    assertNotNull(getActionGraph().getGeneratingAction(fooOut));
    clearAnalysisResult();

    scratch.overwriteFile("actions_not_registered/BUILD",
        "cc_binary(name = 'foo', linkshared = 1, srcs = ['foo.cc'])");

    reporter.removeHandler(failFastHandler);

    try {
      update("//actions_not_registered:foo");
      fail("This build should fail because: 'linkshared' used in non-shared library");
    } catch (ViewCreationFailedException e) {
      assertNull(getActionGraph().getGeneratingAction(fooOut));
    }
  }

  /**
   * Regression test:
   * "skyframe: ArtifactFactory and ConfiguredTargets out of sync".
   */
  @Test
  public void testSkyframeAnalyzeRuleThenItsOutputFile() throws Exception {
    scratch.file("pkg/BUILD",
        "testing_dummy_rule(name='foo', ",
        "                   srcs=['a.src'],",
        "                   outs=['a.out'])");

    scratch.file("pkg2/BUILD",
        "testing_dummy_rule(name='foo', ",
        "                   srcs=['a.src'],",
        "                   outs=['a.out'])");
    String aoutLabel = "//pkg:a.out";

    update("//pkg2:foo");
    update("//pkg:foo");
    scratch.overwriteFile("pkg2/BUILD",
        "testing_dummy_rule(name='foo', ",
        "                   srcs=['a.src'],",
        "                   outs=['a.out'])",
        "# Comment");

    update("//pkg:a.out");

    // However, a ConfiguredTarget was actually produced.
    ConfiguredTarget target = Iterables.getOnlyElement(getAnalysisResult().getTargetsToBuild());
    assertEquals(aoutLabel, target.getLabel().toString());

    Artifact aout = Iterables.getOnlyElement(
        target.getProvider(FileProvider.class).getFilesToBuild());
    Action action = getGeneratingAction(aout);
    assertSame(FailAction.class, action.getClass());
  }

  /**
   * Tests that skyframe reports the root cause as being the target that depended on the symlink
   * cycle.
   */
  @Test
  public void testRootCauseReportingFileSymlinks() throws Exception {
    scratch.file("gp/BUILD",
        "sh_library(name = 'gp', deps = ['//p'])");
    scratch.file("p/BUILD",
        "sh_library(name = 'p', deps = ['//c'])");
    scratch.file("c/BUILD",
        "sh_library(name = 'c', deps = [':c1', ':c2'])",
        "sh_library(name = 'c1', deps = ['//cycles1'])",
        "sh_library(name = 'c2', deps = ['//cycles2'])");
    Path cycles1BuildFilePath = scratch.file("cycles1/BUILD",
        "sh_library(name = 'cycles1', srcs = glob(['*.sh']))");
    Path cycles2BuildFilePath = scratch.file("cycles2/BUILD",
        "sh_library(name = 'cycles2', srcs = glob(['*.sh']))");
    cycles1BuildFilePath.getParentDirectory().getRelative("cycles1.sh").createSymbolicLink(
        new PathFragment("cycles1.sh"));
    cycles2BuildFilePath.getParentDirectory().getRelative("cycles2.sh").createSymbolicLink(
        new PathFragment("cycles2.sh"));
    reporter.removeHandler(failFastHandler);
    EventBus eventBus = new EventBus();
    LoadingFailureRecorder recorder = new LoadingFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//gp");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events).hasSize(2);
    assertTrue(recorder.events.toString(), recorder.events.contains(
        Pair.of(Label.parseAbsolute("//gp"), Label.parseAbsolute("//cycles1"))));
    assertTrue(recorder.events.toString(), recorder.events.contains(
        Pair.of(Label.parseAbsolute("//gp"), Label.parseAbsolute("//cycles2"))));
  }

  /**
   * Regression test for bug when a configured target has missing deps, but also depends
   * transitively on an error. We build //foo:query, which depends on a valid and an invalid target
   * pattern. We ensure that by the time it requests its dependent target patterns, the invalid one
   * is ready, and throws (though not before the request is registered). Then, when bubbling the
   * invalid target pattern error up, we ensure that it bubbles into //foo:query, which must cope
   * with the combination of an error and a missing dep.
   */
  @Test
  public void testGenQueryWithBadTargetAndUnfinishedTarget() throws Exception {
    // The target //foo:zquery is used to force evaluation of //foo:nosuchtarget before the target
    // patterns in //foo:query are enqueued for evaluation. That way, //foo:query will depend on one
    // invalid target pattern and two target patterns that haven't been evaluated yet.
    // It is important that 'query' come before 'zquery' alphabetically, so that when the error is
    // bubbling up, it goes to the //foo:query node -- we use a graph implementation in which the
    // reverse deps of each entry are ordered alphabetically. It is also important that a missing
    // target pattern is requested before the exception is thrown, so we have both //foo:b and
    // //foo:z missing from the deps, in the hopes that at least one of them will come before
    // //foo:nosuchtarget.
    scratch.file("foo/BUILD",
        "genquery(name = 'query',",
        "         expression = 'deps(//foo:b) except //foo:nosuchtarget except //foo:z',",
        "         scope = ['//foo:a'])",
        "genquery(name = 'zquery',",
        "         expression = 'deps(//foo:nosuchtarget)',",
        "         scope = ['//foo:a'])",
        "sh_library(name = 'a')",
        "sh_library(name = 'b')",
        "sh_library(name = 'z')"
    );
    Listener listener =
        new Listener() {
          private final CountDownLatch errorDone = new CountDownLatch(1);
          private final CountDownLatch realQueryStarted = new CountDownLatch(1);

          @Override
          public void accept(SkyKey key, EventType type, Order order, Object context) {
            if (!key.functionName().equals(SkyFunctions.TARGET_PATTERN)) {
              return;
            }
            String label = ((TargetPatternKey) key.argument()).getPattern();
            if (label.equals("//foo:nosuchtarget")) {
              if (type == EventType.SET_VALUE) {
                // Inform //foo:query-dep-registering thread that it may proceed.
                errorDone.countDown();
                // Wait to make sure //foo:query-dep-registering process has started.
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    realQueryStarted, "//foo:query did not request dep in time");
              } else if (type == EventType.ADD_REVERSE_DEP
                  && context.toString().contains("foo:query")) {
                // Make sure that when foo:query requests foo:nosuchtarget, it's already done.
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    errorDone, "//foo:nosuchtarget did not evaluate in time");
              }
            } else if ((label.equals("//foo:b") || label.equals("//foo:z"))
                && type == EventType.CREATE_IF_ABSENT) {
              // Inform error-evaluating thread that it may throw an exception.
              realQueryStarted.countDown();
              TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                  errorDone, "//foo:nosuchtarget did not evaluate in time");
              // Don't let the target pattern //foo:{b,z} get enqueued for evaluation until we
              // receive an interrupt signal from the threadpool. The interrupt means that
              // evaluation is shutting down, and so //foo:{b,z} definitely won't get evaluated.
              CountDownLatch waitForInterrupt = new CountDownLatch(1);
              try {
                waitForInterrupt.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
                throw new IllegalStateException("node was not interrupted in time");
              } catch (InterruptedException e) {
                // Expected.
                Thread.currentThread().interrupt();
              }
            }
          }
        };
    NotifyingInMemoryGraph graph = new DeterministicInMemoryGraph(listener);
    setGraphForTesting(graph);
    reporter.removeHandler(failFastHandler);
    try {
      update("//foo:query", "//foo:zquery");
      fail();
    } catch (ViewCreationFailedException e) {
      Truth.assertThat(e.getMessage())
          .contains("Analysis of target '//foo:query' failed; build aborted");
    }
    TrackingAwaiter.INSTANCE.assertNoErrors();
    graph.assertNoExceptions();
  }

  /**
   * Tests that rules with configurable attributes can be accessed through {@link
   * com.google.devtools.build.lib.skyframe.PostConfiguredTargetFunction}.
   * This is a regression test for a Bazel crash.
   */
  @Test
  public void testPostProcessedConfigurableAttributes() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors from action conflicts.
    scratch.file("conflict/BUILD",
        "config_setting(name = 'a', values = {'test_arg': 'a'})",
        "cc_library(name='x', srcs=select({':a': ['a.cc'], '//conditions:default': ['foo.cc']}))",
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    AnalysisResult result = update(
        defaultFlags().with(Flag.KEEP_GOING),
        "//conflict:_objs/x/conflict/foo.pic.o",
        "//conflict:x");
    assertThat(result.hasError()).isTrue();
    // Expect to reach this line without a Precondition-triggered NullPointerException.
    assertContainsEvent(
        "file 'conflict/_objs/x/conflict/foo.pic.o' is generated by these conflicting actions");
  }

  @Test
  public void testCycleDueToJavaLauncherConfiguration() throws Exception {
    scratch.file("foo/BUILD",
        "java_binary(name = 'java', srcs = ['DoesntMatter.java'])",
        "cc_binary(name = 'cpp', data = [':java'])");
    // Everything is fine - the dependency graph is acyclic.
    update("//foo:java", "//foo:cpp");
    // Now there will be an analysis-phase cycle because the java_binary now has an implicit dep on
    // the cc_binary launcher.
    useConfiguration("--java_launcher=//foo:cpp");
    reporter.removeHandler(failFastHandler);
    try {
      update("//foo:java", "//foo:cpp");
      fail();
    } catch (ViewCreationFailedException expected) {
      Truth.assertThat(expected.getMessage())
          .matches("Analysis of target '//foo:(java|cpp)' failed; build aborted.*");
    }
    assertContainsEvent("cycle in dependency graph");
    assertContainsEvent("This cycle occurred because of a configuration option");
  }

  @Test
  public void testDependsOnBrokenTarget() throws Exception {
    scratch.file("foo/BUILD",
        "sh_test(name = 'test', srcs = ['test.sh'], data = ['//bar:data'])");
    scratch.file("bar/BUILD",
        "BROKEN BROKEN BROKEN!!!");
    reporter.removeHandler(failFastHandler);
    try {
      update("//foo:test");
      fail();
    } catch (LoadingFailedException expected) {
      Truth.assertThat(expected.getMessage())
          .matches("Loading failed; build aborted.*");
    } catch (ViewCreationFailedException expected) {
      Truth.assertThat(expected.getMessage())
          .matches("Analysis of target '//foo:test' failed; build aborted.*");
    }
  }

  /** Runs the same test with the reduced loading phase. */
  @TestSpec(size = Suite.SMALL_TESTS)
  @RunWith(JUnit4.class)
  public static class WithSkyframeLoadingPhase extends BuildViewTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags().with(Flag.SKYFRAME_LOADING_PHASE);
    }
  }
}
