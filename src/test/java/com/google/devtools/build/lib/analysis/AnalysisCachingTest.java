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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.util.AnalysisCachingTestBase;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Analysis caching tests.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class AnalysisCachingTest extends AnalysisCachingTestBase {

  @Test
  public void testSimpleCleanAnalysis() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])");
    update("//java/a:A");
    ConfiguredTarget javaTest = getConfiguredTarget("//java/a:A");
    assertNotNull(javaTest);
    assertNotNull(javaTest.getProvider(JavaSourceJarsProvider.class));
  }

  @Test
  public void testTickTock() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])",
        "java_test(name = 'B',",
        "          srcs = ['B.java'])");
    update("//java/a:A");
    update("//java/a:B");
    update("//java/a:A");
  }

  @Test
  public void testFullyCached() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])");
    update("//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertSame(old, current);
  }

  @Test
  public void testSubsetCached() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])",
        "java_test(name = 'B',",
        "          srcs = ['B.java'])");
    update("//java/a:A", "//java/a:B");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertSame(old, current);
  }

  @Test
  public void testDependencyChanged() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'],",
        "          deps = ['//java/b'])");
    scratch.file("java/b/BUILD",
        "java_library(name = 'b',",
        "             srcs = ['B.java'])");
    update("//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    scratch.overwriteFile("java/b/BUILD",
        "java_library(name = 'b',",
        "             srcs = ['C.java'])");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertNotSame(old, current);
  }

  @Test
  public void testTopLevelChanged() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'],",
        "          deps = ['//java/b'])");
    scratch.file("java/b/BUILD",
        "java_library(name = 'b',",
        "             srcs = ['B.java'])");
    update("//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    scratch.overwriteFile("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertNotSame(old, current);
  }

  // Regression test for:
  // "action conflict detection is incorrect if conflict is in non-top-level configured targets".
  @Test
  public void testActionConflictInDependencyImpliesTopLevelTargetFailure() throws Exception {
    scratch.file("conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])",
        "cc_binary(name='foo', deps=['x'], data=['_objs/x/conflict/foo.pic.o'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:foo");
    assertContainsEvent("file 'conflict/_objs/x/conflict/foo.pic.o' " + CONFLICT_MSG);
    assertThat(getAnalysisResult().getTargetsToBuild()).isEmpty();
  }

  /**
   * Generating the same output from two targets is ok if we build them on successive builds
   * and invalidate the first target before we build the second target. This is a strictly weaker
   * test than if we didn't invalidate the first target, but since Skyframe can't pass then, this
   * test could be useful for it. Actually, since Skyframe makes multiple update calls, it manages
   * to unregister actions even when it shouldn't, and so this test can incorrectly pass. However,
   * {@code SkyframeExecutorTest#testNoActionConflictWithInvalidatedTarget} tests it more
   * rigorously.
   */
  @Test
  public void testNoActionConflictWithInvalidatedTarget() throws Exception {
    scratch.file("conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    update("//conflict:x");
    ConfiguredTarget conflict = getConfiguredTarget("//conflict:x");
    Action oldAction = getGeneratingAction(getBinArtifact("_objs/x/conflict/foo.pic.o", conflict));
    assertEquals("//conflict:x", oldAction.getOwner().getLabel().toString());
    scratch.overwriteFile("conflict/BUILD",
        "cc_library(name='newx', srcs=['foo.cc'])", // Rename target.
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    update(defaultFlags(), "//conflict:_objs/x/conflict/foo.pic.o");
    ConfiguredTarget objsConflict = getConfiguredTarget("//conflict:_objs/x/conflict/foo.pic.o");
    Action newAction =
        getGeneratingAction(getBinArtifact("_objs/x/conflict/foo.pic.o", objsConflict));
    assertEquals("//conflict:_objs/x/conflict/foo.pic.o",
        newAction.getOwner().getLabel().toString());
  }

  /**
   * Generating the same output from multiple actions is causing an error.
   */
  @Test
  public void testActionConflictCausesError() throws Exception {
    scratch.file("conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING),
        "//conflict:x", "//conflict:_objs/x/conflict/foo.pic.o");
    assertContainsEvent("file 'conflict/_objs/x/conflict/foo.pic.o' " + CONFLICT_MSG);
  }

  @Test
  public void testNoActionConflictErrorAfterClearedAnalysis() throws Exception {
    scratch.file("conflict/BUILD",
                "cc_library(name='x', srcs=['foo.cc'])",
                "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING),
        "//conflict:x", "//conflict:_objs/x/conflict/foo.pic.o");
    // We want to force a "dropConfiguredTargetsNow" operation, which won't inform the
    // invalidation receiver about the dropped configured targets.
    getView().clearAnalysisCache(ImmutableList.<ConfiguredTarget>of());
    assertContainsEvent("file 'conflict/_objs/x/conflict/foo.pic.o' " + CONFLICT_MSG);
    eventCollector.clear();
    scratch.overwriteFile("conflict/BUILD",
        "cc_library(name='x', srcs=['baz.cc'])",
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    update(defaultFlags().with(Flag.KEEP_GOING),
        "//conflict:x", "//conflict:_objs/x/conflict/foo.pic.o");
    assertNoEvents();
  }
  
  /**
   * For two conflicting actions whose primary inputs are different, no list diff detail should be
   * part of the output.
   */
  @Test
  public void testConflictingArtifactsErrorWithNoListDetail() throws Exception {
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(
        defaultFlags().with(Flag.KEEP_GOING),
        "//conflict:x",
        "//conflict:_objs/x/conflict/foo.pic.o");

    assertContainsEvent("file 'conflict/_objs/x/conflict/foo.pic.o' " + CONFLICT_MSG);
    assertDoesNotContainEvent("MandatoryInputs");
    assertDoesNotContainEvent("Outputs");
  }

  /**
   * For two conflicted actions whose primary inputs are the same, list diff (max 5) should be part
   * of the output.
   */
  @Test
  public void testConflictingArtifactsWithListDetail() throws Exception {
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo1.cc', 'foo2.cc', 'foo3.cc', 'foo4.cc', 'foo5.cc'"
            + ", 'foo6.cc'])",
        "genrule(name = 'foo', outs=['_objs/x/conflict/foo1.pic.o'], srcs=['foo1.cc', 'foo2.cc', "
            + "'foo3.cc', 'foo4.cc', 'foo5.cc', 'foo6.cc'], cmd='', output_to_bindir=1)");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:x", "//conflict:foo");

    Event event =
        assertContainsEvent("file 'conflict/_objs/x/conflict/foo1.pic.o' " + CONFLICT_MSG);
    assertContainsEvent("MandatoryInputs");
    assertContainsEvent("Outputs");

    // Validate that maximum of 5 artifacts in MandatoryInputs are part of output.
    Pattern pattern = Pattern.compile("\tconflict\\/foo[1-6].cc");
    Matcher matcher = pattern.matcher(event.getMessage());
    int matchCount = 0;
    while (matcher.find()) {
      matchCount++;
    }

    assertEquals(5, matchCount);
  }

  /**
   * The current action conflict detection code will only mark one of the targets as having an
   * error, and with multi-threaded analysis it is not deterministic which one that will be.
   */
  @Test
  public void testActionConflictMarksTargetInvalid() throws Exception {
    scratch.file("conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/conflict/foo.pic.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING),
        "//conflict:x", "//conflict:_objs/x/conflict/foo.pic.o");
    ConfiguredTarget a = getConfiguredTarget("//conflict:x");
    ConfiguredTarget b = getConfiguredTarget("//conflict:_objs/x/conflict/foo.pic.o");
    assertTrue(hasTopLevelAnalysisError(a) ^ hasTopLevelAnalysisError(b));
  }

  /**
   *  BUILD file involved in BUILD-file cycle is changed
   */
  @Test
  public void testBuildFileInCycleChanged() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'],",
        "          deps = ['//java/b'])");
    scratch.file("java/b/BUILD",
        "java_library(name = 'b',",
        "          srcs = ['B.java'],",
        "          deps = ['//java/c'])");
    scratch.file("java/c/BUILD",
        "java_library(name = 'c',",
        "          srcs = ['C.java'],",
        "          deps = ['//java/b'])");
    // expect error
    reporter.removeHandler(failFastHandler);
    update(defaultFlags().with(Flag.KEEP_GOING), "//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    // drop dependency on from b to c
    scratch.overwriteFile("java/b/BUILD",
        "java_library(name = 'b',",
        "             srcs = ['B.java'])");
    eventCollector.clear();
    reporter.addHandler(failFastHandler);
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertNotSame(old, current);
  }

  private void assertNoTargetsVisited() {
    Set<?> analyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertEquals(analyzedTargets.toString(), 0, analyzedTargets.size());
  }

  @Test
  public void testSecondRunAllCacheHits() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])");
    update("//java/a:A");
    update("//java/a:A");
    assertNoTargetsVisited();
  }

  @Test
  public void testDependencyAllCacheHits() throws Exception {
    scratch.file("java/a/BUILD",
        "java_library(name = 'x', srcs = ['A.java'], deps = ['y'])",
        "java_library(name = 'y', srcs = ['B.java'])");
    update("//java/a:x");
    Set<?> oldAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertTrue(oldAnalyzedTargets.size() >= 2);  // could be greater due to implicit deps
    assertEquals(1, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:x"));
    assertEquals(1, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:y"));
    update("//java/a:y");
    assertNoTargetsVisited();
  }

  @Test
  public void testSupersetNotAllCacheHits() throws Exception {
    scratch.file("java/a/BUILD",
        // It's important that all targets are of the same rule class, otherwise the second update
        // call might analyze more than one extra target because of potential implicit dependencies.
        "java_library(name = 'x', srcs = ['A.java'], deps = ['y'])",
        "java_library(name = 'y', srcs = ['B.java'], deps = ['z'])",
        "java_library(name = 'z', srcs = ['C.java'])");
    update("//java/a:y");
    Set<?> oldAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertTrue(oldAnalyzedTargets.size() >= 3);  // could be greater due to implicit deps
    assertEquals(0, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:x"));
    assertEquals(1, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:y"));
    update("//java/a:x");
    Set<?> newAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertTrue(newAnalyzedTargets.size() >= 1);  // could be greater due to implicit deps
    assertEquals(1, countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:x"));
    assertEquals(0, countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:y"));
  }

  @Test
  public void testExtraActions() throws Exception {
    scratch.file("java/com/google/a/BUILD", "java_library(name='a', srcs=['A.java'])");
    scratch.file("java/com/google/b/BUILD", "java_library(name='b', srcs=['B.java'])");
    scratch.file("extra/BUILD",
        "extra_action(name = 'extra',",
        "             out_templates = ['$(OWNER_LABEL_DIGEST)_$(ACTION_ID).tst'],",
        "             cmd = '')",
        "action_listener(name = 'listener',",
        "                mnemonics = ['Javac'],",
        "                extra_actions = [':extra'])");

    useConfiguration("--experimental_action_listener=//extra:listener");
    update("//java/com/google/a:a");
    update("//java/com/google/b:b");
  }

  @Test
  public void testExtraActionsCaching() throws Exception {
    scratch.file("java/a/BUILD", "java_library(name='a', srcs=['A.java'])");
    scratch.file("extra/BUILD",
        "extra_action(name = 'extra',",
        "             out_templates = ['$(OWNER_LABEL_DIGEST)_$(ACTION_ID).tst'],",
        "             cmd = 'echo $(EXTRA_ACTION_FILE)')",
        "action_listener(name = 'listener',",
        "                mnemonics = ['Javac'],",
        "                extra_actions = [':extra'])");
    useConfiguration("--experimental_action_listener=//extra:listener");

    update("//java/a:a");
    getConfiguredTarget("//java/a:a");

    scratch.overwriteFile("extra/BUILD",
        "extra_action(name = 'extra',",
        "             out_templates = ['$(OWNER_LABEL_DIGEST)_$(ACTION_ID).tst'],",
        "             cmd = 'echo $(BUG)')", // <-- change here
        "action_listener(name = 'listener',",
        "                mnemonics = ['Javac'],",
        "                extra_actions = [':extra'])");
    reporter.removeHandler(failFastHandler);
    try {
      update("//java/a:a");
      fail();
    } catch (ViewCreationFailedException e) {
      assertThat(e.getMessage()).contains("Analysis of target '//java/a:a' failed");
      assertContainsEvent("Unable to expand make variables: $(BUG)");
    }
  }

  @Test
  public void testConfigurationCachingWithWarningReplay() throws Exception {
    useConfiguration("--test_sharding_strategy=experimental_heuristic");
    update();
    assertContainsEvent("Heuristic sharding is intended as a one-off experimentation tool");
    eventCollector.clear();
    update();
    assertContainsEvent("Heuristic sharding is intended as a one-off experimentation tool");
  }
  
  @Test
  public void testWorkspaceStatusCommandIsNotCachedForNullBuild() throws Exception {
    update();
    WorkspaceStatusAction actionA = getView().getLastWorkspaceBuildInfoActionForTesting();
    assertEquals("DummyBuildInfoAction", actionA.getMnemonic());

    workspaceStatusActionFactory.setKey("Second");
    update();
    WorkspaceStatusAction actionB = getView().getLastWorkspaceBuildInfoActionForTesting();
    assertEquals("DummyBuildInfoActionSecond", actionB.getMnemonic());
  }

  @Test
  public void testSkyframeCacheInvalidationBuildFileChange() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])");
    String aTarget = "//java/a:A";
    update(aTarget);
    ConfiguredTarget firstCT = getConfiguredTarget(aTarget);

    scratch.overwriteFile("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['B.java'])");

    update(aTarget);
    ConfiguredTarget updatedCT = getConfiguredTarget(aTarget);
    assertNotSame(firstCT, updatedCT);

    update(aTarget);
    ConfiguredTarget updated2CT = getConfiguredTarget(aTarget);
    assertSame(updatedCT, updated2CT);
  }

  @Test
  public void testSkyframeDifferentPackagesInvalidation() throws Exception {
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])");

    scratch.file("java/b/BUILD",
        "java_test(name = 'B',",
        "          srcs = ['B.java'])");

    String aTarget = "//java/a:A";
    update(aTarget);
    ConfiguredTarget oldAConfTarget = getConfiguredTarget(aTarget);
    String bTarget = "//java/b:B";
    update(bTarget);
    ConfiguredTarget oldBConfTarget = getConfiguredTarget(bTarget);

    scratch.overwriteFile("java/b/BUILD",
        "java_test(name = 'B',",
        "          srcs = ['C.java'])");

    update(aTarget);
    // Check that 'A' was not invalidated because 'B' was modified and invalidated.
    ConfiguredTarget newAConfTarget = getConfiguredTarget(aTarget);
    ConfiguredTarget newBConfTarget = getConfiguredTarget(bTarget);

    assertSame(oldAConfTarget, newAConfTarget);
    assertNotSame(oldBConfTarget, newBConfTarget);
  }

  private int countObjectsPartiallyMatchingRegex(Iterable<? extends Object> elements,
      String toStringMatching) {
    toStringMatching = ".*" + toStringMatching + ".*";
    int result = 0;
    for (Object o : elements) {
      if (o.toString().matches(toStringMatching)) {
        ++result;
      }
    }
    return result;
  }

  @Test
  public void testGetSkyframeEvaluatedTargetKeysOmitsCachedTargets() throws Exception {
    scratch.file("java/a/BUILD",
        "java_library(name = 'x', srcs = ['A.java'], deps = ['z', 'w'])",
        "java_library(name = 'y', srcs = ['B.java'], deps = ['z', 'w'])",
        "java_library(name = 'z', srcs = ['C.java'])",
        "java_library(name = 'w', srcs = ['D.java'])");

    update("//java/a:x");
    Set<?> oldAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertTrue(oldAnalyzedTargets.size() >= 2);  // could be greater due to implicit deps
    assertEquals(1, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:x"));
    assertEquals(0, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:y"));
    assertEquals(1, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:z"));
    assertEquals(1, countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:w"));

    // Unless the build is not fully cached, we get notified about newly evaluated targets, as well
    // as cached top-level targets. For the two tests above to work correctly, we need to ensure
    // that getSkyframeEvaluatedTargetKeys() doesn't return these.
    update("//java/a:x", "//java/a:y", "//java/a:z");
    Set<?> newAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertThat(newAnalyzedTargets).hasSize(2);
    assertEquals(1, countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:B.java"));
    assertEquals(1, countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:y"));
  }
}
