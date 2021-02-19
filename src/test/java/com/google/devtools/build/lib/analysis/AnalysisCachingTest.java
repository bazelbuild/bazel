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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.util.AnalysisCachingTestBase;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestConstants.InternalTestExecutionMode;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsParser;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Analysis caching tests. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class AnalysisCachingTest extends AnalysisCachingTestBase {

  @Test
  public void testSimpleCleanAnalysis() throws Exception {
    scratch.file("java/a/BUILD", "java_test(name = 'A',", "          srcs = ['A.java'])");
    update("//java/a:A");
    ConfiguredTarget javaTest = getConfiguredTarget("//java/a:A");
    assertThat(javaTest).isNotNull();
    assertThat(JavaInfo.getProvider(JavaSourceJarsProvider.class, javaTest)).isNotNull();
  }

  @Test
  public void testTickTock() throws Exception {
    scratch.file(
        "java/a/BUILD",
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
    scratch.file("java/a/BUILD", "java_test(name = 'A',", "          srcs = ['A.java'])");
    update("//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertThat(current).isSameInstanceAs(old);
  }

  @Test
  public void testSubsetCached() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'])",
        "java_test(name = 'B',",
        "          srcs = ['B.java'])");
    update("//java/a:A", "//java/a:B");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertThat(current).isSameInstanceAs(old);
  }

  @Test
  public void testDependencyChanged() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'],",
        "          deps = ['//java/b'])");
    scratch.file("java/b/BUILD", "java_library(name = 'b',", "             srcs = ['B.java'])");
    update("//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    scratch.overwriteFile(
        "java/b/BUILD", "java_library(name = 'b',", "             srcs = ['C.java'])");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertThat(current).isNotSameInstanceAs(old);
  }

  @Test
  public void testTopLevelChanged() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'],",
        "          deps = ['//java/b'])");
    scratch.file("java/b/BUILD", "java_library(name = 'b',", "             srcs = ['B.java'])");
    update("//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    scratch.overwriteFile("java/a/BUILD", "java_test(name = 'A',", "          srcs = ['A.java'])");
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertThat(current).isNotSameInstanceAs(old);
  }

  // Regression test for:
  // "action conflict detection is incorrect if conflict is in non-top-level configured targets".
  @Test
  public void testActionConflictInDependencyImpliesTopLevelTargetFailure() throws Exception {
    if (getInternalTestExecutionMode() != InternalTestExecutionMode.NORMAL) {
      // TODO(b/67529176): conflicts not detected.
      return;
    }
    useConfiguration("--cpu=k8");
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])",
        "cc_binary(name='foo', deps=['x'], data=['_objs/x/foo.o'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:foo");
    assertContainsEvent("file 'conflict/_objs/x/foo.o' " + CONFLICT_MSG);
    assertThat(getAnalysisResult().getTargetsToBuild()).isEmpty();
  }

  /**
   * Generating the same output from two targets is ok if we build them on successive builds and
   * invalidate the first target before we build the second target. This is a strictly weaker test
   * than if we didn't invalidate the first target, but since Skyframe can't pass then, this test
   * could be useful for it. Actually, since Skyframe makes multiple update calls, it manages to
   * unregister actions even when it shouldn't, and so this test can incorrectly pass. However,
   * {@code SkyframeExecutorTest#testNoActionConflictWithInvalidatedTarget} tests it more
   * rigorously.
   */
  @Test
  public void testNoActionConflictWithInvalidatedTarget() throws Exception {
    useConfiguration("--cpu=k8");
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    update("//conflict:x");
    ConfiguredTarget conflict = getConfiguredTarget("//conflict:x");
    Action oldAction = getGeneratingAction(getBinArtifact("_objs/x/foo.o", conflict));
    assertThat(oldAction.getOwner().getLabel().toString()).isEqualTo("//conflict:x");
    scratch.overwriteFile(
        "conflict/BUILD",
        "cc_library(name='newx', srcs=['foo.cc'])", // Rename target.
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    update(defaultFlags(), "//conflict:_objs/x/foo.o");
    ConfiguredTarget objsConflict = getConfiguredTarget("//conflict:_objs/x/foo.o");
    Action newAction = getGeneratingAction(getBinArtifact("_objs/x/foo.o", objsConflict));
    assertThat(newAction.getOwner().getLabel().toString()).isEqualTo("//conflict:_objs/x/foo.o");
  }

  /** Generating the same output from multiple actions is causing an error. */
  @Test
  public void testActionConflictCausesError() throws Exception {
    if (getInternalTestExecutionMode() != InternalTestExecutionMode.NORMAL) {
      // TODO(b/67529176): conflicts not detected.
      return;
    }
    useConfiguration("--cpu=k8");
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:x", "//conflict:_objs/x/foo.o");
    assertContainsEvent("file 'conflict/_objs/x/foo.o' " + CONFLICT_MSG);
  }

  @Test
  public void testNoActionConflictErrorAfterClearedAnalysis() throws Exception {
    if (getInternalTestExecutionMode() != InternalTestExecutionMode.NORMAL) {
      // TODO(b/67529176): conflicts not detected.
      return;
    }
    useConfiguration("--cpu=k8");
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:x", "//conflict:_objs/x/foo.o");
    // We want to force a "dropConfiguredTargetsNow" operation, which won't inform the
    // invalidation receiver about the dropped configured targets.
    skyframeExecutor.clearAnalysisCache(ImmutableList.of(), ImmutableSet.of());
    assertContainsEvent("file 'conflict/_objs/x/foo.o' " + CONFLICT_MSG);
    eventCollector.clear();
    scratch.overwriteFile(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['baz.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:x", "//conflict:_objs/x/foo.o");
    assertNoEvents();
  }

  /**
   * For two conflicting actions whose primary inputs are different, no list diff detail should be
   * part of the output.
   */
  @Test
  public void testConflictingArtifactsErrorWithNoListDetail() throws Exception {
    if (getInternalTestExecutionMode() != InternalTestExecutionMode.NORMAL) {
      // TODO(b/67529176): conflicts not detected.
      return;
    }
    useConfiguration("--cpu=k8");
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:x", "//conflict:_objs/x/foo.o");

    assertContainsEvent("file 'conflict/_objs/x/foo.o' " + CONFLICT_MSG);
    assertDoesNotContainEvent("MandatoryInputs");
    assertDoesNotContainEvent("Outputs");
  }

  /**
   * For two conflicted actions whose primary inputs are the same, list diff (max 5) should be part
   * of the output.
   */
  @Test
  public void testConflictingArtifactsWithListDetail() throws Exception {
    if (getInternalTestExecutionMode() != InternalTestExecutionMode.NORMAL) {
      // TODO(b/67529176): conflicts not detected.
      return;
    }
    useConfiguration("--cpu=k8");
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo1.cc'])",
        "genrule(name = 'foo', outs=['_objs/x/foo1.o'], srcs=['foo1.cc', 'foo2.cc', "
            + "'foo3.cc', 'foo4.cc', 'foo5.cc', 'foo6.cc'], cmd='', output_to_bindir=1)");
    reporter.removeHandler(failFastHandler); // expect errors
    update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:x", "//conflict:foo");

    Event event = assertContainsEvent("file 'conflict/_objs/x/foo1.o' " + CONFLICT_MSG);
    assertContainsEvent("MandatoryInputs");
    assertContainsEvent("Outputs");

    // Validate that maximum of 5 artifacts in MandatoryInputs are part of output.
    Pattern pattern = Pattern.compile("\tconflict\\/foo[2-6].cc");
    Matcher matcher = pattern.matcher(event.getMessage());
    int matchCount = 0;
    while (matcher.find()) {
      matchCount++;
    }

    assertWithMessage(
            "Event does not contain expected number of file conflicts:\n" + event.getMessage())
        .that(matchCount)
        .isEqualTo(5);
  }

  /**
   * The current action conflict detection code will only mark one of the targets as having an
   * error, and with multi-threaded analysis it is not deterministic which one that will be.
   */
  @Test
  public void testActionConflictMarksTargetInvalid() throws Exception {
    if (getInternalTestExecutionMode() != InternalTestExecutionMode.NORMAL) {
      // TODO(b/67529176): conflicts not detected.
      return;
    }
    useConfiguration("--cpu=k8");
    scratch.file(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    reporter.removeHandler(failFastHandler); // expect errors
    int successfulAnalyses =
        update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:x", "//conflict:_objs/x/foo.pic.o")
            .getTargetsToBuild()
            .size();
    assertThat(successfulAnalyses).isEqualTo(1);
  }

  /** BUILD file involved in BUILD-file cycle is changed */
  @Test
  public void testBuildFileInCycleChanged() throws Exception {
    if (getInternalTestExecutionMode() != InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): cycles not properly handled.
      return;
    }
    scratch.file(
        "java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = ['A.java'],",
        "          deps = ['//java/b'])");
    scratch.file(
        "java/b/BUILD",
        "java_library(name = 'b',",
        "          srcs = ['B.java'],",
        "          deps = ['//java/c'])");
    scratch.file(
        "java/c/BUILD",
        "java_library(name = 'c',",
        "          srcs = ['C.java'],",
        "          deps = ['//java/b'])");
    // expect error
    reporter.removeHandler(failFastHandler);
    update(defaultFlags().with(Flag.KEEP_GOING), "//java/a:A");
    ConfiguredTarget old = getConfiguredTarget("//java/a:A");
    // drop dependency on from b to c
    scratch.overwriteFile(
        "java/b/BUILD", "java_library(name = 'b',", "             srcs = ['B.java'])");
    eventCollector.clear();
    reporter.addHandler(failFastHandler);
    update("//java/a:A");
    ConfiguredTarget current = getConfiguredTarget("//java/a:A");
    assertThat(current).isNotSameInstanceAs(old);
  }

  private void assertNoTargetsVisited() {
    Set<?> analyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertWithMessage(analyzedTargets.toString()).that(analyzedTargets.size()).isEqualTo(0);
  }

  @Test
  public void testSecondRunAllCacheHits() throws Exception {
    scratch.file("java/a/BUILD", "java_test(name = 'A',", "          srcs = ['A.java'])");
    update("//java/a:A");
    update("//java/a:A");
    assertNoTargetsVisited();
  }

  @Test
  public void testDependencyAllCacheHits() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "java_library(name = 'x', srcs = ['A.java'], deps = ['y'])",
        "java_library(name = 'y', srcs = ['B.java'])");
    update("//java/a:x");
    Set<?> oldAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertThat(oldAnalyzedTargets.size()).isAtLeast(2); // could be greater due to implicit deps
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:x")).isEqualTo(1);
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:y")).isEqualTo(1);
    update("//java/a:y");
    assertNoTargetsVisited();
  }

  @Test
  public void testSupersetNotAllCacheHits() throws Exception {
    scratch.file(
        "java/a/BUILD",
        // It's important that all targets are of the same rule class, otherwise the second update
        // call might analyze more than one extra target because of potential implicit dependencies.
        "java_library(name = 'x', srcs = ['A.java'], deps = ['y'])",
        "java_library(name = 'y', srcs = ['B.java'], deps = ['z'])",
        "java_library(name = 'z', srcs = ['C.java'])");
    update("//java/a:y");
    Set<?> oldAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertThat(oldAnalyzedTargets.size()).isAtLeast(3); // could be greater due to implicit deps
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:x")).isEqualTo(0);
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:y")).isEqualTo(1);
    update("//java/a:x");
    Set<?> newAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    // Source target and rule target.
    assertThat(newAnalyzedTargets).hasSize(2);
    assertThat(countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:x")).isEqualTo(1);
    assertThat(countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:y")).isEqualTo(0);
  }

  @Test
  public void testExtraActions() throws Exception {
    scratch.file("java/com/google/a/BUILD", "java_library(name='a', srcs=['A.java'])");
    scratch.file("java/com/google/b/BUILD", "java_library(name='b', srcs=['B.java'])");
    scratch.file(
        "extra/BUILD",
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
    scratch.file(
        "extra/BUILD",
        "extra_action(name = 'extra',",
        "             out_templates = ['$(OWNER_LABEL_DIGEST)_$(ACTION_ID).tst'],",
        "             cmd = 'echo $(EXTRA_ACTION_FILE)')",
        "action_listener(name = 'listener',",
        "                mnemonics = ['Javac'],",
        "                extra_actions = [':extra'])");
    useConfiguration("--experimental_action_listener=//extra:listener");

    update("//java/a:a");
    getConfiguredTarget("//java/a:a");

    scratch.overwriteFile(
        "extra/BUILD",
        "extra_action(name = 'extra',",
        "             out_templates = ['$(OWNER_LABEL_DIGEST)_$(ACTION_ID).tst'],",
        "             cmd = 'echo $(BUG)')", // <-- change here
        "action_listener(name = 'listener',",
        "                mnemonics = ['Javac'],",
        "                extra_actions = [':extra'])");
    reporter.removeHandler(failFastHandler);
    ViewCreationFailedException e =
        assertThrows(ViewCreationFailedException.class, () -> update("//java/a:a"));
    assertThat(e).hasMessageThat().contains("Analysis of target '//java/a:a' failed");
    assertContainsEvent("$(BUG) not defined");
  }

  @Test
  public void testConfigurationCachingWithWarningReplay() throws Exception {
    useConfiguration("--strip=always", "--copt=-g");
    update();
    assertContainsEvent("Debug information will be generated and then stripped away");
    eventCollector.clear();
    update();
    assertContainsEvent("Debug information will be generated and then stripped away");
  }

  @Test
  public void testSkyframeCacheInvalidationBuildFileChange() throws Exception {
    scratch.file("java/a/BUILD", "java_test(name = 'A',", "          srcs = ['A.java'])");
    String aTarget = "//java/a:A";
    update(aTarget);
    ConfiguredTarget firstCT = getConfiguredTarget(aTarget);

    scratch.overwriteFile("java/a/BUILD", "java_test(name = 'A',", "          srcs = ['B.java'])");

    update(aTarget);
    ConfiguredTarget updatedCT = getConfiguredTarget(aTarget);
    assertThat(updatedCT).isNotSameInstanceAs(firstCT);

    update(aTarget);
    ConfiguredTarget updated2CT = getConfiguredTarget(aTarget);
    assertThat(updated2CT).isSameInstanceAs(updatedCT);
  }

  @Test
  public void testSkyframeDifferentPackagesInvalidation() throws Exception {
    scratch.file("java/a/BUILD", "java_test(name = 'A',", "          srcs = ['A.java'])");

    scratch.file("java/b/BUILD", "java_test(name = 'B',", "          srcs = ['B.java'])");

    String aTarget = "//java/a:A";
    update(aTarget);
    ConfiguredTarget oldAConfTarget = getConfiguredTarget(aTarget);
    String bTarget = "//java/b:B";
    update(bTarget);
    ConfiguredTarget oldBConfTarget = getConfiguredTarget(bTarget);

    scratch.overwriteFile("java/b/BUILD", "java_test(name = 'B',", "          srcs = ['C.java'])");

    update(aTarget);
    // Check that 'A' was not invalidated because 'B' was modified and invalidated.
    ConfiguredTarget newAConfTarget = getConfiguredTarget(aTarget);
    ConfiguredTarget newBConfTarget = getConfiguredTarget(bTarget);

    assertThat(newAConfTarget).isSameInstanceAs(oldAConfTarget);
    assertThat(newBConfTarget).isNotSameInstanceAs(oldBConfTarget);
  }

  private int countObjectsPartiallyMatchingRegex(
      Iterable<? extends Object> elements, String toStringMatching) {
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
    scratch.file(
        "java/a/BUILD",
        "java_library(name = 'x', srcs = ['A.java'], deps = ['z', 'w'])",
        "java_library(name = 'y', srcs = ['B.java'], deps = ['z', 'w'])",
        "java_library(name = 'z', srcs = ['C.java'])",
        "java_library(name = 'w', srcs = ['D.java'])");

    update("//java/a:x");
    Set<?> oldAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertThat(oldAnalyzedTargets.size()).isAtLeast(2); // could be greater due to implicit deps
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:x")).isEqualTo(1);
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:y")).isEqualTo(0);
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:z")).isEqualTo(1);
    assertThat(countObjectsPartiallyMatchingRegex(oldAnalyzedTargets, "//java/a:w")).isEqualTo(1);

    // Unless the build is not fully cached, we get notified about newly evaluated targets, as well
    // as cached top-level targets. For the two tests above to work correctly, we need to ensure
    // that getSkyframeEvaluatedTargetKeys() doesn't return these.
    update("//java/a:x", "//java/a:y", "//java/a:z");
    Set<?> newAnalyzedTargets = getSkyframeEvaluatedTargetKeys();
    assertThat(newAnalyzedTargets).hasSize(2);
    assertThat(countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:B.java"))
        .isEqualTo(1);
    assertThat(countObjectsPartiallyMatchingRegex(newAnalyzedTargets, "//java/a:y")).isEqualTo(1);
  }

  /** Test options class for testing diff-based analysis cache resetting. */
  public static final class DiffResetOptions extends FragmentOptions {
    public static final OptionDefinition PROBABLY_IRRELEVANT_OPTION =
        OptionsParser.getOptionDefinitionByName(DiffResetOptions.class, "probably_irrelevant");
    public static final OptionDefinition ALSO_IRRELEVANT_OPTION =
        OptionsParser.getOptionDefinitionByName(DiffResetOptions.class, "also_irrelevant");
    public static final PatchTransition CLEAR_IRRELEVANT =
        new PatchTransition() {
          @Override
          public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
            return ImmutableSet.of(DiffResetOptions.class);
          }

          @Override
          public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
            BuildOptionsView cloned = options.clone();
            cloned.get(DiffResetOptions.class).probablyIrrelevantOption = "(cleared)";
            cloned.get(DiffResetOptions.class).alsoIrrelevantOption = "(cleared)";
            return cloned.underlying();
          }
        };

    @Option(
        name = "probably_irrelevant",
        defaultValue = "(unset)",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "This option is irrelevant to non-uses_irrelevant targets and is trimmed from them.")
    public String probablyIrrelevantOption;

    @Option(
        name = "also_irrelevant",
        defaultValue = "(unset)",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "This option is irrelevant to non-uses_irrelevant targets and is trimmed from them.")
    public String alsoIrrelevantOption;

    @Option(
        name = "definitely_relevant",
        defaultValue = "(unset)",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "This option is not trimmed and is used by all targets.")
    public String definitelyRelevantOption;

    @Option(
        name = "also_relevant",
        defaultValue = "(unset)",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "This option is not trimmed and is used by all targets.")
    public String alsoRelevantOption;

    @Option(
        name = "host_relevant",
        defaultValue = "(unset)",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "This option is not trimmed and is used by all host targets.")
    public String hostRelevantOption;

    @Override
    public DiffResetOptions getHost() {
      DiffResetOptions host = ((DiffResetOptions) super.getHost());
      host.definitelyRelevantOption = hostRelevantOption;
      return host;
    }
  }

  /** Test fragment. */
  @StarlarkBuiltin(name = "test_diff_fragment", doc = "fragment for testing differy fragments")
  @RequiresOptions(options = {DiffResetOptions.class})
  public static final class DiffResetFragment extends Fragment implements StarlarkValue {
    public DiffResetFragment(BuildOptions buildOptions) {}
  }

  private void setupDiffResetTesting() throws Exception {
    ImmutableSet<OptionDefinition> optionsThatCanChange =
        ImmutableSet.of(
            DiffResetOptions.PROBABLY_IRRELEVANT_OPTION, DiffResetOptions.ALSO_IRRELEVANT_OPTION);
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DiffResetFragment.class);
    builder.overrideShouldInvalidateCacheForOptionDiffForTesting(
        (newOptions, changedOption, oldValue, newValue) -> {
          return !optionsThatCanChange.contains(changedOption);
        });
    builder.overrideTrimmingTransitionFactoryForTesting(
        (rule) -> {
          if (rule.getRuleClassObject().getName().equals("uses_irrelevant")) {
            return NoTransition.INSTANCE;
          }
          return DiffResetOptions.CLEAR_IRRELEVANT;
        });
    useRuleClassProvider(builder.build());
    scratch.file(
        "test/lib.bzl",
        "def _empty_impl(ctx):",
        "  pass",
        "normal_lib = rule(",
        "    implementation = _empty_impl,",
        "    fragments = ['test_diff_fragment'],",
        "    attrs = {",
        "        'deps': attr.label_list(),",
        "        'host_deps': attr.label_list(cfg='host'),",
        "    },",
        ")",
        "uses_irrelevant = rule(",
        "    implementation = _empty_impl,",
        "    fragments = ['test_diff_fragment'],",
        "    attrs = {",
        "        'deps': attr.label_list(),",
        "        'host_deps': attr.label_list(cfg='host'),",
        "    },",
        ")");
    update();
  }

  @Test
  public void cacheNotClearedWhenOptionsStaySame() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--definitely_relevant=Testing");
    update("//test:top");
    update("//test:top");
    assertNoTargetsVisited();
  }

  @Test
  public void cacheNotClearedWhenOptionsStaySameWithMultiCpu() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--experimental_multi_cpu=k8,ppc", "--definitely_relevant=Testing");
    update("//test:top");
    update("//test:top");
    // these targets were cached and did not need to be reanalyzed
    assertNoTargetsVisited();
  }

  @Test
  public void cacheClearedWhenNonAllowedOptionsChange() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--definitely_relevant=Test 1");
    update("//test:top");
    useConfiguration("--definitely_relevant=Test 2");
    update("//test:top");
    useConfiguration("--definitely_relevant=Test 1");
    update("//test:top");
    // these targets needed to be reanalyzed even though we built them in this configuration
    // just a moment ago
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 1)
            .put("//test:shared", 1)
            .build());
  }

  @Test
  public void cacheClearedWhenNonAllowedHostOptionsChange() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', host_deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--host_relevant=Test 1");
    update("//test:top");
    useConfiguration("--host_relevant=Test 2");
    update("//test:top");
    useConfiguration("--host_relevant=Test 1");
    update("//test:top");
    // these targets needed to be reanalyzed even though we built them in this configuration
    // just a moment ago
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 1)
            .put("//test:shared", 1)
            .build());
  }

  @Test
  public void cacheClearedWhenMultiCpuChanges() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--experimental_multi_cpu=k8,ppc");
    update("//test:top");
    useConfiguration("--experimental_multi_cpu=k8,armeabi-v7a");
    update("//test:top");
    // we needed to reanalyze these in both k8 and armeabi-v7a even though we did the k8 analysis
    // just a moment ago as part of the previous build
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 2)
            .put("//test:shared", 2)
            .build());
  }

  @Test
  public void cacheClearedWhenMultiCpuGetsBigger() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--experimental_multi_cpu=k8,ppc");
    update("//test:top");
    useConfiguration("--experimental_multi_cpu=k8,ppc,armeabi-v7a");
    update("//test:top");
    // we needed to reanalyze these in all of {k8,ppc,armeabi-v7a} even though we did the k8 and ppc
    // analysis just a moment ago as part of the previous build
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 3)
            .put("//test:shared", 3)
            .build());
  }

  @Test
  public void cacheClearedWhenMultiCpuGetsSmaller() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--experimental_multi_cpu=k8,ppc,armeabi-v7a");
    update("//test:top");
    useConfiguration("--experimental_multi_cpu=k8,ppc");
    update("//test:top");
    // we needed to reanalyze these in both k8 and ppc even though we did the k8 and ppc
    // analysis just a moment ago as part of the previous build
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 2)
            .put("//test:shared", 2)
            .build());
  }

  @Test
  public void cacheNotClearedWhenAllowedOptionsChange() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration("--definitely_relevant=Testing", "--probably_irrelevant=Test 1");
    update("//test:top");
    useConfiguration("--definitely_relevant=Testing", "--probably_irrelevant=Test 2");
    update("//test:top");
    // the shared library got to reuse the cached value, while the entry point had to be rebuilt in
    // the new configuration
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 1)
            .put("//test:shared", 0)
            .build());
    useConfiguration("--definitely_relevant=Testing", "--probably_irrelevant=Test 1");
    update("//test:top");
    // now we're back to the old configuration with no cache clears, so no work needed to be done
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 0)
            .put("//test:shared", 0)
            .build());
  }

  @Test
  public void cacheNotClearedWhenAllowedOptionsChangeWithMultiCpu() throws Exception {
    setupDiffResetTesting();
    scratch.file(
        "test/BUILD",
        "load(':lib.bzl', 'normal_lib', 'uses_irrelevant')",
        "uses_irrelevant(name='top', deps=[':shared'])",
        "normal_lib(name='shared')");
    useConfiguration(
        "--experimental_multi_cpu=k8,ppc",
        "--definitely_relevant=Testing",
        "--probably_irrelevant=Test 1");
    update("//test:top");
    useConfiguration(
        "--experimental_multi_cpu=k8,ppc",
        "--definitely_relevant=Testing",
        "--probably_irrelevant=Test 2");
    update("//test:top");
    // the shared library got to reuse the cached value, while the entry point had to be rebuilt in
    // the new configurations
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 2)
            .put("//test:shared", 0)
            .build());
    useConfiguration(
        "--experimental_multi_cpu=k8,ppc",
        "--definitely_relevant=Testing",
        "--probably_irrelevant=Test 1");
    update("//test:top");
    // now we're back to the old configurations with no cache clears, so no work needed to be done
    assertNumberOfAnalyzedConfigurationsOfTargets(
        ImmutableMap.<String, Integer>builder()
            .put("//test:top", 0)
            .put("//test:shared", 0)
            .build());
  }

  @Test
  public void cacheClearedWhenRedundantDefinesChange_collapseDuplicateDefinesDisabled()
      throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--nocollapse_duplicate_defines", "--define=a=1", "--define=a=2");
    update("//test:top");
    useConfiguration("--nocollapse_duplicate_defines", "--define=a=2");
    update("//test:top");
    assertNumberOfAnalyzedConfigurationsOfTargets(ImmutableMap.of("//test:top", 1));
  }

  @Test
  public void cacheNotClearedWhenRedundantDefinesChange() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--collapse_duplicate_defines", "--define=a=1", "--define=a=2");
    update("//test:top");
    useConfiguration("--collapse_duplicate_defines", "--define=a=2");
    update("//test:top");
    assertNumberOfAnalyzedConfigurationsOfTargets(ImmutableMap.of("//test:top", 0));
  }

  @Test
  public void noCacheClearMessageAfterCleanWithSameOptions() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration();
    update("//test:top");
    cleanSkyframe();
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void noCacheClearMessageAfterCleanWithDifferentOptions() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--definitely_relevant=before");
    update("//test:top");
    cleanSkyframe();
    useConfiguration("--definitely_relevant=after");
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void noCacheClearMessageAfterDiscardAnalysisCacheThenCleanWithSameOptions()
      throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--discard_analysis_cache");
    update("//test:top");
    cleanSkyframe();
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void noCacheClearMessageAfterDiscardAnalysisCacheThenCleanWithChangedOptions()
      throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--definitely_relevant=before", "--discard_analysis_cache");
    update("//test:top");
    cleanSkyframe();
    useConfiguration("--definitely_relevant=after", "--discard_analysis_cache");
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void cacheClearMessageAfterDiscardAnalysisCacheBuild() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--probably_irrelevant=yeah",
        "--discard_analysis_cache");
    update("//test:top");
    eventCollector.clear();
    update("//test:top");
    assertContainsEvent("--discard_analysis_cache");
    assertDoesNotContainEvent("Build option");
    assertContainsEvent("discarding analysis cache");
  }

  @Test
  public void noCacheClearMessageAfterNonDiscardAnalysisCacheBuild() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--discard_analysis_cache");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1");
    update("//test:top");
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void noCacheClearMessageAfterIrrelevantOptionChanges() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--probably_irrelevant=old");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--probably_irrelevant=new");
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void noCacheClearMessageAfterIrrelevantOptionChangesWithDiffDisabled() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=0", "--probably_irrelevant=old");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=0", "--probably_irrelevant=new");
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void cacheClearMessageAfterNumberOfConfigurationsIncreases() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,ppc");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,k8,ppc");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --experimental_multi_cpu has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterNumberOfConfigurationsDecreases() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,k8,ppc");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,ppc");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --experimental_multi_cpu has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterChangingExperimentalMultiCpu() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,k8");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,ppc");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --experimental_multi_cpu has changed, discarding analysis cache");
  }

  @Test
  public void noCacheClearMessageAfterOnlyChangingExperimentalMultiCpuOrder() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=k8,armeabi-v7a");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,k8");
    eventCollector.clear();
    update("//test:top");
    assertNoEvents();
  }

  @Test
  public void cacheClearMessageAfterChangingFirstCpuOnMultiCpu() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=k8,piii");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,ppc");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --experimental_multi_cpu has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterChangingCpu() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--cpu=k8");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--cpu=armeabi-v7a");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent("Build option --cpu has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterTurningOnExperimentalMultiCpu() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--cpu=armeabi-v7a");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,k8,ppc");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --experimental_multi_cpu has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterTurningOffExperimentalMultiCpu() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--experimental_multi_cpu=armeabi-v7a,k8,ppc");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--cpu=armeabi-v7a");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --experimental_multi_cpu has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterChangingExperimentalMultiCpuAndOtherRelevantOption()
      throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--experimental_multi_cpu=armeabi-v7a,k8,ppc",
        "--definitely_relevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--experimental_multi_cpu=armeabi-v7a,k8",
        "--definitely_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build options --definitely_relevant and --experimental_multi_cpu have changed, "
            + "discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterChangingExperimentalMultiCpuOrderAndOtherRelevantOption()
      throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--experimental_multi_cpu=k8,armeabi-v7a",
        "--definitely_relevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--experimental_multi_cpu=armeabi-v7a,k8",
        "--definitely_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --definitely_relevant has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterSingleRelevantOptionChanges() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=-1", "--definitely_relevant=old");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=-1", "--definitely_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --definitely_relevant has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageDoesNotIncludeIrrelevantOptions() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--definitely_relevant=old",
        "--probably_irrelevant=old",
        "--also_irrelevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--definitely_relevant=new",
        "--probably_irrelevant=new",
        "--also_irrelevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --definitely_relevant has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageDoesNotIncludeUnchangedOptions() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--definitely_relevant=old", "--also_relevant=fixed");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--definitely_relevant=new", "--also_relevant=fixed");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build option --definitely_relevant has changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterRelevantOptionChangeWithDiffDisabled() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration("--max_config_changes_to_show=0", "--definitely_relevant=old");
    update("//test:top");
    useConfiguration("--max_config_changes_to_show=0", "--definitely_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent("Build options have changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterTwoRelevantOptionsChange() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--definitely_relevant=old", "--also_relevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--definitely_relevant=new", "--also_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build options --also_relevant and --definitely_relevant have changed, "
            + "discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterMultipleRelevantOptionsChange() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--definitely_relevant=old",
        "--also_relevant=old",
        "--host_relevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1",
        "--definitely_relevant=new",
        "--also_relevant=new",
        "--host_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build options --also_relevant, --definitely_relevant, and --host_relevant have changed, "
            + "discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterMultipleRelevantOptionsChangeWithDiffLimit() throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=2",
        "--definitely_relevant=old",
        "--also_relevant=old",
        "--host_relevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=2",
        "--definitely_relevant=new",
        "--also_relevant=new",
        "--host_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build options --also_relevant, --definitely_relevant, and 1 more have changed, "
            + "discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterMultipleRelevantOptionsChangeWithSingleDiffLimit()
      throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=1",
        "--definitely_relevant=old",
        "--also_relevant=old",
        "--host_relevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=1",
        "--definitely_relevant=new",
        "--also_relevant=new",
        "--host_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertDoesNotContainEvent("--discard_analysis_cache");
    assertContainsEvent(
        "Build options --also_relevant and 2 more have changed, discarding analysis cache");
  }

  @Test
  public void cacheClearMessageAfterDiscardAnalysisCacheBuildWithRelevantOptionChanges()
      throws Exception {
    setupDiffResetTesting();
    scratch.file("test/BUILD", "load(':lib.bzl', 'normal_lib')", "normal_lib(name='top')");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--discard_analysis_cache", "--definitely_relevant=old");
    update("//test:top");
    useConfiguration(
        "--max_config_changes_to_show=-1", "--discard_analysis_cache", "--definitely_relevant=new");
    eventCollector.clear();
    update("//test:top");
    assertContainsEvent("--discard_analysis_cache");
    assertDoesNotContainEvent("Build option");
    assertContainsEvent("discarding analysis cache");
  }
}
