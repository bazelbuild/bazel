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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.WalkableGraphUtils.exists;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.runtime.QuiescingExecutorsImpl;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.SkyframeExecutorTestHelper;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.util.UUID;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PrepareDepsOfPatternsFunction}. */
@RunWith(JUnit4.class)
public class PrepareDepsOfPatternsFunctionSmartNegationTest extends FoundationTestCase {
  private SkyframeExecutor skyframeExecutor;
  private static final String ADDITIONAL_IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING =
      "config/ignored.txt";

  private static SkyKey getKeyForLabel(Label label) {
    // Note that these tests used to look for TargetMarker SkyKeys before TargetMarker was
    // inlined in TransitiveTraversalFunction. Because TargetMarker is now inlined, it doesn't
    // appear in the graph. Instead, these tests now look for TransitiveTraversal keys.
    return TransitiveTraversalValue.key(label);
  }

  @Before
  public void setUp() throws Exception {
    AnalysisMock analysisMock = AnalysisMock.getAnalysisMockWithoutBuiltinModules();
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(
                getScratch().dir("/install"),
                getScratch().dir("/output"),
                getScratch().dir("/user_root")),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());
    ConfiguredRuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();

    PackageFactory pkgFactory =
        analysisMock
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);
    skyframeExecutor =
        BazelSkyframeExecutorConstants.newBazelSkyframeExecutorBuilder()
            .setPkgFactory(pkgFactory)
            .setFileSystem(fileSystem)
            .setDirectories(directories)
            .setActionKeyContext(new ActionKeyContext())
            .setExtraSkyFunctions(analysisMock.getSkyFunctions(directories))
            .setSyscallCache(SyscallCache.NO_CACHE)
            .setIgnoredPackagePrefixesFunction(
                new IgnoredPackagePrefixesFunction(
                    PathFragment.create(ADDITIONAL_IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING)))
            .build();
    SkyframeExecutorTestHelper.process(skyframeExecutor);
    skyframeExecutor.preparePackageLoading(
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(Root.fromPath(rootDirectory)),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
        Options.getDefaults(PackageOptions.class),
        ImmutableSet.of(),
        Options.getDefaults(BuildLanguageOptions.class),
        UUID.randomUUID(),
        ImmutableMap.of(),
        QuiescingExecutorsImpl.forTesting(),
        new TimestampGranularityMonitor(null));
    skyframeExecutor.setActionEnv(ImmutableMap.of());
    skyframeExecutor.injectExtraPrecomputedValues(analysisMock.getPrecomputedValues());
    scratch.file(ADDITIONAL_IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING);
  }

  @Test
  public void testRecursiveEvaluationFailsOnBadBuildFile() throws Exception {
    // Given a well-formed package "@//foo" and a malformed package "@//foo/foo",
    createFooAndFooFoo();

    // Given a target pattern sequence consisting of a recursive pattern for "//foo/...",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo/...");

    // When PrepareDepsOfPatternsFunction completes evaluation (with no error because it was
    // recovered from),
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph contains package values for "@//foo" and "@//foo/foo",
    assertThat(exists(PackageIdentifier.createInMainRepo("foo"), walkableGraph)).isTrue();
    assertThat(exists(PackageIdentifier.createInMainRepo("foo/foo"), walkableGraph)).isTrue();

    // But the graph does not contain a value for the target "@//foo/foo:foofoo".
    assertThat(exists(getKeyForLabel(Label.create("@//foo/foo", "foofoo")), walkableGraph))
        .isFalse();
  }

  @Test
  public void testNegativePatternBlocksPatternEvaluation() throws Exception {
    // Given a well-formed package "//foo" and a malformed package "//foo/foo",
    createFooAndFooFoo();

    // Given a target pattern sequence consisting of a recursive pattern for "//foo/..." followed
    // by a negative pattern for the malformed package,
    ImmutableList<String> patternSequence = ImmutableList.of("//foo/...", "-//foo/foo/...");

    assertSkipsFoo(patternSequence);
  }

  @Test
  public void testIgnoredPatternBlocksPatternEvaluation() throws Exception {
    // Given a well-formed package "//foo" and a malformed package "//foo/foo",
    createFooAndFooFoo();

    // Given a target pattern sequence consisting of a recursive pattern for "//foo/...",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo/...");

    // and an ignored entry for the malformed package,
    scratch.overwriteFile(ADDITIONAL_IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING, "foo/foo");

    assertSkipsFoo(patternSequence);
  }

  private void assertSkipsFoo(ImmutableList<String> patternSequence) throws Exception {

    // When PrepareDepsOfPatternsFunction completes evaluation (successfully),
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph contains a package value for "@//foo",
    assertThat(exists(PackageIdentifier.createInMainRepo("foo"), walkableGraph)).isTrue();

    // But no package value for "@//foo/foo",
    assertThat(exists(PackageIdentifier.createInMainRepo("foo/foo"), walkableGraph)).isFalse();

    // And the graph does not contain a value for the target "@//foo/foo:foofoo".
    Label label = Label.create("@//foo/foo", "foofoo");
    assertThat(exists(getKeyForLabel(label), walkableGraph)).isFalse();
  }

  @Test
  public void testNegativeNonTBDPatternsAreSkippedWithWarnings() throws Exception {
    // Given a target pattern sequence with a negative non-TBD pattern,
    ImmutableList<String> patternSequence = ImmutableList.of("-//foo/bar");

    // When PrepareDepsOfPatternsFunction completes evaluation,
    getGraphFromPatternsEvaluation(patternSequence);

    // Then a event is published that says that negative non-TBD patterns are skipped.
    assertContainsEvent(
        "Skipping '-//foo/bar, excludedSubdirs=[], filteringPolicy=[]': Negative target patterns of"
            + " types other than \"targets below directory\" are not permitted.");
  }

  // Helpers:

  private WalkableGraph getGraphFromPatternsEvaluation(ImmutableList<String> patternSequence)
      throws InterruptedException {
    SkyKey independentTarget =
        PrepareDepsOfPatternsValue.key(patternSequence, PathFragment.EMPTY_FRAGMENT);
    ImmutableList<SkyKey> singletonTargetPattern = ImmutableList.of(independentTarget);

    // When PrepareDepsOfPatternsFunction completes evaluation,
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(true)
            .setParallelism(100)
            .setEventHandler(new Reporter(new EventBus(), eventCollector))
            .build();
    EvaluationResult<SkyValue> evaluationResult =
        skyframeExecutor.getEvaluator().evaluate(singletonTargetPattern, evaluationContext);
    // The evaluation has no errors if success was expected.
    if (evaluationResult.hasError()) {
      fail(evaluationResult.getError().toString());
    }
    return Preconditions.checkNotNull(evaluationResult.getWalkableGraph());
  }

  private void createFooAndFooFoo() throws IOException {
    scratch.file(
        "foo/BUILD",
        """
        genrule(
            name = "foo",
            outs = ["out.txt"],
            cmd = "touch $@",
        )
        """);
    scratch.file(
        "foo/foo/BUILD", "genrule(name = 'foofoo',", "    This isn't even remotely grammatical.)");
  }
}
