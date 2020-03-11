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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.DefaultBuildOptionsForTesting;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
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
  private static final String ADDITIONAL_BLACKLISTED_PACKAGE_PREFIXES_FILE_PATH_STRING =
      "config/blacklist.txt";

  private static SkyKey getKeyForLabel(Label label) {
    // Note that these tests used to look for TargetMarker SkyKeys before TargetMarker was
    // inlined in TransitiveTraversalFunction. Because TargetMarker is now inlined, it doesn't
    // appear in the graph. Instead, these tests now look for TransitiveTraversal keys.
    return TransitiveTraversalValue.key(label);
  }

  @Before
  public void setUp() throws Exception {
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(
                getScratch().dir("/install"),
                getScratch().dir("/output"),
                getScratch().dir("/user_root")),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            AnalysisMock.get().getProductName());
    ConfiguredRuleClassProvider ruleClassProvider = AnalysisMock.get().createRuleClassProvider();

    PackageFactory pkgFactory =
        AnalysisMock.get()
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);
    skyframeExecutor =
        BazelSkyframeExecutorConstants.newBazelSkyframeExecutorBuilder()
            .setPkgFactory(pkgFactory)
            .setFileSystem(fileSystem)
            .setDirectories(directories)
            .setActionKeyContext(new ActionKeyContext())
            .setDefaultBuildOptions(
                DefaultBuildOptionsForTesting.getDefaultBuildOptionsForTest(ruleClassProvider))
            .setExtraSkyFunctions(AnalysisMock.get().getSkyFunctions(directories))
            .setBlacklistedPackagePrefixesFunction(
                new BlacklistedPackagePrefixesFunction(
                    PathFragment.create(ADDITIONAL_BLACKLISTED_PACKAGE_PREFIXES_FILE_PATH_STRING)))
            .build();
    TestConstants.processSkyframeExecutorForTesting(skyframeExecutor);
    skyframeExecutor.preparePackageLoading(
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(Root.fromPath(rootDirectory)),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
        Options.getDefaults(PackageCacheOptions.class),
        Options.getDefaults(StarlarkSemanticsOptions.class),
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        new TimestampGranularityMonitor(null));
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE,
                Optional.<RootedPath>absent()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.REPOSITORY_OVERRIDES,
                ImmutableMap.<RepositoryName, PathFragment>of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING,
                RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY)));
    scratch.file(ADDITIONAL_BLACKLISTED_PACKAGE_PREFIXES_FILE_PATH_STRING);
  }

  @Test
  public void testRecursiveEvaluationFailsOnBadBuildFile() throws Exception {
    // Given a well-formed package "@//foo" and a malformed package "@//foo/foo",
    createFooAndFooFoo();

    // Given a target pattern sequence consisting of a recursive pattern for "//foo/...",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo/...");

    // When PrepareDepsOfPatternsFunction completes evaluation (with no error because it was
    // recovered from),
    WalkableGraph walkableGraph =
        getGraphFromPatternsEvaluation(
            patternSequence, /*successExpected=*/ true, /*keepGoing=*/ true);

    // Then the graph contains package values for "@//foo" and "@//foo/foo",
    assertThat(exists(PackageValue.key(PackageIdentifier.parse("@//foo")), walkableGraph)).isTrue();
    assertThat(exists(PackageValue.key(PackageIdentifier.parse("@//foo/foo")), walkableGraph))
        .isTrue();

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
  public void testBlacklistPatternBlocksPatternEvaluation() throws Exception {
    // Given a well-formed package "//foo" and a malformed package "//foo/foo",
    createFooAndFooFoo();

    // Given a target pattern sequence consisting of a recursive pattern for "//foo/...",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo/...");

    // and a blacklist for the malformed package,
    scratch.overwriteFile(ADDITIONAL_BLACKLISTED_PACKAGE_PREFIXES_FILE_PATH_STRING, "foo/foo");

    assertSkipsFoo(patternSequence);
  }

  private void assertSkipsFoo(ImmutableList<String> patternSequence) throws Exception {


    // When PrepareDepsOfPatternsFunction completes evaluation (successfully),
    WalkableGraph walkableGraph =
        getGraphFromPatternsEvaluation(
            patternSequence, /*successExpected=*/ true, /*keepGoing=*/ true);

    // Then the graph contains a package value for "@//foo",
    assertThat(exists(PackageValue.key(PackageIdentifier.parse("@//foo")), walkableGraph)).isTrue();

    // But no package value for "@//foo/foo",
    assertThat(exists(PackageValue.key(PackageIdentifier.parse("@//foo/foo")), walkableGraph))
        .isFalse();

    // And the graph does not contain a value for the target "@//foo/foo:foofoo".
    Label label = Label.create("@//foo/foo", "foofoo");
    assertThat(exists(getKeyForLabel(label), walkableGraph)).isFalse();
  }

  @Test
  public void testNegativeNonTBDPatternsAreSkippedWithWarnings() throws Exception {
    // Given a target pattern sequence with a negative non-TBD pattern,
    ImmutableList<String> patternSequence = ImmutableList.of("-//foo/bar");

    // When PrepareDepsOfPatternsFunction completes evaluation,
    getGraphFromPatternsEvaluation(patternSequence, /*successExpected=*/ true, /*keepGoing=*/ true);

    // Then a event is published that says that negative non-TBD patterns are skipped.
    assertContainsEvent(
        "Skipping '-//foo/bar, excludedSubdirs=[], filteringPolicy=[]': Negative target patterns of"
            + " types other than \"targets below directory\" are not permitted.");
  }

  // Helpers:

  private WalkableGraph getGraphFromPatternsEvaluation(
      ImmutableList<String> patternSequence, boolean successExpected, boolean keepGoing)
      throws InterruptedException {
    SkyKey independentTarget = PrepareDepsOfPatternsValue.key(patternSequence, "");
    ImmutableList<SkyKey> singletonTargetPattern = ImmutableList.of(independentTarget);

    // When PrepareDepsOfPatternsFunction completes evaluation,
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setNumThreads(100)
            .setEventHander(new Reporter(new EventBus(), eventCollector))
            .build();
    EvaluationResult<SkyValue> evaluationResult =
        skyframeExecutor.getDriver().evaluate(singletonTargetPattern, evaluationContext);
    // The evaluation has no errors if success was expected.
    assertThat(evaluationResult.hasError()).isNotEqualTo(successExpected);
    return Preconditions.checkNotNull(evaluationResult.getWalkableGraph());
  }

  private void createFooAndFooFoo() throws IOException {
    scratch.file(
        "foo/BUILD", "genrule(name = 'foo',", "    outs = ['out.txt'],", "    cmd = 'touch $@')");
    scratch.file(
        "foo/foo/BUILD", "genrule(name = 'foofoo',", "    This isn't even remotely grammatical.)");
  }
}
