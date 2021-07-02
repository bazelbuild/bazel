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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BzlCompileFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.BzlmodRepoRuleFunction;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.IgnoredPackagePrefixesFunction;
import com.google.devtools.build.lib.skyframe.LocalRepositoryLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BzlmodRepoRuleFunction}. */
@RunWith(JUnit4.class)
public final class BzlmodRepoRuleFunctionTest extends FoundationTestCase {

  private SequentialBuildDriver driver;
  private final RecordingDifferencer differencer = new SequencedRecordingDifferencer();
  private EvaluationContext evaluationContext;

  @Before
  public void setup() throws Exception {
    evaluationContext =
        EvaluationContext.newBuilder().setNumThreads(8).setEventHandler(reporter).build();
    AtomicReference<PathPackageLocator> packageLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, rootDirectory),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            AnalysisMock.get().getProductName());
    ExternalFilesHelper externalFilesHelper =
        ExternalFilesHelper.createForTesting(
            packageLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);

    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder
        .clearWorkspaceFileSuffixForTesting()
        .addStarlarkBootstrap(new RepositoryBootstrap(new StarlarkRepositoryModule()));
    ConfiguredRuleClassProvider ruleClassProvider = builder.build();

    PackageFactory pkgFactory =
        AnalysisMock.get()
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);

    HashFunction hashFunction = fileSystem.getDigestFunction().getHashFunction();
    MemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(FileValue.FILE, new FileFunction(packageLocator))
                .put(
                    FileStateValue.FILE_STATE,
                    new FileStateFunction(
                        new AtomicReference<TimestampGranularityMonitor>(),
                        new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS),
                        externalFilesHelper))
                .put(
                    BzlmodRepoRuleValue.BZLMOD_REPO_RULE,
                    new BzlmodRepoRuleFunction(
                        pkgFactory, ruleClassProvider, directories, getFakeBzlmodRepoRuleHelper()))
                .put(
                    SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
                    new LocalRepositoryLookupFunction(
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(
                    SkyFunctions.PACKAGE,
                    new PackageFunction(
                        /* packageFactory= */ null,
                        /* pkgLocator= */ null,
                        /* showLoadingProgress= */ null,
                        /* packageFunctionCache= */ null,
                        /* compiledBuildFileCache= */ null,
                        /* numPackagesLoaded= */ null,
                        /* bzlLoadFunctionForInlining= */ null,
                        /* packageProgress= */ null,
                        PackageFunction.ActionOnIOExceptionReadingBuildFile.UseOriginalIOException
                            .INSTANCE,
                        PackageFunction.IncrementalityIntent.INCREMENTAL))
                .put(
                    SkyFunctions.PACKAGE_LOOKUP,
                    new PackageLookupFunction(
                        new AtomicReference<>(ImmutableSet.of()),
                        CrossRepositoryLabelViolationStrategy.ERROR,
                        BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .put(SkyFunctions.BZL_COMPILE, new BzlCompileFunction(pkgFactory, hashFunction))
                .put(
                    SkyFunctions.BZL_LOAD,
                    BzlLoadFunction.create(
                        pkgFactory, directories, hashFunction, Caffeine.newBuilder().build()))
                .put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction())
                .put(
                    SkyFunctions.IGNORED_PACKAGE_PREFIXES,
                    new IgnoredPackagePrefixesFunction(
                        /*ignoredPackagePrefixesFile=*/ PathFragment.EMPTY_FRAGMENT))
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, packageLocator.get());
  }

  private FakeBzlmodRepoRuleHelper getFakeBzlmodRepoRuleHelper() {
    ImmutableMap.Builder<String, RepoSpec> repoSpecs = ImmutableMap.builder();
    repoSpecs
        // repos from non-registry overrides
        .put(
        "A",
        RepoSpec.builder()
            .setRuleClassName("local_repository")
            .setAttributes(
                ImmutableMap.of(
                    "name", "A",
                    "path", "/foo/bar/A"))
            .build());
    return new FakeBzlmodRepoRuleHelper(repoSpecs.build());
  }

  @Test
  public void createRepoRule_bazelTools() throws Exception {
    EvaluationResult<BzlmodRepoRuleValue> result =
        driver.evaluate(
            ImmutableList.of(BzlmodRepoRuleValue.key("bazel_tools")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    BzlmodRepoRuleValue bzlmodRepoRuleValue = result.get(BzlmodRepoRuleValue.key("bazel_tools"));
    Rule repoRule = bzlmodRepoRuleValue.getRule();

    assertThat(repoRule.getRuleClass()).isEqualTo("local_repository");
    assertThat(repoRule.getName()).isEqualTo("bazel_tools");
    // In the test, the install base is set to rootDirectory, which is "/workspace".
    assertThat(repoRule.getAttr("path", Type.STRING)).isEqualTo("/workspace/embedded_tools");
  }

  @Test
  public void createRepoRule_overrides() throws Exception {
    EvaluationResult<BzlmodRepoRuleValue> result =
        driver.evaluate(ImmutableList.of(BzlmodRepoRuleValue.key("A")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    BzlmodRepoRuleValue bzlmodRepoRuleValue = result.get(BzlmodRepoRuleValue.key("A"));
    Rule repoRule = bzlmodRepoRuleValue.getRule();

    assertThat(repoRule.getRuleClassObject().isStarlark()).isFalse();
    assertThat(repoRule.getRuleClass()).isEqualTo("local_repository");
    assertThat(repoRule.getName()).isEqualTo("A");
    assertThat(repoRule.getAttr("path", Type.STRING)).isEqualTo("/foo/bar/A");
  }

  @Test
  public void repoRule_notFound() throws Exception {
    EvaluationResult<BzlmodRepoRuleValue> result =
        driver.evaluate(ImmutableList.of(BzlmodRepoRuleValue.key("unknown")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    BzlmodRepoRuleValue bzlmodRepoRuleValue = result.get(BzlmodRepoRuleValue.key("unknown"));

    assertThat(bzlmodRepoRuleValue).isEqualTo(BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE);
  }
}
