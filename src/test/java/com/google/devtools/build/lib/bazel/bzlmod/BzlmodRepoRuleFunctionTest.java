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
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
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
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BzlmodRepoRuleFunction}. */
@RunWith(JUnit4.class)
public final class BzlmodRepoRuleFunctionTest extends FoundationTestCase {

  private MemoizingEvaluator evaluator;
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
    evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(FileValue.FILE, new FileFunction(packageLocator))
                .put(
                    FileStateKey.FILE_STATE,
                    new FileStateFunction(
                        Suppliers.ofInstance(
                            new TimestampGranularityMonitor(BlazeClock.instance())),
                        SyscallCache.NO_CACHE,
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
                        /*packageFactory=*/ null,
                        /*pkgLocator=*/ null,
                        /*showLoadingProgress=*/ null,
                        /*numPackagesLoaded=*/ null,
                        /*bzlLoadFunctionForInlining=*/ null,
                        /*packageProgress=*/ null,
                        PackageFunction.ActionOnIOExceptionReadingBuildFile.UseOriginalIOException
                            .INSTANCE,
                        PackageFunction.IncrementalityIntent.INCREMENTAL,
                        ignored -> ThreadStateReceiver.NULL_INSTANCE))
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
                .put(
                    SkyFunctions.BAZEL_MODULE_RESOLUTION,
                    new SkyFunction() {
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env)
                          throws SkyFunctionException {
                        // Dummy function that returns a dep graph with just the root module in it.
                        return BazelModuleResolutionFunction.createValue(
                            ImmutableMap.of(ModuleKey.ROOT, Module.builder().build()),
                            ImmutableMap.of());
                      }
                    })
                .put(
                    SkyFunctions.REPOSITORY_MAPPING,
                    new SkyFunction() {
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env) {
                        // Dummy function that always falls back.
                        return RepositoryMappingValue.withMapping(
                            RepositoryMapping.ALWAYS_FALLBACK);
                      }
                    })
                .put(
                    SkyFunctions.MODULE_EXTENSION_RESOLUTION,
                    new SkyFunction() {
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env) {
                        // Dummy function that returns nothing.
                        return ModuleExtensionResolutionValue.create(
                            ImmutableMap.of(), ImmutableMap.of(), ImmutableListMultimap.of());
                      }
                    })
                .put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction())
                .put(
                    SkyFunctions.IGNORED_PACKAGE_PREFIXES,
                    new IgnoredPackagePrefixesFunction(
                        /*ignoredPackagePrefixesFile=*/ PathFragment.EMPTY_FRAGMENT))
                .build(),
            differencer);

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, packageLocator.get());

    setupRepoRules();
  }

  private void setupRepoRules() throws Exception {
    scratch.file(rootDirectory.getRelative("tools/build_defs/repo/BUILD").getPathString());
    scratch.file(
        rootDirectory.getRelative("tools/build_defs/repo/http.bzl").getPathString(),
        "def _http_archive_impl(ctx): pass",
        "",
        "http_archive = repository_rule(",
        "    implementation = _http_archive_impl,",
        "    attrs = {",
        "      \"url\": attr.string(),",
        "      \"sha256\": attr.string(),",
        "    })");
    scratch.file(rootDirectory.getRelative("maven/BUILD").getPathString());
    scratch.file(
        rootDirectory.getRelative("maven/repo.bzl").getPathString(),
        "def _maven_repo_impl(ctx): pass",
        "",
        "maven_repo = repository_rule(",
        "    implementation = _maven_repo_impl,",
        "    attrs = {",
        "      \"artifacts\": attr.string_list(),",
        "      \"repositories\": attr.string_list(),",
        "    })");
  }

  private static FakeBzlmodRepoRuleHelper getFakeBzlmodRepoRuleHelper() {
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
                .build())
        // repos from Bazel modules
        .put(
            "B",
            RepoSpec.builder()
                .setBzlFile(
                    // In real world, this will be @bazel_tools//tools/build_defs/repo:http.bzl,
                    "//tools/build_defs/repo:http.bzl")
                .setRuleClassName("http_archive")
                .setAttributes(
                    ImmutableMap.of(
                        "name", "B",
                        "url", "https://foo/bar/B.zip",
                        "sha256", "1234abcd"))
                .build())
        // repos from module rules
        .put(
            "C",
            RepoSpec.builder()
                .setBzlFile("//maven:repo.bzl")
                .setRuleClassName("maven_repo")
                .setAttributes(
                    ImmutableMap.of(
                        "name", "C",
                        "artifacts",
                            ImmutableList.of("junit:junit:4.12", "com.google.guava:guava:19.0"),
                        "repositories",
                            ImmutableList.of(
                                "https://maven.google.com", "https://repo1.maven.org/maven2")))
                .build());
    return new FakeBzlmodRepoRuleHelper(repoSpecs.buildOrThrow());
  }

  @Test
  public void createRepoRule_bazelTools() throws Exception {
    EvaluationResult<BzlmodRepoRuleValue> result =
        evaluator.evaluate(
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
  public void createRepoRule_localConfigPlatform() throws Exception {
    // Skip this test in Blaze because local_config_platform is not available.
    if (!AnalysisMock.get().isThisBazel()) {
      return;
    }
    EvaluationResult<BzlmodRepoRuleValue> result =
        evaluator.evaluate(
            ImmutableList.of(BzlmodRepoRuleValue.key("local_config_platform")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    BzlmodRepoRuleValue bzlmodRepoRuleValue =
        result.get(BzlmodRepoRuleValue.key("local_config_platform"));
    Rule repoRule = bzlmodRepoRuleValue.getRule();

    assertThat(repoRule.getRuleClass()).isEqualTo("local_config_platform");
    assertThat(repoRule.getName()).isEqualTo("local_config_platform");
  }

  @Test
  public void createRepoRule_overrides() throws Exception {
    EvaluationResult<BzlmodRepoRuleValue> result =
        evaluator.evaluate(ImmutableList.of(BzlmodRepoRuleValue.key("A")), evaluationContext);
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
  public void createRepoRule_bazelModules() throws Exception {
    // Using a starlark rule in a RepoSpec requires having run Selection first.
    evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    EvaluationResult<BzlmodRepoRuleValue> result =
        evaluator.evaluate(ImmutableList.of(BzlmodRepoRuleValue.key("B")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    BzlmodRepoRuleValue bzlmodRepoRuleValue = result.get(BzlmodRepoRuleValue.key("B"));
    Rule repoRule = bzlmodRepoRuleValue.getRule();

    assertThat(repoRule.getRuleClassObject().isStarlark()).isTrue();
    assertThat(repoRule.getRuleClass()).isEqualTo("http_archive");
    assertThat(repoRule.getName()).isEqualTo("B");
    assertThat(repoRule.getAttr("url", Type.STRING)).isEqualTo("https://foo/bar/B.zip");
    assertThat(repoRule.getAttr("sha256", Type.STRING)).isEqualTo("1234abcd");
  }

  @Test
  public void createRepoRule_moduleRules() throws Exception {
    // Using a starlark rule in a RepoSpec requires having run Selection first.
    evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    EvaluationResult<BzlmodRepoRuleValue> result =
        evaluator.evaluate(ImmutableList.of(BzlmodRepoRuleValue.key("C")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    BzlmodRepoRuleValue bzlmodRepoRuleValue = result.get(BzlmodRepoRuleValue.key("C"));
    Rule repoRule = bzlmodRepoRuleValue.getRule();

    assertThat(repoRule.getRuleClassObject().isStarlark()).isTrue();
    assertThat(repoRule.getRuleClass()).isEqualTo("maven_repo");
    assertThat(repoRule.getName()).isEqualTo("C");
    assertThat(repoRule.getAttr("artifacts", Type.STRING_LIST))
        .isEqualTo(ImmutableList.of("junit:junit:4.12", "com.google.guava:guava:19.0"));
    assertThat(repoRule.getAttr("repositories", Type.STRING_LIST))
        .isEqualTo(ImmutableList.of("https://maven.google.com", "https://repo1.maven.org/maven2"));
  }

  @Test
  public void createRepoRule_notFound() throws Exception {
    EvaluationResult<BzlmodRepoRuleValue> result =
        evaluator.evaluate(ImmutableList.of(BzlmodRepoRuleValue.key("unknown")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    BzlmodRepoRuleValue bzlmodRepoRuleValue = result.get(BzlmodRepoRuleValue.key("unknown"));

    assertThat(bzlmodRepoRuleValue).isEqualTo(BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE);
  }
}
