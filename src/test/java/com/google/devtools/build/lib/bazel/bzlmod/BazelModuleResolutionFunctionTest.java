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
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelModuleResolutionFunction}. */
@RunWith(JUnit4.class)
public class BazelModuleResolutionFunctionTest extends FoundationTestCase {

  private MemoizingEvaluator evaluator;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setup() throws Exception {
    differencer = new SequencedRecordingDifferencer();
    registryFactory = new FakeRegistry.Factory();
    evaluationContext =
        EvaluationContext.newBuilder().setParallelism(8).setEventHandler(reporter).build();

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

    evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(FileValue.FILE, new FileFunction(packageLocator, directories))
                .put(
                    FileStateKey.FILE_STATE,
                    new FileStateFunction(
                        Suppliers.ofInstance(
                            new TimestampGranularityMonitor(BlazeClock.instance())),
                        SyscallCache.NO_CACHE,
                        externalFilesHelper))
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(
                        TestRuleClassProvider.getRuleClassProvider().getBazelStarlarkEnvironment(),
                        registryFactory,
                        rootDirectory,
                        ImmutableMap.of()))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .put(SkyFunctions.BAZEL_DEP_GRAPH, new BazelDepGraphFunction())
                .put(SkyFunctions.BAZEL_LOCK_FILE, new BazelLockFileFunction(rootDirectory))
                .put(SkyFunctions.BAZEL_MODULE_RESOLUTION, new BazelModuleResolutionFunction())
                .put(SkyFunctions.REPO_SPEC, new RepoSpecFunction(registryFactory))
                .put(
                    SkyFunctions.MODULE_EXTENSION_REPO_MAPPING_ENTRIES,
                    new ModuleExtensionRepoMappingEntriesFunction())
                .put(
                    SkyFunctions.CLIENT_ENVIRONMENT_VARIABLE,
                    new ClientEnvironmentFunction(new AtomicReference<>(ImmutableMap.of())))
                .buildOrThrow(),
            differencer);

    PrecomputedValue.STARLARK_SEMANTICS.set(
        differencer,
        StarlarkSemantics.builder().setBool(BuildLanguageOptions.ENABLE_BZLMOD, true).build());
    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, false);
    ModuleFileFunction.MODULE_OVERRIDES.set(differencer, ImmutableMap.of());
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.OFF);
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.ERROR);
    BazelLockFileFunction.LOCKFILE_MODE.set(differencer, LockfileMode.UPDATE);
    YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of());
  }

  @Test
  public void testBazelInvalidCompatibility() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0dd'])");

    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent("invalid version argument '>5.1.0dd'");
  }

  @Test
  public void testSimpleBazelCompatibilityFailure() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.4'])");

    embedBazelVersion("5.1.4");
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "Bazel version 5.1.4 is not compatible with module \"<root>\" (bazel_compatibility:"
            + " [>5.1.0, <5.1.4])");
  }

  @Test
  public void testBazelCompatibilityWarning() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.4'])");

    embedBazelVersion("5.1.4");
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.WARNING);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isFalse();
    assertContainsEvent(
        "Bazel version 5.1.4 is not compatible with module \"<root>\" (bazel_compatibility:"
            + " [>5.1.0, <5.1.4])");
  }

  @Test
  public void testDisablingBazelCompatibility() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.4'])");

    embedBazelVersion("5.1.4");
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.OFF);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isFalse();
    assertDoesNotContainEvent(
        "Bazel version 5.1.4 is not compatible with module \"<root>\" (bazel_compatibility:"
            + " [>5.1.0, <5.1.4])");
  }

  @Test
  public void testBazelCompatibilitySuccess() throws Exception {
    setupModulesForCompatibility();

    embedBazelVersion("5.1.4-pre.20220421.3");
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testBazelCompatibilityFailure() throws Exception {
    setupModulesForCompatibility();

    embedBazelVersion("5.1.5rc444");
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "Bazel version 5.1.5rc444 is not compatible with module \"b@1.0\" (bazel_compatibility:"
            + " [<=5.1.4, -5.1.2])");
  }

  @Test
  public void testRcIsCompatibleWithReleaseRequirement() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>=6.4.0'])");

    embedBazelVersion("6.4.0rc1");
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testPrereleaseIsNotCompatibleWithReleaseRequirement() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>=6.4.0'])");

    embedBazelVersion("6.4.0-pre-1");
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "Bazel version 6.4.0-pre-1 is not compatible with module \"<root>\" (bazel_compatibility:"
            + " [>=6.4.0])");
  }

  private void embedBazelVersion(String version) {
    // Double-get version-info to determine if it's the cached instance or not, and if not cache it.
    BlazeVersionInfo blazeInfo1 = BlazeVersionInfo.instance();
    BlazeVersionInfo blazeInfo2 = BlazeVersionInfo.instance();
    if (blazeInfo1 != blazeInfo2) {
      BlazeVersionInfo.setBuildInfo(ImmutableMap.of());
      blazeInfo1 = BlazeVersionInfo.instance();
    }

    // embed new version
    Map<String, String> blazeInfo = blazeInfo1.getBuildData();
    blazeInfo.remove(BlazeVersionInfo.BUILD_LABEL);
    blazeInfo.put(BlazeVersionInfo.BUILD_LABEL, version);
  }

  private void setupModulesForCompatibility() throws IOException {
    /* Root depends on "a" which depends on "b"
       The only versions that would work with root, a and b compatibility constrains are between
       -not including- 5.1.2 and 5.1.4.
       Ex: 5.1.3rc44, 5.1.3, 5.1.4-pre22.44
    */
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.6'])",
        "bazel_dep(name = 'a', version = '1.0')");

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("a", "1.0"),
                "module(name='a', version='1.0', bazel_compatibility=['>=5.1.2', '-5.1.4']);",
                "bazel_dep(name='b', version='1.0')")
            .addModule(
                createModuleKey("b", "1.0"),
                "module(name='b', version='1.0', bazel_compatibility=['<=5.1.4', '-5.1.2']);");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
  }

  @Test
  public void testYankedVersionCheckSuccess() throws Exception {
    setupModulesForYankedVersion();
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString())
        .contains(
            "Yanked version detected in your resolved dependency graph: b@1.0, for the reason: 1.0"
                + " is a bad version!");
  }

  @Test
  public void testYankedVersionCheckIgnoredByAll() throws Exception {
    setupModulesForYankedVersion();
    YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of("all"));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testYankedVersionCheckIgnoredBySpecific() throws Exception {
    setupModulesForYankedVersion();
    YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of("b@1.0"));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testBadYankedVersionFormat() throws Exception {
    setupModulesForYankedVersion();
    YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of("b~1.0"));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString())
        .contains(
            "Parsing command line flag --allow_yanked_versions=b~1.0 failed, module versions must"
                + " be of the form '<module name>@<version>'");
  }

  private void setupModulesForYankedVersion() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0')",
        "bazel_dep(name = 'a', version = '1.0')");

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/bar")
            .addModule(
                createModuleKey("a", "1.0"),
                "module(name='a', version='1.0');",
                "bazel_dep(name='b', version='1.0')")
            .addModule(createModuleKey("b", "1.0"), "module(name='b', version='1.0');")
            .addYankedVersion("b", ImmutableMap.of(Version.parse("1.0"), "1.0 is a bad version!"));
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
  }

  @Test
  public void testYankedVersionSideEffects_equalCompatibilityLevel() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0')",
        "bazel_dep(name = 'a', version = '1.0')",
        "bazel_dep(name = 'b', version = '1.1')");

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/bar")
            .addModule(
                createModuleKey("a", "1.0"),
                "module(name='a', version='1.0')",
                "bazel_dep(name='b', version='1.0')")
            .addModule(createModuleKey("c", "1.0"), "module(name='c', version='1.0')")
            .addModule(createModuleKey("c", "1.1"), "module(name='c', version='1.1')")
            .addModule(
                createModuleKey("b", "1.0"),
                "module(name='b', version='1.0', compatibility_level = 2)",
                "bazel_dep(name='c', version='1.1')",
                "print('hello from yanked version')")
            .addModule(
                createModuleKey("b", "1.1"),
                "module(name='b', version='1.1', compatibility_level = 2)",
                "bazel_dep(name='c', version='1.0')")
            .addYankedVersion("b", ImmutableMap.of(Version.parse("1.0"), "1.0 is a bad version!"));

    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isFalse();
    assertThat(result.get(BazelModuleResolutionValue.KEY).getResolvedDepGraph().keySet())
        .containsExactly(
            ModuleKey.ROOT,
            createModuleKey("a", "1.0"),
            createModuleKey("b", "1.1"),
            createModuleKey("c", "1.0"));
    assertDoesNotContainEvent("hello from yanked version");
  }

  @Test
  public void testYankedVersionSideEffects_differentCompatibilityLevel() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0')",
        "bazel_dep(name = 'a', version = '1.0')",
        "bazel_dep(name = 'b', version = '1.1')");

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/bar")
            .addModule(
                createModuleKey("a", "1.0"),
                "module(name='a', version='1.0')",
                "bazel_dep(name='b', version='1.0')")
            .addModule(createModuleKey("c", "1.0"), "module(name='c', version='1.0')")
            .addModule(createModuleKey("c", "1.1"), "module(name='c', version='1.1')")
            .addModule(
                createModuleKey("b", "1.0"),
                "module(name='b', version='1.0', compatibility_level = 2)",
                "bazel_dep(name='c', version='1.1')",
                "print('hello from yanked version')")
            .addModule(
                createModuleKey("b", "1.1"),
                "module(name='b', version='1.1', compatibility_level = 3)",
                "bazel_dep(name='c', version='1.0')")
            .addYankedVersion("b", ImmutableMap.of(Version.parse("1.0"), "1.0 is a bad version!"));

    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString())
        .contains(
            "a@1.0 depends on b@1.0 with compatibility level 2, but <root> depends on b@1.1 with"
                + " compatibility level 3 which is different");
    assertDoesNotContainEvent("hello from yanked version");
  }

  @Test
  public void overrideOnNonexistentModule() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0')",
        "bazel_dep(name = 'a', version = '1.0')",
        "bazel_dep(name = 'b', version = '1.1')",
        "local_path_override(module_name='d', path='whatevs')");

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/bar")
            .addModule(
                createModuleKey("a", "1.0"),
                "module(name='a', version='1.0')",
                "bazel_dep(name='b', version='1.0')")
            .addModule(createModuleKey("c", "1.0"), "module(name='c', version='1.0')")
            .addModule(createModuleKey("c", "1.1"), "module(name='c', version='1.1')")
            .addModule(
                createModuleKey("b", "1.0"),
                "module(name='b', version='1.0')",
                "bazel_dep(name='c', version='1.1')")
            .addModule(
                createModuleKey("b", "1.1"),
                "module(name='b', version='1.1')",
                "bazel_dep(name='c', version='1.0')");

    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString())
        .contains("the root module specifies overrides on nonexistent module(s): d");
  }
}
