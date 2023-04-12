// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
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
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelLockFileFunction}. */
@RunWith(JUnit4.class)
public class BazelLockFileFunctionTest extends FoundationTestCase {

  private MemoizingEvaluator evaluator;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;
  private static SkyFunctionName updateLockfileFunction;

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

    updateLockfileFunction = SkyFunctionName.createHermetic("LockfileWrite");

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
                    new ModuleFileFunction(registryFactory, rootDirectory, ImmutableMap.of()))
                .put(SkyFunctions.BAZEL_LOCK_FILE, new BazelLockFileFunction(rootDirectory))
                .put(
                    SkyFunctions.CLIENT_ENVIRONMENT_VARIABLE,
                    new ClientEnvironmentFunction(
                        new AtomicReference<>(ImmutableMap.of("BZLMOD_ALLOW_YANKED_VERSIONS", ""))))
                .put(
                    updateLockfileFunction,
                    new SkyFunction() {
                      @Nullable
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env)
                          throws SkyFunctionException, InterruptedException {

                        UpdateLockFileKey key = (UpdateLockFileKey) skyKey;
                        BzlmodFlagsAndEnvVars flags = BazelDepGraphFunction.getFlagsAndEnvVars(env);
                        if (flags == null) {
                          return null;
                        }
                        BazelLockFileFunction.updateLockedModule(
                            key.moduleHash(), key.depGraph(), rootDirectory, flags);
                        return new SkyValue() {};
                      }
                    })
                .buildOrThrow(),
            differencer);

    PrecomputedValue.STARLARK_SEMANTICS.set(
        differencer,
        StarlarkSemantics.builder().setBool(BuildLanguageOptions.ENABLE_BZLMOD, true).build());
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of());
    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, true);
    ModuleFileFunction.MODULE_OVERRIDES.set(differencer, ImmutableMap.of());
    BazelModuleResolutionFunction.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of());
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.ERROR);
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.ERROR);
  }

  @Test
  public void simpleModule() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')",
        "bazel_dep(name = 'dep_1', version = '1.0')",
        "bazel_dep(name = 'dep_2', version = '2.0')");

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(ModuleKey.ROOT, rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE).getModule())
            .buildOrThrow();

    UpdateLockFileKey key = UpdateLockFileKey.create("moduleHash", depGraph);
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    result = evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    BazelLockFileValue value = result.get(BazelLockFileValue.KEY);
    assertThat(value.getModuleDepGraph()).isEqualTo(depGraph);
  }

  @Test
  public void simpleModuleWithFlags() throws Exception {
    // Test having --override_module, --ignore_dev_dependency, --check_bazel_compatibility
    // --check_direct_dependencies & --registry
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')");

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(ModuleKey.ROOT, rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE).getModule())
            .buildOrThrow();

    ImmutableList<String> yankedVersions = ImmutableList.of("2.4", "2.3");
    LocalPathOverride override = LocalPathOverride.create("override_path");
    ImmutableList<String> registries = ImmutableList.of("registry1", "registry2");
    ImmutableMap<String, String> moduleOverride = ImmutableMap.of("my_dep_1", override.getPath());

    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, true);
    ModuleFileFunction.REGISTRIES.set(differencer, registries);
    ModuleFileFunction.MODULE_OVERRIDES.set(differencer, ImmutableMap.of("my_dep_1", override));
    BazelModuleResolutionFunction.ALLOWED_YANKED_VERSIONS.set(differencer, yankedVersions);
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.ERROR);
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.ERROR);

    UpdateLockFileKey key = UpdateLockFileKey.create("moduleHash", depGraph);
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    result = evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    BazelLockFileValue value = result.get(BazelLockFileValue.KEY);
    assertThat(value.getModuleDepGraph()).isEqualTo(depGraph);
    assertThat(value.getFlags().ignoreDevDependency()).isTrue();
    assertThat(value.getFlags().cmdRegistries()).isEqualTo(registries);
    assertThat(value.getFlags().cmdModuleOverrides()).isEqualTo(moduleOverride);
    assertThat(value.getFlags().allowedYankedVersions()).isEqualTo(yankedVersions);
    assertThat(value.getFlags().directDependenciesMode())
        .isEqualTo(CheckDirectDepsMode.ERROR.toString());
    assertThat(value.getFlags().compatibilityMode())
        .isEqualTo(BazelCompatibilityMode.ERROR.toString());
  }

  @Test
  public void fullModule() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')",
        "register_toolchains('//my:toolchain', '//my:toolchain2')",
        "ext1 = use_extension('//:defs.bzl','ext_1')",
        "use_repo(ext1,'myrepo')",
        "ext2 = use_extension('@ext//:defs.bzl','ext_2')",
        "ext2.tag(file='@myrepo//:Hello1.txt')",
        "ext2.tag(file='@myrepo//:Hello2.txt')",
        "use_repo(ext2,'ext_repo')");

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(ModuleKey.ROOT, rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE).getModule())
            .buildOrThrow();

    UpdateLockFileKey key = UpdateLockFileKey.create("moduleHash", depGraph);
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    result = evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    BazelLockFileValue value = result.get(BazelLockFileValue.KEY);
    assertThat(value.getModuleDepGraph()).isEqualTo(depGraph);
  }

  @AutoValue
  abstract static class UpdateLockFileKey implements SkyKey {

    abstract String moduleHash();

    abstract ImmutableMap<ModuleKey, Module> depGraph();

    static UpdateLockFileKey create(String moduleHash, ImmutableMap<ModuleKey, Module> depGraph) {
      return new AutoValue_BazelLockFileFunctionTest_UpdateLockFileKey(moduleHash, depGraph);
    }

    @Override
    public SkyFunctionName functionName() {
      return updateLockfileFunction;
    }
  }
}
