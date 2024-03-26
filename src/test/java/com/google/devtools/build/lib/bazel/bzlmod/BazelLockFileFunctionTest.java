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
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction.BazelLockfileFunctionException;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BzlmodRepoRuleFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
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
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;
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
    RepositoryFunction localRepositoryFunction = new LocalRepositoryFunction();
    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(LocalRepositoryRule.NAME, localRepositoryFunction);

    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder
        .clearWorkspaceFileSuffixForTesting()
        .addStarlarkBootstrap(new RepositoryBootstrap(new StarlarkRepositoryModule()));
    ConfiguredRuleClassProvider ruleClassProvider = builder.build();

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
                    SkyFunctions.REPOSITORY_DIRECTORY,
                    new RepositoryDelegatorFunction(
                        repositoryHandlers,
                        null,
                        new AtomicBoolean(true),
                        ImmutableMap::of,
                        directories,
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(
                    BzlmodRepoRuleValue.BZLMOD_REPO_RULE,
                    new BzlmodRepoRuleFunction(ruleClassProvider, directories))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(
                        ruleClassProvider.getBazelStarlarkEnvironment(),
                        registryFactory,
                        rootDirectory,
                        ImmutableMap.of()))
                .put(SkyFunctions.BAZEL_LOCK_FILE, new BazelLockFileFunction(rootDirectory))
                .put(SkyFunctions.REPO_SPEC, new RepoSpecFunction(registryFactory))
                .put(
                    SkyFunctions.MODULE_EXTENSION_REPO_MAPPING_ENTRIES,
                    new ModuleExtensionRepoMappingEntriesFunction())
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
                          throws BazelLockfileFunctionException, InterruptedException {

                        UpdateLockFileKey key = (UpdateLockFileKey) skyKey;
                        BzlmodFlagsAndEnvVars flags = BazelDepGraphFunction.getFlagsAndEnvVars(env);
                        if (flags == null) {
                          return null;
                        }

                        ImmutableMap<String, String> localOverrideHashes =
                            BazelDepGraphFunction.getLocalOverridesHashes(key.overrides(), env);
                        if (localOverrideHashes == null) {
                          return null;
                        }
                        BazelLockFileModule.updateLockfile(
                            rootDirectory,
                            BazelLockFileValue.builder()
                                .setModuleFileHash(key.moduleHash())
                                .setFlags(flags)
                                .setLocalOverrideHashes(localOverrideHashes)
                                .setModuleDepGraph(key.depGraph())
                                .build());

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
    YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of());
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.ERROR);
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.ERROR);
    BazelLockFileFunction.LOCKFILE_MODE.set(differencer, LockfileMode.UPDATE);
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(differencer, ImmutableMap.of());
    RepositoryDelegatorFunction.FORCE_FETCH.set(
        differencer, RepositoryDelegatorFunction.FORCE_FETCH_DISABLED);
    RepositoryDelegatorFunction.VENDOR_DIRECTORY.set(differencer, Optional.empty());
  }

  @Test
  public void simpleModule() throws Exception {
    scratch.overwriteFile(
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
    RootModuleFileValue rootValue = rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE);

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.of(ModuleKey.ROOT, InterimModule.toModule(rootValue.getModule(), null, null));

    UpdateLockFileKey key =
        UpdateLockFileKey.create("moduleHash", depGraph, rootValue.getOverrides());
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
  public void moduleWithFlags() throws Exception {
    // Test having --override_module, --ignore_dev_dependency, --check_bazel_compatibility
    // --check_direct_dependencies & --registry
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')");

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }
    RootModuleFileValue rootValue = rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE);

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.of(ModuleKey.ROOT, InterimModule.toModule(rootValue.getModule(), null, null));

    ImmutableList<String> yankedVersions = ImmutableList.of("2.4", "2.3");
    LocalPathOverride override = LocalPathOverride.create("override_path");
    ImmutableList<String> registries = ImmutableList.of("registry1", "registry2");
    ImmutableMap<String, String> moduleOverride = ImmutableMap.of("my_dep_1", override.getPath());

    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, true);
    ModuleFileFunction.REGISTRIES.set(differencer, registries);
    ModuleFileFunction.MODULE_OVERRIDES.set(differencer, ImmutableMap.of("my_dep_1", override));
    YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.set(differencer, yankedVersions);
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.WARNING);
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.WARNING);

    UpdateLockFileKey key =
        UpdateLockFileKey.create("moduleHash", depGraph, rootValue.getOverrides());
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
        .isEqualTo(CheckDirectDepsMode.WARNING.toString());
    assertThat(value.getFlags().compatibilityMode())
        .isEqualTo(BazelCompatibilityMode.WARNING.toString());
  }

  @Test
  public void moduleWithLocalOverrides() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='root',version='0.1')",
        "local_path_override(module_name='ss',path='code_for_ss')");
    scratch.file(
        rootDirectory.getRelative("code_for_ss/MODULE.bazel").getPathString(),
        "module(name='ss',version='1.0')");
    scratch.file(rootDirectory.getRelative("code_for_ss/WORKSPACE").getPathString());

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }
    RootModuleFileValue rootValue = rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE);

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.of(ModuleKey.ROOT, InterimModule.toModule(rootValue.getModule(), null, null));

    UpdateLockFileKey key =
        UpdateLockFileKey.create("moduleHash", depGraph, rootValue.getOverrides());
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
    assertThat(value.getLocalOverrideHashes()).isNotEmpty();
  }

  @Test
  public void fullModule() throws Exception {
    scratch.overwriteFile(
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
    RootModuleFileValue rootValue = rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE);

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.of(ModuleKey.ROOT, InterimModule.toModule(rootValue.getModule(), null, null));

    UpdateLockFileKey key =
        UpdateLockFileKey.create("moduleHash", depGraph, rootValue.getOverrides());
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
  public void invalidLockfileEmptyFile() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')");

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }
    RootModuleFileValue rootValue = rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE);

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.of(ModuleKey.ROOT, InterimModule.toModule(rootValue.getModule(), null, null));

    UpdateLockFileKey key =
        UpdateLockFileKey.create("moduleHash", depGraph, rootValue.getOverrides());
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel.lock").getPathString(),
        "{\"lockFileVersion\": " + BazelLockFileValue.LOCK_FILE_VERSION + "}");

    result = evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (!result.hasError()) {
      fail("expected error about missing field in the lockfile, but succeeded");
    }
    assertThat(result.getError().toString())
        .contains(
            "Failed to read and parse the MODULE.bazel.lock file with error: "
                + "java.lang.IllegalStateException: Missing required properties: moduleFileHash "
                + "flags localOverrideHashes moduleDepGraph. Try deleting it and rerun the build.");
  }

  @Test
  public void invalidLockfileNullFlag() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')");

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }
    RootModuleFileValue rootValue = rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE);

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.of(ModuleKey.ROOT, InterimModule.toModule(rootValue.getModule(), null, null));

    UpdateLockFileKey key =
        UpdateLockFileKey.create("moduleHash", depGraph, rootValue.getOverrides());
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    JsonObject jsonObject =
        (JsonObject) JsonParser.parseString(scratch.readFile("MODULE.bazel.lock"));
    jsonObject.get("flags").getAsJsonObject().remove("directDependenciesMode");
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel.lock").getPathString(), jsonObject.toString());

    result = evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (!result.hasError()) {
      fail("expected error about missing field in the lockfile, but succeeded");
    }
    assertThat(result.getError().toString())
        .contains(
            "Failed to read and parse the MODULE.bazel.lock file with error: Null"
                + " directDependenciesMode. Try deleting it and rerun the build.");
  }

  @Test
  public void invalidLockfileMalformed() throws Exception {
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')");

    EvaluationResult<RootModuleFileValue> rootResult =
        evaluator.evaluate(
            ImmutableList.of(ModuleFileValue.KEY_FOR_ROOT_MODULE), evaluationContext);
    if (rootResult.hasError()) {
      fail(rootResult.getError().toString());
    }
    RootModuleFileValue rootValue = rootResult.get(ModuleFileValue.KEY_FOR_ROOT_MODULE);

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.of(ModuleKey.ROOT, InterimModule.toModule(rootValue.getModule(), null, null));

    UpdateLockFileKey key =
        UpdateLockFileKey.create("moduleHash", depGraph, rootValue.getOverrides());
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    JsonObject jsonObject =
        (JsonObject) JsonParser.parseString(scratch.readFile("MODULE.bazel.lock"));
    jsonObject.get("flags").getAsJsonObject().addProperty("allowedYankedVersions", "string!");
    scratch.overwriteFile(
        rootDirectory.getRelative("MODULE.bazel.lock").getPathString(), jsonObject.toString());

    result = evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (!result.hasError()) {
      fail("expected error about invalid field value in the lockfile, but succeeded");
    }
    Pattern expectedExceptionMessage =
        Pattern.compile(
            Pattern.quote(
                    "Failed to read and parse the MODULE.bazel.lock file with error:"
                        + " java.lang.IllegalStateException: Expected BEGIN_ARRAY but was STRING at"
                        + " line 1 column 129 path $.flags.allowedYankedVersions")
                + ".*"
                + Pattern.quote("Try deleting it and rerun the build."),
            Pattern.DOTALL);
    assertThat(result.getError().toString()).containsMatch(expectedExceptionMessage);
  }

  @AutoValue
  abstract static class UpdateLockFileKey implements SkyKey {

    abstract String moduleHash();

    abstract ImmutableMap<ModuleKey, Module> depGraph();

    abstract ImmutableMap<String, ModuleOverride> overrides();

    static UpdateLockFileKey create(
        String moduleHash,
        ImmutableMap<ModuleKey, Module> depGraph,
        ImmutableMap<String, ModuleOverride> overrides) {
      return new AutoValue_BazelLockFileFunctionTest_UpdateLockFileKey(
          moduleHash, depGraph, overrides);
    }

    @Override
    public SkyFunctionName functionName() {
      return updateLockfileFunction;
    }
  }
}
