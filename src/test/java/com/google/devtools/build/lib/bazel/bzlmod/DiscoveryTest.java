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
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.InterimModuleBuilder;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
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
import com.google.devtools.build.lib.vfs.Path;
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
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Discovery}. */
@RunWith(JUnit4.class)
public class DiscoveryTest extends FoundationTestCase {

  private Path workspaceRoot;
  private MemoizingEvaluator evaluator;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @AutoValue
  abstract static class DiscoveryValue implements SkyValue {
    static final SkyFunctionName FUNCTION_NAME = SkyFunctionName.createHermetic("test_discovery");
    static final SkyKey KEY = () -> FUNCTION_NAME;

    static DiscoveryValue create(ImmutableMap<ModuleKey, InterimModule> depGraph) {
      return new AutoValue_DiscoveryTest_DiscoveryValue(depGraph);
    }

    abstract ImmutableMap<ModuleKey, InterimModule> getDepGraph();
  }

  static class DiscoveryFunction implements SkyFunction {
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      RootModuleFileValue root =
          (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
      if (root == null) {
        return null;
      }
      ImmutableMap<ModuleKey, InterimModule> depGraph;
      try {
        depGraph = Discovery.run(env, root);
      } catch (ExternalDepsException e) {
        throw new BazelModuleResolutionFunction.BazelModuleResolutionFunctionException(
            e, SkyFunctionException.Transience.PERSISTENT);
      }
      return depGraph == null ? null : DiscoveryValue.create(depGraph);
    }
  }

  @Before
  public void setup() throws Exception {
    setUpWithBuiltinModules(ImmutableMap.of());
  }

  private void setUpWithBuiltinModules(ImmutableMap<String, NonRegistryOverride> builtinModules)
      throws Exception {
    workspaceRoot = scratch.dir("/ws");
    differencer = new SequencedRecordingDifferencer();
    evaluationContext =
        EvaluationContext.newBuilder().setParallelism(8).setEventHandler(reporter).build();
    registryFactory = new FakeRegistry.Factory();
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

    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(LocalRepositoryRule.NAME, new LocalRepositoryFunction());
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
                .put(DiscoveryValue.FUNCTION_NAME, new DiscoveryFunction())
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(
                        ruleClassProvider.getBazelStarlarkEnvironment(),
                        registryFactory,
                        workspaceRoot,
                        builtinModules))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
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
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(differencer, ImmutableMap.of());
    RepositoryDelegatorFunction.FORCE_FETCH.set(
        differencer, RepositoryDelegatorFunction.FORCE_FETCH_DISABLED);
    RepositoryDelegatorFunction.VENDOR_DIRECTORY.set(differencer, Optional.empty());

    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, packageLocator.get());
    RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.set(
        differencer, Optional.empty());
    PrecomputedValue.REPO_ENV.set(differencer, ImmutableMap.of());
    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, false);
    ModuleFileFunction.MODULE_OVERRIDES.set(differencer, ImmutableMap.of());
    YankedVersionsUtil.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of());
  }

  @Test
  public void testSimpleDiamond() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='1.0')",
        "bazel_dep(name='ccc',version='2.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "1.0"),
                "module(name='bbb', version='1.0');bazel_dep(name='ddd',version='3.0')")
            .addModule(
                createModuleKey("ccc", "2.0"),
                "module(name='ccc', version='2.0');bazel_dep(name='ddd',version='3.0')")
            .addModule(
                createModuleKey("ddd", "3.0"),
                // Add a random override here; it should be ignored
                "module(name='ddd', version='3.0');local_path_override(module_name='ff',path='f')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "1.0"))
                .addDep("ccc", createModuleKey("ccc", "2.0"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "1.0")
                .addDep("ddd", createModuleKey("ddd", "3.0"))
                .setRegistry(registry)
                .buildEntry(),
            InterimModuleBuilder.create("ccc", "2.0")
                .addDep("ddd", createModuleKey("ddd", "3.0"))
                .setRegistry(registry)
                .buildEntry(),
            InterimModuleBuilder.create("ddd", "3.0").setRegistry(registry).buildEntry());
  }

  @Test
  public void testDevDependency() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='1.0')",
        "bazel_dep(name='ccc',version='1.0',dev_dependency=True)");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "1.0"),
                "module(name='bbb', version='1.0')",
                "bazel_dep(name='ccc',version='2.0',dev_dependency=True)")
            .addModule(createModuleKey("ccc", "1.0"), "module(name='ccc', version='1.0')")
            .addModule(createModuleKey("ccc", "2.0"), "module(name='ccc', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "1.0"))
                .addDep("ccc", createModuleKey("ccc", "1.0"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "1.0").setRegistry(registry).buildEntry(),
            InterimModuleBuilder.create("ccc", "1.0").setRegistry(registry).buildEntry());
  }

  @Test
  public void testIgnoreDevDependency() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='1.0')",
        "bazel_dep(name='ccc',version='1.0',dev_dependency=True)");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "1.0"),
                "module(name='bbb', version='1.0')",
                "bazel_dep(name='ccc',version='2.0',dev_dependency=True)")
            .addModule(createModuleKey("ccc", "1.0"), "module(name='ccc', version='1.0')")
            .addModule(createModuleKey("ccc", "2.0"), "module(name='ccc', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, true);

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "1.0"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "1.0").setRegistry(registry).buildEntry());
  }

  @Test
  public void testCircularDependency() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "1.0"),
                "module(name='bbb', version='1.0');bazel_dep(name='ccc',version='2.0')")
            .addModule(
                createModuleKey("ccc", "2.0"),
                "module(name='ccc', version='2.0');bazel_dep(name='bbb',version='1.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "1.0"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "1.0")
                .addDep("ccc", createModuleKey("ccc", "2.0"))
                .setRegistry(registry)
                .buildEntry(),
            InterimModuleBuilder.create("ccc", "2.0")
                .addDep("bbb", createModuleKey("bbb", "1.0"))
                .setRegistry(registry)
                .buildEntry());
  }

  @Test
  public void testCircularDependencyOnRootModule() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "1.0"),
                "module(name='bbb', version='1.0');bazel_dep(name='aaa',version='2.0')")
            .addModule(createModuleKey("aaa", "2.0"), "module(name='aaa', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "1.0"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "1.0")
                .addDep("aaa", ModuleKey.ROOT)
                .addOriginalDep("aaa", createModuleKey("aaa", "2.0"))
                .setRegistry(registry)
                .buildEntry());
  }

  @Test
  public void testSingleVersionOverride() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='0.1')",
        "single_version_override(module_name='ccc',version='2.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "0.1"),
                "module(name='bbb', version='0.1');bazel_dep(name='ccc',version='1.0')")
            .addModule(createModuleKey("ccc", "1.0"), "module(name='ccc', version='1.0');")
            .addModule(createModuleKey("ccc", "2.0"), "module(name='ccc', version='2.0');");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "0.1"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "0.1")
                .addDep("ccc", createModuleKey("ccc", "2.0"))
                .addOriginalDep("ccc", createModuleKey("ccc", "1.0"))
                .setRegistry(registry)
                .buildEntry(),
            InterimModuleBuilder.create("ccc", "2.0").setRegistry(registry).buildEntry());
  }

  @Test
  public void testRegistryOverride() throws Exception {
    FakeRegistry registry1 =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "0.1"),
                "module(name='bbb', version='0.1');bazel_dep(name='ccc',version='1.0')")
            .addModule(createModuleKey("ccc", "1.0"), "module(name='ccc', version='1.0');");
    FakeRegistry registry2 =
        registryFactory
            .newFakeRegistry("/bar")
            .addModule(
                createModuleKey("ccc", "1.0"),
                "module(name='ccc', version='1.0');bazel_dep(name='bbb',version='0.1')");
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='0.1')",
        "single_version_override(module_name='ccc',registry='" + registry2.getUrl() + "')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry1.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "0.1"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "0.1")
                .addDep("ccc", createModuleKey("ccc", "1.0"))
                .setRegistry(registry1)
                .buildEntry(),
            InterimModuleBuilder.create("ccc", "1.0")
                .addDep("bbb", createModuleKey("bbb", "0.1"))
                .setRegistry(registry2)
                .buildEntry());
  }

  @Test
  public void testLocalPathOverride() throws Exception {
    Path pathToC = scratch.dir("/pathToC");
    scratch.file(
        pathToC.getRelative("MODULE.bazel").getPathString(), "module(name='ccc',version='2.0')");
    scratch.file(pathToC.getRelative("WORKSPACE").getPathString());
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='aaa',version='0.1')",
        "bazel_dep(name='bbb',version='0.1')",
        "local_path_override(module_name='ccc',path='" + pathToC.getPathString() + "')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("bbb", "0.1"),
                "module(name='bbb', version='0.1');bazel_dep(name='ccc',version='1.0')")
            .addModule(createModuleKey("ccc", "1.0"), "module(name='ccc', version='1.0');");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("aaa", "0.1")
                .setKey(ModuleKey.ROOT)
                .addDep("bbb", createModuleKey("bbb", "0.1"))
                .buildEntry(),
            InterimModuleBuilder.create("bbb", "0.1")
                .addDep("ccc", createModuleKey("ccc", ""))
                .addOriginalDep("ccc", createModuleKey("ccc", "1.0"))
                .setRegistry(registry)
                .buildEntry(),
            InterimModuleBuilder.create("ccc", "2.0")
                .setKey(createModuleKey("ccc", ""))
                .buildEntry());
  }

  @Test
  public void testBuiltinModules_forRoot() throws Exception {
    ImmutableMap<String, NonRegistryOverride> builtinModules =
        ImmutableMap.of(
            "bazel_tools",
            LocalPathOverride.create(rootDirectory.getRelative("tools").getPathString()),
            "local_config_platform",
            LocalPathOverride.create(rootDirectory.getRelative("localplat").getPathString()));
    setUpWithBuiltinModules(builtinModules);
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='foo',version='2.0')");
    scratch.file(rootDirectory.getRelative("tools/WORKSPACE").getPathString());
    scratch.file(
        rootDirectory.getRelative("tools/MODULE.bazel").getPathString(),
        "module(name='bazel_tools',version='1.0')",
        "bazel_dep(name='foo',version='1.0')");
    scratch.file(rootDirectory.getRelative("localplat/WORKSPACE").getPathString());
    scratch.file(
        rootDirectory.getRelative("localplat/MODULE.bazel").getPathString(),
        "module(name='local_config_platform')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(createModuleKey("foo", "1.0"), "module(name='foo', version='1.0')")
            .addModule(createModuleKey("foo", "2.0"), "module(name='foo', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        evaluator.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getDepGraph().entrySet())
        .containsExactly(
            InterimModuleBuilder.create("", "")
                .addDep("bazel_tools", createModuleKey("bazel_tools", ""))
                .addDep("local_config_platform", createModuleKey("local_config_platform", ""))
                .addDep("foo", createModuleKey("foo", "2.0"))
                .buildEntry(),
            InterimModuleBuilder.create("bazel_tools", "1.0")
                .setKey(createModuleKey("bazel_tools", ""))
                .addDep("local_config_platform", createModuleKey("local_config_platform", ""))
                .addDep("foo", createModuleKey("foo", "1.0"))
                .buildEntry(),
            InterimModuleBuilder.create("local_config_platform", "")
                .setKey(createModuleKey("local_config_platform", ""))
                .addDep("bazel_tools", createModuleKey("bazel_tools", ""))
                .buildEntry(),
            InterimModuleBuilder.create("foo", "1.0")
                .addDep("bazel_tools", createModuleKey("bazel_tools", ""))
                .addDep("local_config_platform", createModuleKey("local_config_platform", ""))
                .setRegistry(registry)
                .buildEntry(),
            InterimModuleBuilder.create("foo", "2.0")
                .addDep("bazel_tools", createModuleKey("bazel_tools", ""))
                .addDep("local_config_platform", createModuleKey("local_config_platform", ""))
                .setRegistry(registry)
                .buildEntry());
  }
}
