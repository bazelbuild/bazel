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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BzlmodRepoRuleFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
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
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ModuleFileFunction}. */
@RunWith(JUnit4.class)
public class ModuleFileFunctionTest extends FoundationTestCase {

  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setup() throws Exception {
    differencer = new SequencedRecordingDifferencer();
    evaluationContext =
        EvaluationContext.newBuilder().setNumThreads(8).setEventHandler(reporter).build();
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

    PackageFactory packageFactory =
        AnalysisMock.get()
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);

    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(LocalRepositoryRule.NAME, new LocalRepositoryFunction());
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
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(registryFactory, rootDirectory))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .put(
                    SkyFunctions.REPOSITORY_DIRECTORY,
                    new RepositoryDelegatorFunction(
                        repositoryHandlers,
                        null,
                        new AtomicBoolean(true),
                        ImmutableMap::of,
                        directories,
                        ManagedDirectoriesKnowledge.NO_MANAGED_DIRECTORIES,
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(
                    BzlmodRepoRuleValue.BZLMOD_REPO_RULE,
                    new BzlmodRepoRuleFunction(
                        packageFactory,
                        ruleClassProvider,
                        directories,
                        new BzlmodRepoRuleHelperImpl()))
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(differencer, ImmutableMap.of());
    RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING.set(
        differencer, RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, packageLocator.get());
    RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.set(
        differencer, Optional.empty());
    PrecomputedValue.REPO_ENV.set(differencer, ImmutableMap.of());
    RepositoryDelegatorFunction.OUTPUT_VERIFICATION_REPOSITORY_RULES.set(
        differencer, ImmutableSet.of());
    RepositoryDelegatorFunction.RESOLVED_FILE_FOR_VERIFICATION.set(differencer, Optional.empty());
    RepositoryDelegatorFunction.ENABLE_BZLMOD.set(differencer, true);
  }

  @Test
  public void testRootModule() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1',compatibility_level=4)",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0',repo_name='see')",
        "single_version_override(module_name='D',version='18')",
        "local_path_override(module_name='E',path='somewhere/else')",
        "multiple_version_override(module_name='F',versions=['1.0','2.0'])",
        "archive_override(module_name='G',urls=['https://hello.com/world.zip'])");
    FakeRegistry registry = registryFactory.newFakeRegistry();
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<RootModuleFileValue> result =
        driver.evaluate(ImmutableList.of(ModuleFileValue.keyForRootModule()), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    RootModuleFileValue rootModuleFileValue = result.get(ModuleFileValue.keyForRootModule());
    assertThat(rootModuleFileValue.getModule())
        .isEqualTo(
            Module.builder()
                .setName("A")
                .setVersion(Version.parse("0.1"))
                .setCompatibilityLevel(4)
                .addDep("B", createModuleKey("B", "1.0"))
                .addDep("see", createModuleKey("C", "2.0"))
                .build());
    assertThat(rootModuleFileValue.getOverrides())
        .containsExactly(
            "D", SingleVersionOverride.create(Version.parse("18"), "", ImmutableList.of(), 0),
            "E", LocalPathOverride.create("somewhere/else"),
            "F",
                MultipleVersionOverride.create(
                    ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""),
            "G",
                ArchiveOverride.create(
                    ImmutableList.of("https://hello.com/world.zip"),
                    ImmutableList.of(),
                    "",
                    "",
                    0));
    assertThat(rootModuleFileValue.getNonRegistryOverrideCanonicalRepoNameLookup())
        .containsExactly("E.", "E", "G.", "G");
  }

  @Test
  public void testRootModule_noModuleFunctionIsOkay() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry = registryFactory.newFakeRegistry();
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<RootModuleFileValue> result =
        driver.evaluate(ImmutableList.of(ModuleFileValue.keyForRootModule()), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    RootModuleFileValue rootModuleFileValue = result.get(ModuleFileValue.keyForRootModule());
    assertThat(rootModuleFileValue.getModule())
        .isEqualTo(Module.builder().addDep("B", createModuleKey("B", "1.0")).build());
    assertThat(rootModuleFileValue.getOverrides()).isEmpty();
    assertThat(rootModuleFileValue.getNonRegistryOverrideCanonicalRepoNameLookup()).isEmpty();
  }

  @Test
  public void testRootModule_badSelfOverride() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='A')",
        "single_version_override(module_name='A',version='7')");
    FakeRegistry registry = registryFactory.newFakeRegistry();
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<RootModuleFileValue> result =
        driver.evaluate(ImmutableList.of(ModuleFileValue.keyForRootModule()), evaluationContext);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString()).contains("invalid override for the root module");
  }

  @Test
  public void testRegistriesCascade() throws Exception {
    // Registry1 has no module B@1.0; registry2 and registry3 both have it. We should be using the
    // B@1.0 from registry2.
    FakeRegistry registry1 = registryFactory.newFakeRegistry();
    FakeRegistry registry2 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B',version='1.0');bazel_dep(name='C',version='2.0')");
    FakeRegistry registry3 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B',version='1.0');bazel_dep(name='D',version='3.0')");
    ModuleFileFunction.REGISTRIES.set(
        differencer, ImmutableList.of(registry1.getUrl(), registry2.getUrl(), registry3.getUrl()));

    SkyKey skyKey = ModuleFileValue.key(createModuleKey("B", "1.0"), null);
    EvaluationResult<ModuleFileValue> result =
        driver.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    ModuleFileValue moduleFileValue = result.get(skyKey);
    assertThat(moduleFileValue.getModule())
        .isEqualTo(
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .setRegistry(registry2)
                .build());
  }

  @Test
  public void testLocalPathOverride() throws Exception {
    // There is an override for B to use the local path "code_for_b", so we shouldn't even be
    // looking at the registry.
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "local_path_override(module_name='B',path='code_for_b')");
    scratch.file(
        rootDirectory.getRelative("code_for_b/MODULE.bazel").getPathString(),
        "module(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0')");
    scratch.file(rootDirectory.getRelative("code_for_b/WORKSPACE").getPathString());
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry()
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B',version='1.0');bazel_dep(name='C',version='3.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    // The version is empty here due to the override.
    SkyKey skyKey =
        ModuleFileValue.key(createModuleKey("B", ""), LocalPathOverride.create("code_for_b"));
    EvaluationResult<ModuleFileValue> result =
        driver.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    ModuleFileValue moduleFileValue = result.get(skyKey);
    assertThat(moduleFileValue.getModule())
        .isEqualTo(
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .build());
  }

  @Test
  public void testRegistryOverride() throws Exception {
    FakeRegistry registry1 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B',version='1.0',compatibility_level=4)\n"
                    + "bazel_dep(name='C',version='2.0')");
    FakeRegistry registry2 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B',version='1.0',compatibility_level=6)\n"
                    + "bazel_dep(name='C',version='3.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry1.getUrl()));

    // Override the registry for B to be registry2 (instead of the default registry1).
    SkyKey skyKey =
        ModuleFileValue.key(
            createModuleKey("B", "1.0"),
            SingleVersionOverride.create(Version.EMPTY, registry2.getUrl(), ImmutableList.of(), 0));
    EvaluationResult<ModuleFileValue> result =
        driver.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    ModuleFileValue moduleFileValue = result.get(skyKey);
    assertThat(moduleFileValue.getModule())
        .isEqualTo(
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .setCompatibilityLevel(6)
                .addDep("C", createModuleKey("C", "3.0"))
                .setRegistry(registry2)
                .build());
  }
}
