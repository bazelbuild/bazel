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
import com.google.devtools.build.lib.vfs.Path;
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
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DiscoveryFunction}. */
@RunWith(JUnit4.class)
public class DiscoveryFunctionTest extends FoundationTestCase {

  private Path workspaceRoot;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setup() throws Exception {
    workspaceRoot = scratch.dir("/ws");
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
                .put(SkyFunctions.DISCOVERY, new DiscoveryFunction())
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(registryFactory, workspaceRoot))
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
  public void testSimpleDiamond() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='D',version='3.0')")
            .addModule(
                createModuleKey("C", "2.0"),
                "module(name='C', version='2.0');bazel_dep(name='D',version='3.0')")
            .addModule(createModuleKey("D", "3.0"), "module(name='D', version='3.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.parse("0.1"))
                .addDep("B", createModuleKey("B", "1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("D", createModuleKey("D", "3.0"))
                .setRegistry(registry)
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .addDep("D", createModuleKey("D", "3.0"))
                .setRegistry(registry)
                .build(),
            createModuleKey("D", "3.0"),
            Module.builder()
                .setName("D")
                .setVersion(Version.parse("3.0"))
                .setRegistry(registry)
                .build());
  }

  @Test
  public void testCircularDependency() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='C',version='2.0')")
            .addModule(
                createModuleKey("C", "2.0"),
                "module(name='C', version='2.0');bazel_dep(name='B',version='1.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.parse("0.1"))
                .addDep("B", createModuleKey("B", "1.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .setRegistry(registry)
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .addDep("B", createModuleKey("B", "1.0"))
                .setRegistry(registry)
                .build());
  }

  @Test
  public void testCircularDependencyOnRootModule() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='A',version='2.0')")
            .addModule(createModuleKey("A", "2.0"), "module(name='A', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.parse("0.1"))
                .addDep("B", createModuleKey("B", "1.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("A", createModuleKey("A", ""))
                .setRegistry(registry)
                .build());
  }

  @Test
  public void testSingleVersionOverride() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='0.1')",
        "single_version_override(module_name='C',version='2.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("B", "0.1"),
                "module(name='B', version='0.1');bazel_dep(name='C',version='1.0')")
            .addModule(createModuleKey("C", "1.0"), "module(name='C', version='1.0');")
            .addModule(createModuleKey("C", "2.0"), "module(name='C', version='2.0');");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.parse("0.1"))
                .addDep("B", createModuleKey("B", "0.1"))
                .build(),
            createModuleKey("B", "0.1"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("0.1"))
                .addDep("C", createModuleKey("C", "2.0"))
                .setRegistry(registry)
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .setRegistry(registry)
                .build());
  }

  @Test
  public void testRegistryOverride() throws Exception {
    FakeRegistry registry1 =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("B", "0.1"),
                "module(name='B', version='0.1');bazel_dep(name='C',version='1.0')")
            .addModule(createModuleKey("C", "1.0"), "module(name='C', version='1.0');");
    FakeRegistry registry2 =
        registryFactory
            .newFakeRegistry("/bar")
            .addModule(
                createModuleKey("C", "1.0"),
                "module(name='C', version='1.0');bazel_dep(name='B',version='0.1')");
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='0.1')",
        "single_version_override(module_name='C',registry='" + registry2.getUrl() + "')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry1.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.parse("0.1"))
                .addDep("B", createModuleKey("B", "0.1"))
                .build(),
            createModuleKey("B", "0.1"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("0.1"))
                .addDep("C", createModuleKey("C", "1.0"))
                .setRegistry(registry1)
                .build(),
            createModuleKey("C", "1.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("1.0"))
                .addDep("B", createModuleKey("B", "0.1"))
                .setRegistry(registry2)
                .build());
  }

  @Test
  public void testLocalPathOverride() throws Exception {
    Path pathToC = scratch.dir("/pathToC");
    scratch.file(
        pathToC.getRelative("MODULE.bazel").getPathString(), "module(name='C',version='2.0')");
    scratch.file(pathToC.getRelative("WORKSPACE").getPathString());
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='0.1')",
        "local_path_override(module_name='C',path='" + pathToC.getPathString() + "')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("B", "0.1"),
                "module(name='B', version='0.1');bazel_dep(name='C',version='1.0')")
            .addModule(createModuleKey("C", "1.0"), "module(name='C', version='1.0');");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.parse("0.1"))
                    .addDep("B", createModuleKey("B", "0.1"))
                    .build(),
            createModuleKey("B", "0.1"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("0.1"))
                    .addDep("C", createModuleKey("C", ""))
                    .setRegistry(registry)
                    .build(),
            createModuleKey("C", ""),
                Module.builder().setName("C").setVersion(Version.parse("2.0")).build());
  }
}
