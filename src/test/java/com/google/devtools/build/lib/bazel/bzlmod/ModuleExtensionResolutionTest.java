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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BzlCompileFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.BzlmodRepoRuleFunction;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.ExternalPackageFunction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.IgnoredPackagePrefixesFunction;
import com.google.devtools.build.lib.skyframe.LocalRepositoryLookupFunction;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingFunction;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsFunction;
import com.google.devtools.build.lib.skyframe.WorkspaceFileFunction;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
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
import org.mockito.Mockito;

/** Tests for module extension resolution. */
@RunWith(JUnit4.class)
public class ModuleExtensionResolutionTest extends FoundationTestCase {

  private Path workspaceRoot;
  private Path modulesRoot;
  private MemoizingEvaluator evaluator;
  private EvaluationContext evaluationContext;
  private FakeRegistry registry;
  private RecordingDifferencer differencer;

  @Before
  public void setup() throws Exception {
    workspaceRoot = scratch.dir("/ws");
    modulesRoot = scratch.dir("/modules");
    differencer = new SequencedRecordingDifferencer();
    evaluationContext =
        EvaluationContext.newBuilder().setNumThreads(8).setEventHandler(reporter).build();
    FakeRegistry.Factory registryFactory = new FakeRegistry.Factory();
    registry = registryFactory.newFakeRegistry(modulesRoot.getPathString());
    AtomicReference<PathPackageLocator> packageLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(workspaceRoot)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, rootDirectory),
            workspaceRoot,
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
    HashFunction hashFunction = fileSystem.getDigestFunction().getHashFunction();

    DownloadManager downloadManager = Mockito.mock(DownloadManager.class);
    SingleExtensionEvalFunction singleExtensionEvalFunction =
        new SingleExtensionEvalFunction(
            packageFactory, directories, ImmutableMap::of, downloadManager);
    StarlarkRepositoryFunction starlarkRepositoryFunction =
        new StarlarkRepositoryFunction(downloadManager);

    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(LocalRepositoryRule.NAME, new LocalRepositoryFunction());
    evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(FileValue.FILE, new FileFunction(packageLocator))
                .put(
                    FileStateValue.FILE_STATE,
                    new FileStateFunction(
                        new AtomicReference<>(),
                        new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS),
                        externalFilesHelper))
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(registryFactory, workspaceRoot))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .put(SkyFunctions.BZL_COMPILE, new BzlCompileFunction(packageFactory, hashFunction))
                .put(
                    SkyFunctions.BZL_LOAD,
                    BzlLoadFunction.create(
                        packageFactory, directories, hashFunction, Caffeine.newBuilder().build()))
                .put(SkyFunctions.STARLARK_BUILTINS, new StarlarkBuiltinsFunction(packageFactory))
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
                .put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction())
                .put(
                    SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
                    new LocalRepositoryLookupFunction(
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(
                    SkyFunctions.IGNORED_PACKAGE_PREFIXES,
                    new IgnoredPackagePrefixesFunction(
                        /*ignoredPackagePrefixesFile=*/ PathFragment.EMPTY_FRAGMENT))
                .put(SkyFunctions.RESOLVED_HASH_VALUES, new ResolvedHashesFunction())
                .put(SkyFunctions.REPOSITORY_MAPPING, new RepositoryMappingFunction())
                .put(
                    SkyFunctions.EXTERNAL_PACKAGE,
                    new ExternalPackageFunction(
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(
                    WorkspaceFileValue.WORKSPACE_FILE,
                    new WorkspaceFileFunction(
                        ruleClassProvider,
                        packageFactory,
                        directories,
                        /*bzlLoadFunctionForInlining=*/ null))
                .put(
                    SkyFunctions.REPOSITORY_DIRECTORY,
                    new RepositoryDelegatorFunction(
                        repositoryHandlers,
                        starlarkRepositoryFunction,
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
                .put(SkyFunctions.BAZEL_MODULE_RESOLUTION, new BazelModuleResolutionFunction())
                .put(SkyFunctions.SINGLE_EXTENSION_USAGES, new SingleExtensionUsagesFunction())
                .put(SkyFunctions.SINGLE_EXTENSION_EVAL, singleExtensionEvalFunction)
                .put(
                    SkyFunctions.MODULE_EXTENSION_RESOLUTION,
                    new ModuleExtensionResolutionFunction())
                .build(),
            differencer);

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
    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, false);
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.WARNING);

    // Set up a simple repo rule.
    registry.addModule(
        createModuleKey("data_repo", "1.0"), "module(name='data_repo',version='1.0')");
    scratch.file(modulesRoot.getRelative("data_repo.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("data_repo.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("data_repo.1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', 'data = '+json.encode(ctx.attr.data))",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl,",
        "  attrs={'data':attr.string()})");
  }

  @Test
  public void simpleExtension() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "ext.tag(name='foo', data='fu')",
        "ext.tag(name='bar', data='ba')",
        "use_repo(ext, 'foo', 'bar')");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {'name':attr.string(),'data':attr.string()})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_repo(name=tag.name,data=tag.data)",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@bar//:data.bzl', bar_data='data')",
        "data = 'foo:'+foo_data+' bar:'+bar_data");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("foo:fu bar:ba");
  }

  @Test
  public void multipleModules() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='root')",
        "use_repo(ext,'ext_repo')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "data=ext_data");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='quux',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='foo@1.0')");
    registry.addModule(
        createModuleKey("bar", "2.0"),
        "module(name='bar',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='quux',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='bar@2.0')");
    registry.addModule(
        createModuleKey("quux", "1.0"),
        "module(name='quux',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='quux@1.0')");
    registry.addModule(
        createModuleKey("quux", "2.0"),
        "module(name='quux',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='quux@2.0')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(modulesRoot.getRelative("ext.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("ext.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("ext.1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = 'modules:'",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + tag.data",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'data':attr.string()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("modules: root foo@1.0 bar@2.0 quux@2.0");
  }

  @Test
  public void multipleModules_devDependency() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='root')",
        "use_repo(ext,'ext_repo')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "data=ext_data");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='foo@1.0')");
    registry.addModule(
        createModuleKey("bar", "2.0"),
        "module(name='bar',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='bar@2.0')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(modulesRoot.getRelative("ext.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("ext.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("ext.1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = 'modules:'",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + tag.data",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'data':attr.string()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("modules: root bar@2.0");
  }

  @Test
  public void multipleModules_ignoreDevDependency() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='root')",
        "use_repo(ext,'ext_repo')");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext',dev_dependency=True)",
        "ext.tag(data='foo@1.0')");
    registry.addModule(
        createModuleKey("bar", "2.0"),
        "module(name='bar',version='2.0')",
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(data='bar@2.0')");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(modulesRoot.getRelative("ext.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("ext.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("ext.1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = 'modules:'",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + tag.data",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'data':attr.string()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, true);

    SkyKey skyKey =
        BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("@ext.1.0.ext.ext_repo//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("modules: bar@2.0");
  }

  @Test
  public void labels_readInModuleExtension() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='//:requirements.txt')",
        "use_repo(ext,'ext_repo')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "data=ext_data");
    scratch.file(workspaceRoot.getRelative("requirements.txt").getPathString(), "get up at 6am.");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='@bar//:requirements.txt')");
    registry.addModule(createModuleKey("bar", "2.0"), "module(name='bar',version='2.0')");
    scratch.file(modulesRoot.getRelative("bar.2.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("bar.2.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("bar.2.0/requirements.txt").getPathString(), "go to bed at 11pm.");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(modulesRoot.getRelative("ext.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("ext.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("ext.1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  data_str = 'requirements:'",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + ctx.read(tag.file).strip()",
        "  data_repo(name='ext_repo',data=data_str)",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("requirements: get up at 6am. go to bed at 11pm.");
  }

  @Test
  public void labels_passedOnToRepoRule() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='//:requirements.txt')",
        "use_repo(ext,'ext_repo')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "data=ext_data");
    scratch.file(workspaceRoot.getRelative("requirements.txt").getPathString(), "get up at 6am.");

    registry.addModule(
        createModuleKey("foo", "1.0"),
        "module(name='foo',version='1.0')",
        "bazel_dep(name='ext',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='@bar//:requirements.txt')");
    registry.addModule(createModuleKey("bar", "2.0"), "module(name='bar',version='2.0')");
    scratch.file(modulesRoot.getRelative("bar.2.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("bar.2.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("bar.2.0/requirements.txt").getPathString(), "go to bed at 11pm.");

    registry.addModule(createModuleKey("ext", "1.0"), "module(name='ext',version='1.0')");
    scratch.file(modulesRoot.getRelative("ext.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("ext.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("ext.1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  content = ' '.join([ctx.read(l).strip() for l in ctx.attr.files])",
        "  ctx.file('data.bzl', 'data='+json.encode(content))",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl, attrs={'files':attr.label_list()})",
        "",
        "def _ext_impl(ctx):",
        "  data_files = []",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_files.append(tag.file)",
        "  data_repo(name='ext_repo',files=data_files)",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("get up at 6am. go to bed at 11pm.");
  }

  @Test
  public void labels_fromExtensionGeneratedRepo() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='ext',version='1.0')",
        "myext = use_extension('//:defs.bzl','myext')",
        "use_repo(myext,'myrepo')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag(file='@myrepo//:requirements.txt')",
        "use_repo(ext,'ext_repo')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "data=ext_data");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "def _myrepo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  ctx.file('requirements.txt', 'get up at 6am.')",
        "myrepo = repository_rule(implementation=_myrepo_impl)",
        "",
        "def _myext_impl(ctx):",
        "  myrepo(name='myrepo')",
        "myext=module_extension(implementation=_myext_impl)");
    scratch.file(workspaceRoot.getRelative("requirements.txt").getPathString(), "get up at 6am.");

    registry.addModule(createModuleKey("ext", "1.0"), "module(name='ext',version='1.0')");
    scratch.file(modulesRoot.getRelative("ext.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("ext.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("ext.1.0/defs.bzl").getPathString(),
        "def _data_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  content = ' '.join([ctx.read(l).strip() for l in ctx.attr.files])",
        "  ctx.file('data.bzl', 'data='+json.encode(content))",
        "data_repo = repository_rule(",
        "  implementation=_data_repo_impl, attrs={'files':attr.label_list()})",
        "",
        "def _ext_impl(ctx):",
        "  data_files = []",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_files.append(tag.file)",
        "  data_repo(name='ext_repo',files=data_files)",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("get up at 6am.");
  }

  @Test
  public void labels_constructedInModuleExtension() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='ext',version='1.0')",
        "ext = use_extension('@ext//:defs.bzl','ext')",
        "ext.tag()",
        "use_repo(ext,'ext_repo')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext_repo//:data.bzl', ext_data='data')",
        "data=ext_data");

    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.file(modulesRoot.getRelative("foo.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("foo.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("foo.1.0/requirements.txt").getPathString(), "get up at 6am.");
    registry.addModule(createModuleKey("bar", "2.0"), "module(name='bar',version='2.0')");
    scratch.file(modulesRoot.getRelative("bar.2.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("bar.2.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("bar.2.0/requirements.txt").getPathString(), "go to bed at 11pm.");

    registry.addModule(
        createModuleKey("ext", "1.0"),
        "module(name='ext',version='1.0')",
        "bazel_dep(name='foo',version='1.0')",
        "bazel_dep(name='bar',version='2.0')",
        "bazel_dep(name='data_repo',version='1.0')");
    scratch.file(modulesRoot.getRelative("ext.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("ext.1.0/BUILD").getPathString());
    scratch.file(
        modulesRoot.getRelative("ext.1.0/defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        // The Label() call on the following line should work, using ext.1.0's repo mapping.
        "  data_str = 'requirements: ' + ctx.read(Label('@foo//:requirements.txt')).strip()",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      data_str += ' ' + ctx.read(tag.file).strip()",
        "  data_repo(name='ext_repo',data=data_str)",
        // So should the attr.label default value on the following line.
        "tag=tag_class(attrs={'file':attr.label(default='@bar//:requirements.txt')})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("requirements: get up at 6am. go to bed at 11pm.");
  }

  /** Tests that a complex-typed attribute (here, string_list_dict) behaves well on a tag. */
  @Test
  public void complexTypedAttribute() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "ext.tag(data={'foo':['val1','val2'],'bar':['val3','val4']})",
        "use_repo(ext, 'foo', 'bar')");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {'data':attr.string_list_dict()})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      for key in tag.data:",
        "        data_repo(name=key,data=','.join(tag.data[key]))",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@bar//:data.bzl', bar_data='data')",
        "data = 'foo:'+foo_data+' bar:'+bar_data");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("foo:val1,val2 bar:val3,val4");
  }

  /**
   * Tests that a complex-typed attribute (here, string_list_dict) behaves well when it has a
   * default value and is omitted in a tag.
   */
  @Test
  public void complexTypedAttribute_default() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "ext.tag()",
        "use_repo(ext, 'foo', 'bar')");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "tag = tag_class(attrs = {",
        "  'data': attr.string_list_dict(",
        "    default = {'foo':['val1','val2'],'bar':['val3','val4']},",
        ")})",
        "def _ext_impl(ctx):",
        "  for mod in ctx.modules:",
        "    for tag in mod.tags.tag:",
        "      for key in tag.data:",
        "        data_repo(name=key,data=','.join(tag.data[key]))",
        "ext = module_extension(implementation=_ext_impl, tag_classes={'tag':tag})");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@foo//:data.bzl', foo_data='data')",
        "load('@bar//:data.bzl', bar_data='data')",
        "data = 'foo:'+foo_data+' bar:'+bar_data");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("foo:val1,val2 bar:val3,val4");
  }

  @Test
  public void generatedReposHaveCorrectMappings() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('//:defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext//:data.bzl', ext_data='data')",
        "data=ext_data");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "def _ext_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', \"\"\"load('@foo//:data.bzl', foo_data='data')",
        "load('@_internal//:data.bzl', internal_data='data')",
        "data = 'foo: '+foo_data+' internal: '+internal_data",
        "\"\"\")",
        "ext_repo = repository_rule(implementation=_ext_repo_impl)",
        "",
        "def _internal_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', 'data='+json.encode('internal-stuff'))",
        "internal_repo = repository_rule(implementation=_internal_repo_impl)",
        "",
        "def _ext_impl(ctx):",
        "  internal_repo(name='_internal')",
        "  ext_repo(name='ext')",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.file(modulesRoot.getRelative("foo.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("foo.1.0/BUILD").getPathString());
    scratch.file(modulesRoot.getRelative("foo.1.0/data.bzl").getPathString(), "data = 'foo-stuff'");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("foo: foo-stuff internal: internal-stuff");
  }

  @Test
  public void generatedReposHaveCorrectMappings_internalRepoWins() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='foo',version='1.0')",
        "ext = use_extension('//:defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext//:data.bzl', ext_data='data')",
        "data=ext_data");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "def _ext_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', \"\"\"load('@foo//:data.bzl', foo_data='data')",
        "data = 'the foo I see is '+foo_data",
        "\"\"\")",
        "ext_repo = repository_rule(implementation=_ext_repo_impl)",
        "",
        "def _internal_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', 'data='+json.encode('inner-foo'))",
        "internal_repo = repository_rule(implementation=_internal_repo_impl)",
        "",
        "def _ext_impl(ctx):",
        "  internal_repo(name='foo')",
        "  ext_repo(name='ext')",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.file(modulesRoot.getRelative("foo.1.0/WORKSPACE").getPathString());
    scratch.file(modulesRoot.getRelative("foo.1.0/BUILD").getPathString());
    scratch.file(modulesRoot.getRelative("foo.1.0/data.bzl").getPathString(), "data = 'outer-foo'");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data"))
        .isEqualTo("the foo I see is inner-foo");
  }

  @Test
  public void generatedReposHaveCorrectMappings_strictDepsViolation() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "ext = use_extension('//:defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext//:data.bzl', ext_data='data')",
        "data=ext_data");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "def _ext_repo_impl(ctx):",
        "  ctx.file('WORKSPACE')",
        "  ctx.file('BUILD')",
        "  ctx.file('data.bzl', \"\"\"load('@foo//:data.bzl', 'data')\"\"\")",
        "ext_repo = repository_rule(implementation=_ext_repo_impl)",
        "",
        "def _ext_impl(ctx):",
        "  ext_repo(name='ext')",
        "tag=tag_class(attrs={'file':attr.label()})",
        "ext=module_extension(implementation=_ext_impl,tag_classes={'tag':tag})");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains("Repository '@foo' is not visible from repository '@.ext.ext'");
  }

  @Test
  public void wrongModuleExtensionLabel() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "ext = use_extension('//foo/defs.bzl','ext')",
        "use_repo(ext,'ext')");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext//:data.bzl', ext_data='data')",
        "data=ext_data");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains(
            "Label '//foo/defs.bzl:defs.bzl' is invalid because 'foo/defs.bzl' is not a package");
  }

  @Test
  public void nativeExistingRuleIsEmpty() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "bazel_dep(name='data_repo', version='1.0')",
        "ext = use_extension('//:defs.bzl', 'ext')",
        "use_repo(ext, 'ext')");
    scratch.file(
        workspaceRoot.getRelative("defs.bzl").getPathString(),
        "load('@data_repo//:defs.bzl','data_repo')",
        "def _ext_impl(ctx):",
        "  if not native.existing_rules():",
        "    data_repo(name='ext',data='haha')",
        "ext = module_extension(implementation=_ext_impl)");
    scratch.file(workspaceRoot.getRelative("BUILD").getPathString());
    scratch.file(
        workspaceRoot.getRelative("data.bzl").getPathString(),
        "load('@ext//:data.bzl', ext_data='data')",
        "data = ext_data");

    SkyKey skyKey = BzlLoadValue.keyForBuild(Label.parseAbsoluteUnchecked("//:data.bzl"));
    EvaluationResult<BzlLoadValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    assertThat(result.get(skyKey).getModule().getGlobal("data")).isEqualTo("haha");
  }
}
