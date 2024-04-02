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
package com.google.devtools.build.lib.analysis.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionRepoMappingEntriesFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpecFunction;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionEvalFunction;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionUsagesFunction;
import com.google.devtools.build.lib.bazel.bzlmod.YankedVersionsUtil;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.packages.util.LoadingMock;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.MockPythonSupport;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.packages.PackageFactoryBuilderWithSkyframeForTesting;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.mockito.Mockito;

/** Create a mock client for the analysis phase, as well as a configuration factory. */
public abstract class AnalysisMock extends LoadingMock {

  public static AnalysisMock get() {
    try {
      Class<?> providerClass = Class.forName(TestConstants.TEST_ANALYSIS_MOCK);
      Field instanceField = providerClass.getField("INSTANCE");
      return (AnalysisMock) instanceField.get(null);
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  public static AnalysisMock getAnalysisMockWithoutBuiltinModules() {
    return new AnalysisMock.Delegate(AnalysisMock.get()) {
      @Override
      public ImmutableMap<String, NonRegistryOverride> getBuiltinModules(
          BlazeDirectories directories) {
        return ImmutableMap.of();
      }
    };
  }

  @Override
  public String getProductName() {
    return TestConstants.PRODUCT_NAME;
  }

  public ImmutableList<String> getEmbeddedTools() {
    return TestConstants.EMBEDDED_TOOLS;
  }

  @Override
  public PackageFactoryBuilderWithSkyframeForTesting getPackageFactoryBuilderForTesting(
      BlazeDirectories directories) {
    return super.getPackageFactoryBuilderForTesting(directories)
        .setExtraSkyFunctions(getSkyFunctions(directories))
        .setExtraPrecomputeValues(getPrecomputedValues());
  }

  /**
   * This is called from test setup to create the mock directory layout needed to create the
   * configuration.
   */
  public void setupMockClient(MockToolsConfig mockToolsConfig) throws IOException {
    ImmutableList<String> workspaceContents = getWorkspaceContents(mockToolsConfig);
    setupMockClient(mockToolsConfig, workspaceContents);
  }

  public abstract void setupMockClient(
      MockToolsConfig mockToolsConfig, List<String> getWorkspaceContents) throws IOException;

  /** Returns the contents of WORKSPACE. */
  public abstract ImmutableList<String> getWorkspaceContents(MockToolsConfig config);

  /** Returns the repos defined in the contents of WORKSPACE above. */
  public abstract ImmutableList<String> getWorkspaceRepos();

  /**
   * This is called from test setup to create any necessary mock workspace files in the <code>
   * _embedded_binaries</code> directory.
   */
  public abstract void setupMockWorkspaceFiles(Path embeddedBinariesRoot) throws IOException;

  /** Creates a mock tools repository. */
  public void setupMockToolsRepository(MockToolsConfig config) throws IOException {
    // Do nothing by default.
  }

  @Override
  public abstract ConfiguredRuleClassProvider createRuleClassProvider();

  public abstract boolean isThisBazel();

  public abstract MockCcSupport ccSupport();

  public abstract MockPythonSupport pySupport();

  public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(BlazeDirectories directories) {
    // Some tests require the local_repository rule so we need the appropriate SkyFunctions.
    ImmutableMap.Builder<String, RepositoryFunction> repositoryHandlers =
        new ImmutableMap.Builder<String, RepositoryFunction>()
            .put(LocalRepositoryRule.NAME, new LocalRepositoryFunction())
            .put(AndroidSdkRepositoryRule.NAME, new AndroidSdkRepositoryFunction())
            .put(AndroidNdkRepositoryRule.NAME, new AndroidNdkRepositoryFunction());

    addExtraRepositoryFunctions(repositoryHandlers);

    DownloadManager downloadManager = Mockito.mock(DownloadManager.class);

    return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
        .put(
            SkyFunctions.REPOSITORY_DIRECTORY,
            new RepositoryDelegatorFunction(
                repositoryHandlers.buildKeepingLast(),
                new StarlarkRepositoryFunction(downloadManager),
                new AtomicBoolean(true),
                ImmutableMap::of,
                directories,
                BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
        .put(
            SkyFunctions.MODULE_FILE,
            new ModuleFileFunction(
                createRuleClassProvider().getBazelStarlarkEnvironment(),
                FakeRegistry.DEFAULT_FACTORY,
                directories.getWorkspace(),
                getBuiltinModules(directories)))
        .put(SkyFunctions.BAZEL_DEP_GRAPH, new BazelDepGraphFunction())
        .put(SkyFunctions.BAZEL_LOCK_FILE, new BazelLockFileFunction(directories.getWorkspace()))
        .put(SkyFunctions.BAZEL_MODULE_RESOLUTION, new BazelModuleResolutionFunction())
        .put(
            SkyFunctions.SINGLE_EXTENSION_EVAL,
            new SingleExtensionEvalFunction(directories, ImmutableMap::of, downloadManager))
        .put(SkyFunctions.SINGLE_EXTENSION_USAGES, new SingleExtensionUsagesFunction())
        .put(SkyFunctions.REPO_SPEC, new RepoSpecFunction(FakeRegistry.DEFAULT_FACTORY))
        .put(
            SkyFunctions.MODULE_EXTENSION_REPO_MAPPING_ENTRIES,
            new ModuleExtensionRepoMappingEntriesFunction())
        .put(
            SkyFunctions.CLIENT_ENVIRONMENT_VARIABLE,
            new ClientEnvironmentFunction(new AtomicReference<>(ImmutableMap.of())))
        .buildOrThrow();
  }

  public ImmutableList<PrecomputedValue.Injected> getPrecomputedValues() {
    // PrecomputedValues required by SkyFunctions in getSkyFunctions()
    return ImmutableList.of(
        PrecomputedValue.injected(PrecomputedValue.REPO_ENV, ImmutableMap.of()),
        PrecomputedValue.injected(ModuleFileFunction.MODULE_OVERRIDES, ImmutableMap.of()),
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.REPOSITORY_OVERRIDES, ImmutableMap.of()),
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty()),
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.FORCE_FETCH,
            RepositoryDelegatorFunction.FORCE_FETCH_DISABLED),
        PrecomputedValue.injected(RepositoryDelegatorFunction.VENDOR_DIRECTORY, Optional.empty()),
        PrecomputedValue.injected(ModuleFileFunction.REGISTRIES, ImmutableList.of()),
        PrecomputedValue.injected(ModuleFileFunction.IGNORE_DEV_DEPS, false),
        PrecomputedValue.injected(ModuleFileFunction.MODULE_OVERRIDES, ImmutableMap.of()),
        PrecomputedValue.injected(YankedVersionsUtil.ALLOWED_YANKED_VERSIONS, ImmutableList.of()),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, CheckDirectDepsMode.WARNING),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE, BazelCompatibilityMode.ERROR),
        PrecomputedValue.injected(BazelLockFileFunction.LOCKFILE_MODE, LockfileMode.UPDATE));
  }

  // Allow subclasses to add extra repository functions.
  public abstract void addExtraRepositoryFunctions(
      ImmutableMap.Builder<String, RepositoryFunction> repositoryHandlers);

  /** Returns the built-in modules. */
  public abstract ImmutableMap<String, NonRegistryOverride> getBuiltinModules(
      BlazeDirectories directories);

  /**
   * Stub class for tests to extend in order to update a small amount of {@link AnalysisMock}
   * functionality.
   */
  public static class Delegate extends AnalysisMock {

    private final AnalysisMock delegate;

    public Delegate(AnalysisMock delegate) {
      this.delegate = delegate;
    }

    @Override
    public void setupMockClient(MockToolsConfig mockToolsConfig, List<String> workspaceContents)
        throws IOException {
      delegate.setupMockClient(mockToolsConfig, workspaceContents);
    }

    @Override
    public ImmutableList<String> getWorkspaceContents(MockToolsConfig mockToolsConfig) {
      return delegate.getWorkspaceContents(mockToolsConfig);
    }

    @Override
    public ImmutableList<String> getWorkspaceRepos() {
      return delegate.getWorkspaceRepos();
    }

    @Override
    public void setupMockWorkspaceFiles(Path embeddedBinariesRoot) throws IOException {
      delegate.setupMockWorkspaceFiles(embeddedBinariesRoot);
    }

    @Override
    public void setupMockToolsRepository(MockToolsConfig config) throws IOException {
      delegate.setupMockToolsRepository(config);
    }

    @Override
    public ConfiguredRuleClassProvider createRuleClassProvider() {
      return delegate.createRuleClassProvider();
    }

    @Override
    public boolean isThisBazel() {
      return delegate.isThisBazel();
    }

    @Override
    public MockCcSupport ccSupport() {
      return delegate.ccSupport();
    }

    @Override
    public MockPythonSupport pySupport() {
      return delegate.pySupport();
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(
              Maps.filterKeys(
                  super.getSkyFunctions(directories),
                  fnName -> !fnName.equals(SkyFunctions.MODULE_FILE)))
          .put(
              SkyFunctions.MODULE_FILE,
              new ModuleFileFunction(
                  createRuleClassProvider().getBazelStarlarkEnvironment(),
                  FakeRegistry.DEFAULT_FACTORY,
                  directories.getWorkspace(),
                  getBuiltinModules(directories)))
          .buildOrThrow();
    }

    @Override
    public ImmutableMap<String, NonRegistryOverride> getBuiltinModules(
        BlazeDirectories directories) {
      return delegate.getBuiltinModules(directories);
    }

    @Override
    public void addExtraRepositoryFunctions(
        ImmutableMap.Builder<String, RepositoryFunction> repositoryHandlers) {
      delegate.addExtraRepositoryFunctions(repositoryHandlers);
    }
  }
}
