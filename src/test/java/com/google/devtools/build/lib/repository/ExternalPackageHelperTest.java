// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.repository;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.ExternalPackageFunction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.LocalRepositoryLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RegisteredExecutionPlatformsFunction;
import com.google.devtools.build.lib.skyframe.RegisteredToolchainsFunction;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.WorkspaceASTFunction;
import com.google.devtools.build.lib.skyframe.WorkspaceFileFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ExternalPackageHelper}. */
@RunWith(JUnit4.class)
public class ExternalPackageHelperTest extends BuildViewTestCase {
  private static final Root MOCK_ROOT = mockRoot();

  private static Root mockRoot() {
    Root root = mock(Root.class);
    when(root.isAbsolute()).thenReturn(false);
    return root;
  }

  private static final EvaluationContext EVALUATION_OPTIONS =
      EvaluationContext.newBuilder()
          .setKeepGoing(false)
          .setNumThreads(SkyframeExecutor.DEFAULT_THREAD_COUNT)
          .setEventHandler(NullEventHandler.INSTANCE)
          .build();

  private SequentialBuildDriver driver;

  @Before
  public void createEnvironment() {
    AnalysisMock analysisMock = AnalysisMock.get();
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages =
        new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, rootDirectory),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());
    ExternalFilesHelper externalFilesHelper =
        ExternalFilesHelper.createForTesting(
            pkgLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            deletedPackages,
            CrossRepositoryLabelViolationStrategy.ERROR,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        FileStateValue.FILE_STATE,
        new FileStateFunction(
            new AtomicReference<TimestampGranularityMonitor>(),
            new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS),
            externalFilesHelper));
    skyFunctions.put(FileValue.FILE, new FileFunction(pkgLocator));
    RuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    skyFunctions.put(SkyFunctions.WORKSPACE_AST, new WorkspaceASTFunction(ruleClassProvider));
    skyFunctions.put(
        WorkspaceFileValue.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            analysisMock
                .getPackageFactoryBuilderForTesting(directories)
                .build(ruleClassProvider, fileSystem),
            directories,
            /*bzlLoadFunctionForInlining=*/ null));
    skyFunctions.put(
        SkyFunctions.PACKAGE,
        new PackageFunction(
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.EXTERNAL_PACKAGE,
        new ExternalPackageFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
        new LocalRepositoryLookupFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));

    // Helper Skyfunctions to call ExternalPackageUtil.
    skyFunctions.put(GET_RULE_BY_NAME_FUNCTION, new GetRuleByNameFunction());
    skyFunctions.put(
        GET_REGISTERED_EXECUTION_PLATFORMS_FUNCTION, new GetRegisteredExecutionPlatformsFunction());
    skyFunctions.put(GET_REGISTERED_TOOLCHAINS_FUNCTION, new GetRegisteredToolchainsFunction());

    RecordingDifferencer differencer = new SequencedRecordingDifferencer();
    MemoizingEvaluator evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.set(
        differencer, Optional.empty());
  }

  @Test
  public void getRuleByName() throws Exception {
    if (!analysisMock.isThisBazel()) {
      return;
    }
    scratch.overwriteFile("WORKSPACE", "local_repository(name = 'foo', path = 'path/to/repo')");

    SkyKey key = getRuleByNameKey("foo");
    EvaluationResult<GetRuleByNameValue> result = getRuleByName(key);

    assertThatEvaluationResult(result).hasNoError();

    Rule rule = result.get(key).rule();
    assertThat(rule).isNotNull();
    assertThat(rule.getName()).isEqualTo("foo");
  }

  EvaluationResult<GetRuleByNameValue> getRuleByName(SkyKey key) throws InterruptedException {
    return driver.<GetRuleByNameValue>evaluate(ImmutableList.of(key), EVALUATION_OPTIONS);
  }

  @Test
  public void getRuleByName_missing() throws Exception {
    if (!analysisMock.isThisBazel()) {
      return;
    }
    scratch.overwriteFile("WORKSPACE", "local_repository(name = 'foo', path = 'path/to/repo')");

    SkyKey key = getRuleByNameKey("bar");
    EvaluationResult<GetRuleByNameValue> result = getRuleByName(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("The rule named 'bar' could not be resolved");
  }

  @Test
  public void getRegisteredToolchains_addedInWorkspace() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE", "register_toolchains(", "  '//toolchain:tc1',", "  '//toolchain:tc2')");

    assertThat(getRegisteredToolchains())
        // There are default toolchains that are always registered, so just check for the ones added
        .containsAtLeast("//toolchain:tc1", "//toolchain:tc2")
        .inOrder();
  }

  @Test
  public void getRegisteredToolchains_workspaceBazelPreferred() throws Exception {
    scratch.overwriteFile("WORKSPACE", "register_toolchains('//toolchain:WORKSPACE')");
    scratch.overwriteFile("WORKSPACE.bazel", "register_toolchains('//toolchain:WORKSPACE_bazel')");

    getRegisteredToolchains();
    ImmutableList<String> registeredToolchains = getRegisteredToolchains();

    assertThat(registeredToolchains).contains("//toolchain:WORKSPACE_bazel");
    assertThat(registeredToolchains).doesNotContain("//toolchain:WORKSPACE");
  }

  @Test
  public void getRegisteredToolchains_workspaceBazelNotAFile() throws Exception {
    scratch.overwriteFile("WORKSPACE", "register_toolchains('//toolchain:WORKSPACE')");
    scratch.dir("WORKSPACE.bazel");

    getRegisteredToolchains();
    ImmutableList<String> registeredToolchains = getRegisteredToolchains();

    assertThat(registeredToolchains).contains("//toolchain:WORKSPACE");
  }

  @Test
  public void findWorkspaceFile_firstIsFile() throws Exception {
    ExternalPackageHelper helper =
        new ExternalPackageHelper(
            ImmutableList.of(BuildFileName.WORKSPACE, BuildFileName.WORKSPACE_DOT_BAZEL));
    Environment env = createMockEnvironment();
    mockFileValue(env, BuildFileName.WORKSPACE, /*isFile=*/ true);

    assertThat(helper.findWorkspaceFile(env)).isEqualTo(rootedPath(BuildFileName.WORKSPACE));
  }

  @Test
  public void findWorkspaceFile_firstNotAFileFallbackToSecond() throws Exception {
    ExternalPackageHelper helper =
        new ExternalPackageHelper(
            ImmutableList.of(BuildFileName.WORKSPACE_DOT_BAZEL, BuildFileName.WORKSPACE));
    Environment env = createMockEnvironment();
    mockFileValue(env, BuildFileName.WORKSPACE_DOT_BAZEL, /*isFile=*/ false);
    mockFileValue(env, BuildFileName.WORKSPACE, /*isFile=*/ true);

    assertThat(helper.findWorkspaceFile(env)).isEqualTo(rootedPath(BuildFileName.WORKSPACE));
  }

  @Test
  public void findWorkspaceFile_noneAreFilesReturnLast() throws Exception {
    ExternalPackageHelper helper =
        new ExternalPackageHelper(
            ImmutableList.of(BuildFileName.WORKSPACE_DOT_BAZEL, BuildFileName.WORKSPACE));
    Environment env = createMockEnvironment();
    mockFileValue(env, BuildFileName.WORKSPACE_DOT_BAZEL, /*isFile=*/ false);
    mockFileValue(env, BuildFileName.WORKSPACE, /*isFile=*/ false);

    assertThat(helper.findWorkspaceFile(env)).isEqualTo(rootedPath(BuildFileName.WORKSPACE));
  }

  @Test
  public void getRegisteredExecutionPlatforms() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE", "register_execution_platforms(", "  '//platform:ep1',", "  '//platform:ep2')");

    SkyKey key = () -> GET_REGISTERED_EXECUTION_PLATFORMS_FUNCTION;
    EvaluationResult<GetRegisteredExecutionPlatformsValue> result =
        getRegisteredExecutionPlatforms(key);

    assertThatEvaluationResult(result).hasNoError();

    assertThat(result.get(key).registeredExecutionPlatforms())
        .containsExactly("//platform:ep1", "//platform:ep2")
        .inOrder();
  }

  EvaluationResult<GetRegisteredExecutionPlatformsValue> getRegisteredExecutionPlatforms(SkyKey key)
      throws InterruptedException {
    return driver.<GetRegisteredExecutionPlatformsValue>evaluate(
        ImmutableList.of(key), EVALUATION_OPTIONS);
  }

  // HELPER SKYFUNCTIONS

  private static Environment createMockEnvironment() throws InterruptedException {
    Environment env = mock(Environment.class);
    when(env.getValue(PrecomputedValue.PATH_PACKAGE_LOCATOR.getKeyForTesting()))
        .thenReturn(
            new PrecomputedValue(
                new PathPackageLocator(
                    mock(Path.class), ImmutableList.of(MOCK_ROOT), ImmutableList.of())));
    return env;
  }

  private static void mockFileValue(Environment env, BuildFileName buildFileName, boolean isFile)
      throws InterruptedException {
    FileValue fileValue = mock(FileValue.class);
    when(fileValue.isFile()).thenReturn(isFile);
    when(env.getValue(FileValue.key(rootedPath(buildFileName)))).thenReturn(fileValue);
  }

  private static RootedPath rootedPath(BuildFileName buildFileName) {
    return RootedPath.toRootedPath(MOCK_ROOT, buildFileName.getFilenameFragment());
  }

  private ImmutableList<String> getRegisteredToolchains() throws InterruptedException {
    SkyKey key = getRegisteredToolchainsKey();
    EvaluationResult<GetRegisteredToolchainsValue> result =
        driver.evaluate(ImmutableList.of(key), EVALUATION_OPTIONS);

    assertThatEvaluationResult(result).hasNoError();
    return result.get(key).registeredToolchains();
  }

  // GetRuleByName.
  private static SkyKey getRuleByNameKey(String ruleName) {
    return new Key(ruleName);
  }

  private static final SkyFunctionName GET_RULE_BY_NAME_FUNCTION =
      SkyFunctionName.createHermetic("GET_RULE_BY_NAME");

  @AutoValue
  abstract static class GetRuleByNameValue implements SkyValue {
    abstract Rule rule();

    static GetRuleByNameValue create(Rule rule) {
      return new AutoValue_ExternalPackageHelperTest_GetRuleByNameValue(rule);
    }
  }

  private static final class GetRuleByNameFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      String ruleName = (String) skyKey.argument();

      Rule rule =
          BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER.getRuleByName(ruleName, env);
      if (rule == null) {
        return null;
      }
      return GetRuleByNameValue.create(rule);
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  // GetRegisteredToolchains.
  private static SkyKey getRegisteredToolchainsKey() {
    return () -> GET_REGISTERED_TOOLCHAINS_FUNCTION;
  }

  private static final SkyFunctionName GET_REGISTERED_TOOLCHAINS_FUNCTION =
      SkyFunctionName.createHermetic("GET_REGISTERED_TOOLCHAINS");

  @AutoValue
  abstract static class GetRegisteredToolchainsValue implements SkyValue {
    abstract ImmutableList<String> registeredToolchains();

    static GetRegisteredToolchainsValue create(Iterable<String> registeredToolchains) {
      return new AutoValue_ExternalPackageHelperTest_GetRegisteredToolchainsValue(
          ImmutableList.copyOf(registeredToolchains));
    }
  }

  private static final class GetRegisteredToolchainsFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      List<String> registeredToolchains = RegisteredToolchainsFunction.getRegisteredToolchains(env);
      if (registeredToolchains == null) {
        return null;
      }
      return GetRegisteredToolchainsValue.create(registeredToolchains);
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  private static final SkyFunctionName GET_REGISTERED_EXECUTION_PLATFORMS_FUNCTION =
      SkyFunctionName.createHermetic("GET_REGISTERED_EXECUTION_PLATFORMS_FUNCTION");

  @AutoValue
  abstract static class GetRegisteredExecutionPlatformsValue implements SkyValue {
    abstract ImmutableList<String> registeredExecutionPlatforms();

    static GetRegisteredExecutionPlatformsValue create(
        Iterable<String> registeredExecutionPlatforms) {
      return new AutoValue_ExternalPackageHelperTest_GetRegisteredExecutionPlatformsValue(
          ImmutableList.copyOf(registeredExecutionPlatforms));
    }
  }

  private static final class GetRegisteredExecutionPlatformsFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      List<String> registeredExecutionPlatforms =
          RegisteredExecutionPlatformsFunction.getWorkspaceExecutionPlatforms(env);
      if (registeredExecutionPlatforms == null) {
        return null;
      }
      return GetRegisteredExecutionPlatformsValue.create(registeredExecutionPlatforms);
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  static class Key extends AbstractSkyKey<String> {
    private Key(String arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return GET_RULE_BY_NAME_FUNCTION;
    }
  }
}
