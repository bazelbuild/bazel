// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildtool.util;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Throwables.throwIfUnchecked;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.eventbus.SubscriberExceptionContext;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil.DummyWorkspaceStatusActionContext;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.includescanning.IncludeScanningModule;
import com.google.devtools.build.lib.integration.util.IntegrationMock;
import com.google.devtools.build.lib.network.ConnectivityStatusProvider;
import com.google.devtools.build.lib.network.NoOpConnectivityModule;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.BlazeWorkspace;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestConstants.InternalTestExecutionMode;
import com.google.devtools.build.lib.testutil.TestFileOutErr;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.CommandUtils;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.After;
import org.junit.Before;

/**
 * A base class for integration tests that use the {@link BuildTool}. These tests basically run a
 * little build and check what happens.
 *
 * <p>All integration tests are at least size medium.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
public abstract class BuildIntegrationTestCase {

  /** Thrown when an integration test case fails. */
  public static class IntegrationTestExecException extends ExecException {
    public IntegrationTestExecException(String message) {
      super(message);
    }

    public IntegrationTestExecException(String message, Throwable cause) {
      super(message, cause);
    }

    @Override
    public ActionExecutionException toActionExecutionException(
        String messagePrefix, boolean verboseFailures, Action action) {
      String message = messagePrefix + getMessage();
      // Append cause.getMessage() if it's different from getMessage(). It typically
      // isn't but if it is we'd like to surface cause.getMessage() as part of the
      // exception message.
      if (getCause() != null && !getMessage().equals(getCause().getMessage())) {
        message += ": " + getCause().getMessage();
      }
      // The detailed code doesn't matter, but it should be well-formed.
      DetailedExitCode code =
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setSpawn(Spawn.newBuilder().setCode(Code.NON_ZERO_EXIT))
                  .build());
      return new ActionExecutionException(message, getCause(), action, true, code);
    }
  }

  protected FileSystem fileSystem;
  protected EventCollectionApparatus events = new EventCollectionApparatus();
  protected OutErr outErr = OutErr.SYSTEM_OUT_ERR;
  protected Path testRoot;
  protected ServerDirectories serverDirectories;
  protected BlazeDirectories directories;
  protected MockToolsConfig mockToolsConfig;
  protected BinTools binTools;

  protected BlazeRuntimeWrapper runtimeWrapper;
  protected Path outputBase;
  protected String outputBaseName = "outputBase";

  private Path workspace;
  protected RecordingExceptionHandler subscriberException = new RecordingExceptionHandler();

  @Before
  public final void createFilesAndMocks() throws Exception  {
    runPriorToBeforeMethods();
    events.setFailFast(false);
    // TODO(mschaller): This will ignore any attempt by Blaze modules to provide a filesystem;
    // consider something better.
    this.fileSystem = createFileSystem();
    this.testRoot = createTestRoot(fileSystem);

    outputBase = testRoot.getRelative(outputBaseName);
    outputBase.createDirectoryAndParents();
    workspace = testRoot.getRelative(getDesiredWorkspaceRelative());
    beforeCreatingWorkspace(workspace);
    workspace.createDirectoryAndParents();
    serverDirectories = createServerDirectories();
    directories =
        new BlazeDirectories(
            serverDirectories,
            workspace,
            /* defaultSystemJavabase= */ null,
            TestConstants.PRODUCT_NAME);
    binTools =
        IntegrationMock.get().getIntegrationBinTools(fileSystem, directories);
    mockToolsConfig = new MockToolsConfig(workspace, realFileSystem());
    setupMockTools();
    createRuntimeWrapper();
  }

  protected ServerDirectories createServerDirectories() {
    return new ServerDirectories(
        /*installBase=*/ outputBase,
        /*outputBase=*/ outputBase,
        /*outputUserRoot=*/ outputBase,
        /*execRootBase=*/ outputBase.getRelative("execroot"),
        // Arbitrary install base hash.
        /*installMD5=*/ "83bc4458738962b9b77480bac76164a9");
  }

  protected void createRuntimeWrapper() throws Exception {
    runtimeWrapper =
        new BlazeRuntimeWrapper(
            events,
            serverDirectories,
            directories,
            binTools,
            getRuntimeBuilder().setEventBusExceptionHandler(subscriberException)) {
          @Override
          protected void finalizeBuildResult(BuildResult result) {
            finishBuildResult(result);
          }
        };
    setupOptions();
  }

  protected void runPriorToBeforeMethods() throws Exception {
    // Allows tests such as SkyframeIntegrationInvalidationTest to execute code before all @Before
    // methods are being run.
  }

  @After
  public final void cleanUp() throws Exception  {
    if (subscriberException.getException() != null) {
      throwIfUnchecked(subscriberException.getException());
      throw new RuntimeException(subscriberException.getException());
    }
    LoggingUtil.installRemoteLoggerForTesting(null);
    testRoot.deleteTreesBelow(); // (comment out during debugging)
  }

  /**
   * A helper class that can be used to record exceptions that occur on the event bus, by passing an
   * instance of it to BlazeRuntime#setEventBusExceptionHandler.
   */
  public static final class RecordingExceptionHandler implements SubscriberExceptionHandler {
    private Throwable exception;

    @Override
    public void handleException(Throwable exception, SubscriberExceptionContext context) {
      System.err.println("subscriber exception: ");
      exception.printStackTrace();
      if (this.exception == null) {
        this.exception = exception;
      }
    }

    public Throwable getException() {
      return exception;
    }
  }

  /**
   * Returns the relative path (from {@code testRoot}) to the desired workspace. This method may be
   * called in {@link #createFilesAndMocks}, so overrides this method should not use any variables
   * that may not have been initialized yet.
   */
  protected PathFragment getDesiredWorkspaceRelative() {
    return PathFragment.create(TestConstants.WORKSPACE_NAME);
  }

  protected InternalTestExecutionMode getInternalTestExecutionMode() {
    return InternalTestExecutionMode.NORMAL;
  }

  /**
   * Called in #setUp before creating the workspace directory. Subclasses should override this
   * if they want to a non-standard filesystem setup, e.g. introduce symlinked directories.
   */
  protected void beforeCreatingWorkspace(@SuppressWarnings("unused") Path workspace)
      throws Exception {}

  protected void finishBuildResult(@SuppressWarnings("unused") BuildResult result) {}

  protected boolean realFileSystem() {
    return true;
  }

  protected FileSystem createFileSystem() throws Exception {
    return FileSystems.getNativeFileSystem();
  }

  protected Path createTestRoot(FileSystem fileSystem) {
    return fileSystem.getPath(TestUtils.tmpDir());
  }

  // This is only here to support HaskellNonIntegrationTest. You should not call or override this
  // method.
  protected void setupMockTools() throws IOException {
    // (Almost) every integration test calls BuildView.doLoadingPhase, which loads the default
    // crosstool, etc.  So we create these package here.
    AnalysisMock.get().setupMockClient(mockToolsConfig);
  }

  protected FileSystem getFileSystem() {
    return fileSystem;
  }

  protected BlazeModule getBuildInfoModule() {
    return new BlazeModule() {
      @Override
      public void workspaceInit(
          BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
        builder.setWorkspaceStatusActionFactory(
            new AnalysisTestUtil.DummyWorkspaceStatusActionFactory());
      }

      @Override
      public void registerActionContexts(
          ModuleActionContextRegistry.Builder registryBuilder,
          CommandEnvironment env,
          BuildRequest buildRequest) {
        registryBuilder.register(
            WorkspaceStatusAction.Context.class, new DummyWorkspaceStatusActionContext());
      }
    };
  }

  // There should be one getSpawnModule at a time, as we lack infrastructure to decide from
  // which Module to take the SpawnActionContext for specific actions.
  protected BlazeModule getSpawnModule() {
    return new StandaloneModule();
  }

  /** Gets a module containing rules (by default, using the TestRuleClassProvider) */
  protected BlazeModule getRulesModule() {
    return TestRuleModule.getModule();
  }

  /** Gets a module to set up the strategies. */
  protected BlazeModule getStrategyModule() {
    return TestStrategyModule.getModule();
  }

  private static BlazeModule getNoResolvedFileModule() {
    return new BlazeModule() {
      @Override
      public ImmutableList<Injected> getPrecomputedValues() {
        return ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty()));
      }
    };
  }

  /**
   * Gets a module that returns a connectivity status.
   *
   * @return a Blaze module that implements {@link ConnectivityStatusProvider}
   */
  protected BlazeModule getConnectivityModule() {
    return new NoOpConnectivityModule();
  }

  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    OptionsParser startupOptionsParser =
        OptionsParser.builder().optionsClasses(getStartupOptionClasses()).build();
    startupOptionsParser.parse(getStartupOptions());
    BlazeModule connectivityModule = getConnectivityModule();
    checkState(
        connectivityModule instanceof ConnectivityStatusProvider,
        "Module returned by getConnectivityModule() does not implement ConnectivityStatusProvider");
    return new BlazeRuntime.Builder()
        .setFileSystem(fileSystem)
        .setProductName(TestConstants.PRODUCT_NAME)
        .setStartupOptionsProvider(startupOptionsParser)
        .addBlazeModule(connectivityModule)
        .addBlazeModule(getNoResolvedFileModule())
        .addBlazeModule(getSpawnModule())
        .addBlazeModule(new IncludeScanningModule())
        .addBlazeModule(getBuildInfoModule())
        .addBlazeModule(getRulesModule())
        .addBlazeModule(getStrategyModule());
  }

  protected List<String> getStartupOptions() {
    return ImmutableList.of();
  }

  protected ImmutableList<Class<? extends OptionsBase>> getStartupOptionClasses() {
    return ImmutableList.of(BlazeServerStartupOptions.class);
  }

  protected void setupOptions() throws Exception {
    runtimeWrapper.resetOptions();

    runtimeWrapper.addOptions(
        // Set visibility to public so that test cases don't have to bother
        // with visibility declarations
        "--default_visibility=public",

        // Don't show progress messages unless we need to, to keep the noise down.
        "--noshow_progress",

        // Don't use ijars, because we don't have the executable in these tests
        "--nouse_ijars");

    runtimeWrapper.addOptions("--experimental_extended_sanity_checks");
    runtimeWrapper.addOptions(TestConstants.PRODUCT_SPECIFIC_FLAGS);
    // TODO(rosica): Remove this once g3 is migrated.
    runtimeWrapper.addOptions("--noincompatible_use_specific_tool_files");
    // TODO(rosica): Remove this once g3 is migrated.
    runtimeWrapper.addOptions("--noincompatible_make_thinlto_command_lines_standalone");
  }

  protected void resetOptions() {
    runtimeWrapper.resetOptions();
  }

  protected void addOptions(String... args) {
    runtimeWrapper.addOptions(args);
  }

  protected OptionsParser createOptionsParser() {
    return runtimeWrapper.createOptionsParser();
  }

  protected Action getGeneratingAction(Artifact artifact) {
    ActionAnalysisMetadata action = getActionGraph().getGeneratingAction(artifact);

    if (action != null) {
      checkState(
          action instanceof Action, "%s is not a proper Action object", action.prettyPrint());
      return (Action) action;
    } else {
      return null;
    }
  }
  /**
   * Returns the path to the executable that label "target" identifies.
   *
   * <p>Assumes that the specified target is executable, i.e. defines getExecutable; use {@link
   * #getArtifacts} instead if this is not the case.
   *
   * @param target the label of the target whose executable location is requested.
   */
  protected Path getExecutableLocation(String target)
      throws LabelSyntaxException, NoSuchPackageException, NoSuchTargetException,
          InterruptedException, TransitionException, InvalidConfigurationException {
    return getExecutable(getConfiguredTarget(target)).getPath();
  }

  /**
   * Given a label (which has typically, but not necessarily, just been built), returns the
   * collection of files that it produces.
   *
   * @param target the label of the target whose artifacts are requested.
   */
  protected Iterable<Artifact> getArtifacts(String target)
      throws LabelSyntaxException, NoSuchPackageException, NoSuchTargetException,
          InterruptedException, TransitionException, InvalidConfigurationException {
    return getFilesToBuild(getConfiguredTarget(target)).toList();
  }

  /**
   * Given a label (which has typically, but not necessarily, just been built), returns the
   * configured target for it using the request configuration.
   *
   * @param target the label of the requested target.
   */
  protected ConfiguredTarget getConfiguredTarget(String target)
      throws LabelSyntaxException, NoSuchPackageException, NoSuchTargetException,
          InterruptedException, TransitionException, InvalidConfigurationException {
    getPackageManager()
        .getTarget(events.reporter(), Label.parseAbsolute(target, ImmutableMap.of()));
    return getSkyframeExecutor()
        .getConfiguredTargetForTesting(events.reporter(), label(target), getTargetConfiguration());
  }

  protected ConfiguredTarget getConfiguredTarget(
      ExtendedEventHandler eventHandler, Label label, BuildConfiguration config)
      throws TransitionException, InvalidConfigurationException, InterruptedException {
    return getSkyframeExecutor().getConfiguredTargetForTesting(eventHandler, label, config);
  }

  /**
   * Gets all the already computed configured targets.
   */
  protected Iterable<ConfiguredTarget> getAllConfiguredTargets() {
    return SkyframeExecutorTestUtils.getAllExistingConfiguredTargets(getSkyframeExecutor());
  }

  /** Gets an existing configured target. */
  protected ConfiguredTarget getExistingConfiguredTarget(String target)
      throws InterruptedException, LabelSyntaxException {
    ConfiguredTarget existingConfiguredTarget =
        SkyframeExecutorTestUtils.getExistingConfiguredTarget(
            getSkyframeExecutor(), label(target), getTargetConfiguration());
    assertWithMessage(target).that(existingConfiguredTarget).isNotNull();
    return existingConfiguredTarget;
  }

  protected BuildConfigurationCollection getConfigurationCollection() {
    return runtimeWrapper.getConfigurationCollection();
  }

  /**
   * Returns the target configuration for the most recent build, as created in Blaze's master
   * configuration creation phase.
   *
   * <p>Tries to find the configuration used by all of the top-level targets in the last invocation.
   * If they used multiple different configurations, or if none of them had a configuration, then
   * falls back to the base top-level configuration.
   */
  protected BuildConfiguration getTargetConfiguration() {
    BuildConfiguration baseConfiguration =
        Iterables.getOnlyElement(getConfigurationCollection().getTargetConfigurations());
    BuildResult result = getResult();
    if (result == null) {
      return baseConfiguration;
    }
    Set<BuildConfiguration> topLevelTargetConfigurations =
        result
            .getActualTargets()
            .stream()
            .map((ct) -> getConfiguration(ct))
            .filter((config) -> config != null)
            .collect(toImmutableSet());
    if (topLevelTargetConfigurations.size() != 1) {
      return baseConfiguration;
    }
    return Iterables.getOnlyElement(topLevelTargetConfigurations);
  }

  protected BuildConfiguration getHostConfiguration() {
    return getConfigurationCollection().getHostConfiguration();
  }

  protected TopLevelArtifactContext getTopLevelArtifactContext() {
    return getRequest().getTopLevelArtifactContext();
  }

  /**
   * Convenience wrapper around buildTool.syncPackageCache() and buildTool.build() that creates and
   * executes a BuildRequest. Returns the BuildRequest on success (it is also subsequently
   * accessible via {@link #getRequest}, even in case of abnormal termination). Also redirects the
   * output from the reporter's event handler to go to this.OutErr during the build, and redirects
   * System.out/System.err to go via the reporter (and hence to this.OutErr) during the build.
   */
  public BuildResult buildTarget(String... targets) throws Exception {
    events.setOutErr(this.outErr);
    runtimeWrapper.executeBuild(Arrays.asList(targets));
    return runtimeWrapper.getLastResult();
  }

  /**
   * Creates a BuildRequest for either blaze build or blaze analyze, using the
   * currently-installed request options.
   * @param commandName blaze build or analyze command
   * @param targets the targets to be built
   */
  protected BuildRequest createNewRequest(String commandName, String... targets) throws Exception {
    runtimeWrapper.initializeOptionsParser();
    return runtimeWrapper.createRequest(commandName, Arrays.asList(targets));
  }

  /**
   * Utility function: parse a string as a label.
   */
  protected static Label label(String labelString) throws LabelSyntaxException {
    return Label.parseAbsolute(labelString, ImmutableMap.of());
  }

  protected String run(Artifact executable, String... arguments) throws Exception {
    Map<String, String> environment = null;
    return run(executable.getPath(), null, environment, arguments);
  }

  /**
   * This runs an executable using the executor instance configured for
   * this test.
   */
  protected String run(Path executable, String... arguments) throws Exception {
    Map<String, String> environment = null;
    return run(executable, null, environment, arguments);
  }

  protected String run(Path executable, Path workingDirectory, String... arguments)
      throws ExecException, InterruptedException {
    return run(executable, workingDirectory, null, arguments);
  }

  protected String run(
      Path executable, Path workingDirectory, Map<String, String> environment, String... arguments)
      throws ExecException, InterruptedException {
    RecordingOutErr outErr = new RecordingOutErr();
    try {
      run(executable, workingDirectory, outErr, environment, arguments);
    } catch (ExecException e) {
      throw new IntegrationTestExecException(
          "failed to execute '"
              + executable.getPathString()
              + "'\n----- captured stdout:\n"
              + outErr.outAsLatin1()
              + "\n----- captured stderr:"
              + outErr.errAsLatin1()
              + "\n----- Reason",
          e.getCause());
    }

    return outErr.outAsLatin1();
  }

  protected void run(Path executable, OutErr outErr, String... arguments) throws Exception {
    run(executable, null, outErr, null, arguments);
  }

  private void run(
      Path executable,
      Path workingDirectory,
      OutErr outErr,
      Map<String, String> environment,
      String... arguments)
      throws ExecException, InterruptedException {
    if (workingDirectory == null) {
      workingDirectory = directories.getWorkspace();
    }
    List<String> argv = Lists.newArrayList(arguments);
    argv.add(0, executable.toString());
    Map<String, String> env =
        (environment != null ? environment : getTargetConfiguration().getLocalShellEnvironment());
    TestFileOutErr testOutErr = new TestFileOutErr();
    try {
      execute(workingDirectory, env, argv, testOutErr, /* verboseFailures= */ false);
    } finally {
      testOutErr.dumpOutAsLatin1(outErr.getOutputStream());
      testOutErr.dumpErrAsLatin1(outErr.getErrorStream());
    }
  }

  /**
   * Writes a number of lines of text to a source file using Latin-1 encoding.
   *
   * @param relativePath the path relative to the workspace root.
   * @param lines the lines of text to write to the file.
   * @return the path of the created file.
   * @throws IOException if the file could not be written.
   */
  public Path write(String relativePath, String... lines) throws IOException {
    Path path = getWorkspace().getRelative(relativePath);
    return writeAbsolute(path, lines);
  }

  /**
   * Same as {@link #write}, but with an absolute path.
   */
  protected Path writeAbsolute(Path path, String... lines) throws IOException {
    FileSystemUtils.writeIsoLatin1(path, lines);
    return path;
  }

  /**
   * Creates folders on the path to {@code relativeLinkPath} and a symlink to {@code target} at
   * {@code relativeLinkPath} (equivalent to {@code ln -s <target> <relativeLinkPath>}).
   */
  protected void createSymlink(String target, String relativeLinkPath) throws IOException {
    Path path = getWorkspace().getRelative(relativeLinkPath);
    path.getParentDirectory().createDirectoryAndParents();
    path.createSymbolicLink(PathFragment.create(target));
  }

  /**
   * The TimestampGranularityMonitor operates on the files created by the
   * request and thus does not help here. Calling this method ensures that files
   * we modify as part of the test environment are considered as changed.
   */
  protected static void waitForTimestampGranularity() throws Exception {
    // Ext4 has a nanosecond granularity. Empirically, tmpfs supports ~5ms increments on
    // Ubuntu Trusty.
    Thread.sleep(10 /*ms*/);
  }

  /**
   * Fork/exec/wait the specified command.  A utility method for subclasses.
   */
  protected String exec(String... argv) throws CommandException {
    return new String(new Command(argv).execute().getStdout(), StandardCharsets.UTF_8);
  }

  /**
   * Performs a local direct spawn execution given spawn information broken out
   * into individual arguments. Directs standard out/err to {@code outErr}.
   *
   * @param workingDirectory the directory from which to execute the subprocess
   * @param environment the environment map to provide to the subprocess. If
   *        null, the environment is inherited from the parent process.
   * @param argv the argument vector including the command itself
   * @param outErr the out+err stream pair to receive stdout and stderr from the
   *        subprocess
   * @throws ExecException if any kind of abnormal termination or command
   *         exception occurs
   */
  public static void execute(
      Path workingDirectory,
      Map<String, String> environment,
      List<String> argv,
      FileOutErr outErr,
      boolean verboseFailures)
      throws ExecException {
    Command command =
        new CommandBuilder()
            .addArgs(argv)
            .setEnv(environment)
            .setWorkingDir(workingDirectory)
            .build();
    try {
      command.execute(outErr.getOutputStream(), outErr.getErrorStream());
    } catch (AbnormalTerminationException e) { // non-zero exit or signal or I/O problem
      IntegrationTestExecException e2 =
          new IntegrationTestExecException(CommandUtils.describeCommandFailure(verboseFailures, e));
      e2.initCause(e); // We don't pass cause=e to the ExecException constructor
      // since we don't want it to contribute to the exception
      // message again; it's already in describeCommandFailure().
      throw e2;
    } catch (CommandException e) {
      IntegrationTestExecException e2 =
          new IntegrationTestExecException(CommandUtils.describeCommandFailure(verboseFailures, e));
      e2.initCause(e); // We don't pass cause=e to the ExecException constructor
      // since we don't want it to contribute to the exception
      // message again; it's already in describeCommandFailure().
      throw e2;
    }
  }

  protected String readContentAsLatin1String(Artifact artifact) throws IOException {
    return new String(FileSystemUtils.readContentAsLatin1(artifact.getPath()));
  }

  /**
   * Given a collection of Artifacts, returns a corresponding set of strings of the form "<root>
   * <relpath>", such as "bin x/libx.a". Such strings make assertions easier to write.
   *
   * <p>The returned set preserves the order of the input.
   */
  protected Set<String> artifactsToStrings(NestedSet<Artifact> artifacts) {
    return AnalysisTestUtil.artifactsToStrings(getConfigurationCollection(), artifacts.toList());
  }

  protected ActionsTestUtil actionsTestUtil() {
    return new ActionsTestUtil(getActionGraph());
  }

  protected Artifact getExecutable(TransitiveInfoCollection target) {
    return target.getProvider(FilesToRunProvider.class).getExecutable();
  }

  protected NestedSet<Artifact> getFilesToBuild(TransitiveInfoCollection target) {
    return target.getProvider(FileProvider.class).getFilesToBuild();
  }

  protected final BuildConfiguration getConfiguration(ConfiguredTarget ct) {
    return getSkyframeExecutor()
        .getConfiguration(NullEventHandler.INSTANCE, ct.getConfigurationKey());
  }

  /**
   * Returns the BuildRequest of the last call to buildTarget().
   */
  protected BuildRequest getRequest() {
    return runtimeWrapper.getLastRequest();
  }

  /**
   * Returns the BuildResultof the last call to buildTarget().
   */
  protected BuildResult getResult() {
    return runtimeWrapper.getLastResult();
  }

  /**
   * Returns the {@link BlazeRuntime} in use.
   */
  protected BlazeRuntime getRuntime() {
    return runtimeWrapper.getRuntime();
  }

  protected BlazeWorkspace getBlazeWorkspace() {
    return runtimeWrapper.getRuntime().getWorkspace();
  }

  protected ConfiguredTargetAndData getConfiguredTargetAndTarget(
      ExtendedEventHandler eventHandler, Label label, BuildConfiguration config)
      throws TransitionException, InvalidConfigurationException, InterruptedException {
    return getSkyframeExecutor().getConfiguredTargetAndDataForTesting(eventHandler, label, config);
  }

  protected ActionGraph getActionGraph() {
    return getSkyframeExecutor().getActionGraph(events.reporter());
  }

  protected CommandEnvironment getCommandEnvironment() {
    return runtimeWrapper.getCommandEnvironment();
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return runtimeWrapper.getSkyframeExecutor();
  }

  protected PackageManager getPackageManager() {
    return getSkyframeExecutor().getPackageManager();
  }

  protected Path getOutputBase() {
    return outputBase;
  }

  protected Path getWorkspace() {
    return workspace;
  }

  /**
   * Assertion-checks that the expected error was reported,
   */
  protected void assertContainsError(String expectedError) {
    for (Event error : events.errors()) {
      if (error.getMessage().contains(expectedError)) {
        return;
      }
    }
    fail("didn't find expected error: \"" + expectedError + "\"");
  }
}
