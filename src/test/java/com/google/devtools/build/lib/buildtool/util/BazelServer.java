// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.TopLevelConfigRequestedEvent;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil.DummyWorkspaceStatusActionContext;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.TestPolicy;
import com.google.devtools.build.lib.integration.util.IntegrationMock;
import com.google.devtools.build.lib.metrics.MetricsModule;
import com.google.devtools.build.lib.metrics.PostGCMemoryUseRecorder.GcAfterBuildModule;
import com.google.devtools.build.lib.metrics.PostGCMemoryUseRecorder.PostGCMemoryUseRecorderModule;
import com.google.devtools.build.lib.network.NoOpConnectivityModule;
import com.google.devtools.build.lib.outputfilter.OutputFilteringModule;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.BlazeService;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.NoSpawnCacheModule;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.CleanCommand;
import com.google.devtools.build.lib.runtime.commands.CoverageCommand;
import com.google.devtools.build.lib.runtime.commands.CqueryCommand;
import com.google.devtools.build.lib.runtime.commands.InfoCommand;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.runtime.commands.RunCommand;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.sandbox.SandboxModule;
import com.google.devtools.build.lib.server.FailureDetails.Command;
import com.google.devtools.build.lib.server.FailureDetails.Command.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.shell.WindowsSubprocessFactory;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.BuildResultListener;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkymeldModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestServices;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.SerializedAbruptExitException;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.build.lib.worker.WorkerModule;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

/**
 * In-process integration test server for Bazel and Blaze.
 *
 * <p>{@code BazelServer} dispatches commands ({@code build}, {@code test}, {@code query}, {@code
 * cquery}, etc.) through {@link BlazeCommandDispatcher#exec}, exercising the full server lifecycle
 * including startup/command option parsing, invocation policy, {@link CommandEnvironment} setup,
 * {@link BlazeModule#beforeCommand} and {@link BlazeModule#afterCommand} hooks, and {@link
 * BlazeCommand#exec}. Each command returns a {@link CommandResult} with the resulting {@link
 * ExitCode}, {@link DetailedExitCode}, {@link FailureDetail}, captured output, and events.
 *
 * <p>Implements {@link TestRule} and {@link AutoCloseable} so it can be used via composition as a
 * JUnit {@code @Rule}, constructed on demand via {@link Builder}, or used through the {@link
 * BazelIntegrationTestCase} convenience base class while providing in-process introspection of
 * {@link SkyframeExecutor}, {@link ConfiguredTarget}s, and events.
 */
public class BazelServer implements TestRule, AutoCloseable {

  static {
    WindowsSubprocessFactory.maybeInstallWindowsSubprocessFactory();
  }

  private final List<BlazeModule> userModules;

  private boolean initialized = false;
  private FileSystem fileSystem;
  private Path outputBase;
  private Path workspaceDir;
  private TestWorkspace workspace;
  private EventCollectionApparatus events;
  private BlazeRuntime runtime;
  private BlazeCommandDispatcher dispatcher;
  private ServerModule serverModule;
  private BlazeDirectories directories;
  private final List<String> persistentOptions = new ArrayList<>();
  private OutErr customOutErr;

  protected BazelServer(List<BlazeModule> userModules) {
    this.userModules = userModules;
  }

  /** Initializes the server and workspace if not already initialized. */
  protected final void ensureInitialized() {
    if (!initialized) {
      try {
        startInternal(false);
      } catch (Exception e) {
        throw new IllegalStateException("Failed to start BazelServer", e);
      }
    }
  }

  /** Returns a new {@link Builder}. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder and lifecycle rule for {@link BazelServer}. */
  public static class Builder implements TestRule {

    protected final List<BlazeModule> modules = new ArrayList<>();
    private final List<BazelServer> activeServers = Collections.synchronizedList(new ArrayList<>());

    public Builder() {}

    /** Adds a {@link BlazeModule} to be registered with the server runtime. */
    @CanIgnoreReturnValue
    public Builder addBlazeModule(BlazeModule module) {
      this.modules.add(module);
      return this;
    }

    /** Builds and initializes a {@link BazelServer} with the current builder configuration. */
    public BazelServer build() {
      BazelServer server = createServer();
      server.ensureInitialized();
      activeServers.add(server);
      return server;
    }

    /** Factory method to create the server instance. Subclasses can override. */
    protected BazelServer createServer() {
      return new BazelServer(new ArrayList<>(modules));
    }

    @Override
    public Statement apply(Statement base, Description description) {
      return new Statement() {
        @Override
        public void evaluate() throws Throwable {
          activeServers.clear();
          try {
            base.evaluate();
          } finally {
            for (BazelServer server : activeServers) {
              server.close();
            }
            activeServers.clear();
          }
        }
      };
    }
  }

  @Override
  public Statement apply(Statement base, Description description) {
    return new Statement() {
      @Override
      public void evaluate() throws Throwable {
        ensureInitialized();
        try {
          base.evaluate();
        } finally {
          close();
        }
      }
    };
  }

  /** Returns the digest hash function used by the server filesystem. */
  protected DigestHashFunction getDigestHashFunction() {
    return DigestHashFunction.SHA256;
  }

  /** Returns the startup option classes parsed during server initialization. */
  protected ImmutableList<Class<? extends OptionsBase>> getStartupOptionClasses() {
    return ImmutableList.of(BlazeServerStartupOptions.class);
  }

  /** Returns the default services registered with the server runtime. */
  protected ImmutableList<BlazeService> getDefaultServices() {
    return TestServices.BLAZE_SERVICES;
  }

  /**
   * Returns the spawn strategy modules registered before strategy modules.
   *
   * <p>Subclasses can override this method to customize spawn strategies.
   */
  protected ImmutableList<BlazeModule> getSpawnModules(
      ServerDirectories serverDirectories, BlazeDirectories directories) {
    return AnalysisMock.get().isThisBazel()
        ? ImmutableList.of(new StandaloneModule(), new SandboxModule())
        : ImmutableList.of(new StandaloneModule());
  }

  /** Returns the default modules registered with the server runtime. */
  protected ImmutableList<BlazeModule> getDefaultModules(
      ServerDirectories serverDirectories, BlazeDirectories directories) {
    ImmutableList.Builder<BlazeModule> modules = ImmutableList.builder();
    modules.add(serverModule);
    modules.add(new OutputFilteringModule());
    modules.add(new NoOpConnectivityModule());
    modules.add(new SkymeldModule());
    modules.add(new CredentialModule());

    modules.addAll(getSpawnModules(serverDirectories, directories));

    modules.add(createBuildInfoModule());
    modules.add(TestRuleModule.getModule());
    modules.add(TestStrategyModule.getModule());

    if (AnalysisMock.get().isThisBazel()) {
      modules.add(new NoSpawnCacheModule());
      modules.add(new WorkerModule());
    }

    modules.add(AnalysisMock.get().getBazelRepositoryModule(directories));
    modules.add(new PostGCMemoryUseRecorderModule());
    modules.add(new GcAfterBuildModule());
    modules.add(new MetricsModule());
    return modules.build();
  }

  /** Returns the {@link BlazeCommand} used for the {@code run} command, or null if unsupported. */
  @Nullable
  protected BlazeCommand getRunCommand() {
    return new RunCommand(TestPolicy.EMPTY_POLICY);
  }

  /** Sets up mock tools and mock client in the workspace. */
  protected void setupMockClient(MockToolsConfig mockToolsConfig) throws IOException {
    AnalysisMock.get().setupMockToolsRepository(mockToolsConfig);
    AnalysisMock.get().setupMockClient(mockToolsConfig);
  }

  /** Hook for subclasses to post-process the workspace after initialization. */
  protected void postProcessWorkspace(TestWorkspace workspace) throws IOException {}

  /** Hook for subclasses to post-process the runtime builder before building runtime. */
  protected void postProcessRuntimeBuilder(BlazeRuntime.Builder runtimeBuilder) throws Exception {}

  /** Starts the server and initializes the workspace environment. */
  private void startInternal(boolean keepWorkspace) throws Exception {
    if (initialized) {
      return;
    }

    events = new EventCollectionApparatus(EventKind.ERRORS_WARNINGS_AND_INFO);
    events.setFailFast(false);

    fileSystem = FileSystems.getNativeFileSystem(getDigestHashFunction());

    Path testRoot = fileSystem.getPath(TestUtils.tmpDir());
    outputBase = testRoot.getRelative("outputBase");
    workspaceDir = testRoot.getRelative("test_workspace");
    workspace = new TestWorkspace(workspaceDir);

    if (!keepWorkspace) {
      cleanDirectory(outputBase);
      outputBase.createDirectoryAndParents();

      cleanDirectory(workspaceDir);
      workspaceDir.createDirectoryAndParents();
      postProcessWorkspace(workspace);
    } else {
      if (!outputBase.exists()) {
        outputBase.createDirectoryAndParents();
      }
      if (!workspaceDir.exists()) {
        workspaceDir.createDirectoryAndParents();
      }
    }

    ServerDirectories serverDirectories =
        new ServerDirectories(
            /* installBase= */ outputBase,
            /* outputBase= */ outputBase,
            /* outputUserRoot= */ outputBase,
            /* execRootBase= */ outputBase.getRelative(ServerDirectories.EXECROOT),
            /* virtualSourceRoot= */ null,
            /* installMD5= */ "83bc4458738962b9b77480bac76164a9");

    directories = new BlazeDirectories(serverDirectories, workspaceDir, TestConstants.PRODUCT_NAME);

    BinTools binTools = IntegrationMock.get().getIntegrationBinTools(fileSystem, directories);
    if (!keepWorkspace) {
      MockToolsConfig mockToolsConfig =
          new MockToolsConfig(workspaceDir, /* realFileSystem= */ true);
      setupMockClient(mockToolsConfig);
    }

    serverModule = new ServerModule(events);

    OptionsParser startupOptionsParser =
        OptionsParser.builder().optionsClasses(getStartupOptionClasses()).build();
    startupOptionsParser.parse(ImmutableList.of());

    BlazeRuntime.Builder runtimeBuilder =
        new BlazeRuntime.Builder()
            .setFileSystem(fileSystem)
            .setProductName(TestConstants.PRODUCT_NAME)
            .setBugReporter(BugReporter.defaultInstance())
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(startupOptionsParser);

    for (BlazeModule module : getDefaultModules(serverDirectories, directories)) {
      runtimeBuilder.addBlazeModule(module);
    }
    for (BlazeService service : getDefaultServices()) {
      runtimeBuilder.addBlazeService(service);
    }
    for (BlazeModule userModule : userModules) {
      runtimeBuilder.addBlazeModule(userModule);
    }

    prepareRuntimeBuilder(runtimeBuilder);
    postProcessRuntimeBuilder(runtimeBuilder);

    runtime = runtimeBuilder.build();
    runtime.initWorkspace(directories, binTools);
    dispatcher = new BlazeCommandDispatcher(runtime);

    persistentOptions.clear();

    initialized = true;
  }

  private static void cleanDirectory(@Nullable Path dir) {
    try {
      if (dir == null || !dir.exists()) {
        return;
      }
      if (OS.getCurrent() == OS.WINDOWS) {
        bestEffortDeleteTreesBelow(dir);
      } else {
        dir.deleteTreesBelow();
      }
    } catch (IOException ignored) {
      // Best-effort cleanup.
    }
  }

  private static void bestEffortDeleteTreesBelow(Path path) throws IOException {
    for (Dirent dirent : path.readdir(Symlinks.NOFOLLOW)) {
      Path child = path.getRelative(dirent.getName());
      if (dirent.getType() == Dirent.Type.DIRECTORY) {
        try {
          child.deleteTree();
        } catch (IOException e) {
          bestEffortDeleteTreesBelow(child);
        }
        continue;
      }
      try {
        child.delete();
      } catch (IOException ignored) {
        // Best-effort file deletion on Windows where DLLs or JARs may remain locked by the JVM.
      }
    }
  }

  /** Shuts down the server and cleans up workspace directories. */
  @Override
  public void close() {
    if (events != null) {
      events.clear();
    }
    try {
      try {
        SkyframeExecutor executor = getSkyframeExecutor();
        if (executor != null && executor.getEvaluator() != null) {
          executor.getEvaluator().cleanupInterningPools();
        }
      } finally {
        if (runtime != null) {
          runtime.getBlazeModules().forEach(BlazeModule::blazeShutdown);
        }
      }
    } finally {
      cleanDirectory(workspaceDir);
      cleanDirectory(outputBase);
      initialized = false;
      customOutErr = null;
    }
  }

  /** Executes a `build` command with the given arguments. */
  @CanIgnoreReturnValue
  public CommandResult build(String... args) throws Exception {
    return runCommand("build", args);
  }

  /** Executes a `test` command with the given arguments. */
  @CanIgnoreReturnValue
  public CommandResult test(String... args) throws Exception {
    return runCommand("test", args);
  }

  /** Executes a `query` command with the given arguments. */
  @CanIgnoreReturnValue
  public CommandResult query(String... args) throws Exception {
    return runCommand("query", args);
  }

  /** Executes a `cquery` command with the given arguments. */
  @CanIgnoreReturnValue
  public CommandResult cquery(String... args) throws Exception {
    return runCommand("cquery", args);
  }

  /** Executes an arbitrary command via {@link BlazeCommandDispatcher}. */
  @CanIgnoreReturnValue
  public CommandResult runCommand(String command, String... args) throws Exception {
    if (!initialized) {
      throw new IllegalStateException("BazelServer has been closed");
    }

    List<String> fullArgs = new ArrayList<>();
    fullArgs.add(command);
    if (command.equals("build")
        || command.equals("test")
        || command.equals("run")
        || command.equals("cquery")) {
      fullArgs.addAll(getDefaultCommandOptions());
    }
    fullArgs.addAll(persistentOptions);
    fullArgs.addAll(Arrays.asList(args));

    ByteArrayOutputStream stdoutStream = new ByteArrayOutputStream();
    ByteArrayOutputStream stderrStream = new ByteArrayOutputStream();
    OutErr outErr;
    if (customOutErr != null) {
      outErr =
          OutErr.create(
              new TeeOutputStream(stdoutStream, customOutErr.getOutputStream()),
              new TeeOutputStream(stderrStream, customOutErr.getErrorStream()));
    } else {
      outErr = OutErr.create(stdoutStream, stderrStream);
    }

    Instant start = Instant.now();
    BlazeCommandResult result = dispatcher.exec(fullArgs, "bazel-integration-test", outErr);
    Duration duration = Duration.between(start, Instant.now());

    return new CommandResult(
        result, stdoutStream.toString(UTF_8), stderrStream.toString(UTF_8), duration);
  }

  /** Sets a custom {@link OutErr} to which command stdout and stderr are forwarded. */
  public void setOutErr(OutErr outErr) {
    this.customOutErr = outErr;
  }

  private static final class TeeOutputStream extends OutputStream {
    private final OutputStream out1;
    private final OutputStream out2;

    TeeOutputStream(OutputStream out1, OutputStream out2) {
      this.out1 = out1;
      this.out2 = out2;
    }

    @Override
    public void write(int b) throws IOException {
      out1.write(b);
      out2.write(b);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      out1.write(b, off, len);
      out2.write(b, off, len);
    }

    @Override
    public void flush() throws IOException {
      out1.flush();
      out2.flush();
    }

    @Override
    public void close() throws IOException {
      try {
        out1.close();
      } finally {
        out2.close();
      }
    }
  }

  /** Adds persistent options that will be passed to subsequent commands. */
  public void addOptions(String... options) {
    persistentOptions.addAll(Arrays.asList(options));
  }

  /** Adds persistent options that will be passed to subsequent commands. */
  public void addOptions(List<String> options) {
    persistentOptions.addAll(options);
  }

  /** Clears all persistent options. */
  public void resetOptions() {
    persistentOptions.clear();
  }

  /** Returns the list of persistent options passed to subsequent commands. */
  public ImmutableList<String> getPersistentOptions() {
    return ImmutableList.copyOf(persistentOptions);
  }

  /** Access to workspace file operations. */
  public TestWorkspace workspace() {
    if (!initialized) {
      throw new IllegalStateException("BazelServer has been closed");
    }
    return workspace;
  }

  @Nullable
  public CommandEnvironment getCommandEnvironment() {
    return serverModule.getCommandEnvironment();
  }

  @Nullable
  public BuildResultListener getBuildResultListener() {
    return serverModule.getBuildResultListener();
  }

  public SkyframeExecutor getSkyframeExecutor() {
    CommandEnvironment env = getCommandEnvironment();
    if (env != null) {
      return env.getSkyframeExecutor();
    }
    if (runtime != null && runtime.getWorkspace() != null) {
      return runtime.getWorkspace().getSkyframeExecutor();
    }
    throw new IllegalStateException("No SkyframeExecutor available");
  }

  public PackageManager getPackageManager() {
    return getSkyframeExecutor().getPackageManager();
  }

  public ImmutableSet<ConfiguredTarget> getAnalyzedTargets() {
    BuildResultListener listener = getBuildResultListener();
    return listener != null ? listener.getAnalyzedTargets() : ImmutableSet.of();
  }

  public ImmutableList<String> getLabelsOfAnalyzedTargets() {
    return getAnalyzedTargets().stream()
        .map(x -> x.getLabel().toString())
        .collect(toImmutableList());
  }

  public ImmutableSet<ConfiguredTargetKey> getBuiltTargets() {
    BuildResultListener listener = getBuildResultListener();
    return listener != null ? listener.getBuiltTargets() : ImmutableSet.of();
  }

  public ImmutableList<String> getLabelsOfBuiltTargets() {
    return getBuiltTargets().stream().map(x -> x.getLabel().toString()).collect(toImmutableList());
  }

  public ImmutableSet<AspectKey> getAnalyzedAspectKeys() {
    BuildResultListener listener = getBuildResultListener();
    return listener != null ? listener.getAnalyzedAspects().keySet() : ImmutableSet.of();
  }

  public ImmutableList<String> getLabelsOfAnalyzedAspects() {
    return getAnalyzedAspectKeys().stream()
        .map(x -> x.getLabel().toString())
        .collect(toImmutableList());
  }

  public ImmutableSet<AspectKey> getBuiltAspects() {
    BuildResultListener listener = getBuildResultListener();
    return listener != null ? listener.getBuiltAspects() : ImmutableSet.of();
  }

  public ImmutableList<String> getLabelsOfBuiltAspects() {
    return getBuiltAspects().stream().map(x -> x.getLabel().toString()).collect(toImmutableList());
  }

  public ImmutableSet<ConfiguredTarget> getSkippedTargets() {
    BuildResultListener listener = getBuildResultListener();
    return listener != null ? listener.getSkippedTargets() : ImmutableSet.of();
  }

  public ImmutableList<String> getLabelsOfSkippedTargets() {
    return getSkippedTargets().stream()
        .map(x -> x.getLabel().toString())
        .collect(toImmutableList());
  }

  public ImmutableSet<ConfiguredTarget> getAnalyzedTests() {
    BuildResultListener listener = getBuildResultListener();
    return listener != null ? listener.getAnalyzedTests() : ImmutableSet.of();
  }

  public ImmutableList<String> getLabelsOfAnalyzedTests() {
    return getAnalyzedTests().stream().map(x -> x.getLabel().toString()).collect(toImmutableList());
  }

  /** Access to event collection and assertions. */
  public EventCollectionApparatus events() {
    return events;
  }

  /** Returns the initialized {@link BlazeRuntime}. */
  public BlazeRuntime getRuntime() {
    return runtime;
  }

  /** Returns the {@link FileSystem} in use. */
  public FileSystem getFileSystem() {
    return fileSystem;
  }

  protected List<String> getDefaultCommandOptions() {
    ImmutableList.Builder<String> defaults = ImmutableList.builder();
    defaults.add(
        "--default_visibility=public",
        "--noshow_progress",
        "--nouse_ijars",
        "--noexperimental_collect_system_network_usage",
        "--experimental_extended_sanity_checks");
    defaults.addAll(TestConstants.PRODUCT_SPECIFIC_FLAGS);
    defaults.addAll(TestConstants.PRODUCT_SPECIFIC_BUILD_LANG_OPTIONS);
    if (AnalysisMock.get().isThisBazel()) {
      defaults.add("--override_repository=bazel_tools=embedded_tools");
    }
    if (OS.getCurrent() == OS.WINDOWS) {
      defaults.add("--shell_executable=c:/msys64/usr/bin/bash.exe");
    }
    return defaults.build();
  }

  private static BlazeModule createBuildInfoModule() {
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

  private static void prepareRuntimeBuilder(BlazeRuntime.Builder builder)
      throws AbruptExitException {
    var startupOptions = builder.getStartupOptionsProvider();
    var blazeServices = builder.getBlazeServices();
    for (BlazeService blazeService : blazeServices) {
      try {
        blazeService.globalInit(startupOptions, blazeServices);
      } catch (SerializedAbruptExitException e) {
        try {
          FailureDetail failureDetail =
              FailureDetail.parseFrom(
                  e.getSerializedFailureDetail(), ExtensionRegistryLite.getEmptyRegistry());
          throw new AbruptExitException(DetailedExitCode.of(failureDetail), e);
        } catch (InvalidProtocolBufferException ipbe) {
          throw new AbruptExitException(
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(
                          "Failed to parse FailureDetail from SerializedAbruptExitException: "
                              + ipbe.getMessage())
                      .setCommand(Command.newBuilder().setCode(Code.COMMAND_FAILURE_UNKNOWN))
                      .build()),
              ipbe);
        }
      }
    }
    for (BlazeModule blazeModule : builder.getBlazeModules()) {
      blazeModule.globalInit(startupOptions, blazeServices);
    }
  }

  private final class ServerModule extends BlazeModule {
    private final EventCollectionApparatus events;
    private CommandEnvironment env;
    private BuildResultListener buildResultListener;
    private BuildConfigurationValue targetConfiguration;

    ServerModule(EventCollectionApparatus events) {
      this.events = events;
    }

    @Override
    public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
      builder.addCommands(
          new BuildCommand(),
          new QueryCommand(),
          new CqueryCommand(),
          new InfoCommand(),
          new TestCommand(),
          new CoverageCommand(),
          new CleanCommand());
      BlazeCommand runCommand = getRunCommand();
      if (runCommand != null) {
        builder.addCommands(runCommand);
      }
    }

    @Override
    public void beforeCommand(CommandEnvironment env) {
      this.env = env;
      this.buildResultListener = env.getBuildResultListener();
      this.events.initExternal(env.getReporter());
      env.getEventBus().register(this);
    }

    @Subscribe
    @AllowConcurrentEvents
    public void onTopLevelConfigRequested(TopLevelConfigRequestedEvent event) {
      this.targetConfiguration = event.topLevelConfig();
    }

    @Override
    public void afterAnalysis(
        CommandEnvironment env,
        BuildRequest request,
        BuildOptions buildOptions,
        AnalysisResult analysisResult) {
      if (analysisResult.getConfiguration() != null) {
        this.targetConfiguration = analysisResult.getConfiguration();
      }
    }

    @Override
    public void afterTopLevelTargetAnalysis(
        CommandEnvironment env,
        BuildRequest request,
        BuildOptions buildOptions,
        ConfiguredTarget configuredTarget) {
      if (this.targetConfiguration == null && configuredTarget.getConfigurationKey() != null) {
        this.targetConfiguration =
            env.getSkyframeExecutor()
                .getConfiguration(env.getReporter(), configuredTarget.getConfigurationKey());
      }
    }

    CommandEnvironment getCommandEnvironment() {
      return env;
    }

    BuildResultListener getBuildResultListener() {
      return buildResultListener;
    }
  }
}
