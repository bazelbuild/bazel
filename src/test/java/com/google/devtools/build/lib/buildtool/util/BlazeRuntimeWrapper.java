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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.util.io.CommandExtensionReporter.NO_OP_COMMAND_EXTENSION_REPORTER;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.profiler.CollectLocalResourceUsage;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeCommandUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.ClientOptions;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.UiOptions;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.sandbox.SandboxOptions;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.worker.WorkerProcessMetricsCollector;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.Any;
import com.google.protobuf.Message;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * A wrapper for {@link BlazeRuntime} for testing purposes that makes it possible to exercise (most)
 * of the build machinery in integration tests. Note that {@code BlazeCommandDispatcher} is not
 * exercised here.
 */
public class BlazeRuntimeWrapper {

  private final BlazeRuntime runtime;
  private CommandEnvironment env;
  private final EventCollectionApparatus events;
  private boolean commandCreated;

  private BuildRequest lastRequest;
  private BuildResult lastResult;
  private BuildConfigurationValue configuration;
  private ImmutableSet<ConfiguredTarget> topLevelTargets;

  private OptionsParser optionsParser;
  private final List<String> optionsToParse = new ArrayList<>();
  private final Map<String, Object> starlarkOptions = new HashMap<>();
  private final List<Class<? extends OptionsBase>> additionalOptionsClasses = new ArrayList<>();
  private final List<String> crashMessages = new ArrayList<>();

  private final List<Object> eventBusSubscribers = new ArrayList<>();

  private final List<String> workspaceSetupWarnings = new ArrayList<>();

  public BlazeRuntimeWrapper(
      EventCollectionApparatus events,
      ServerDirectories serverDirectories,
      BlazeDirectories directories,
      BinTools binTools,
      BlazeRuntime.Builder builder)
      throws Exception {
    this.events = events;
    runtime =
        builder
            .setServerDirectories(serverDirectories)
            .addBlazeModule(
                new BlazeModule() {
                  @Override
                  public void beforeCommand(CommandEnvironment env) {
                    // This only does something interesting for tests that create their own
                    // BlazeCommandDispatcher. :-(
                    if (BlazeRuntimeWrapper.this.env != env) {
                      BlazeRuntimeWrapper.this.env = env;
                      BlazeRuntimeWrapper.this.lastRequest = null;
                      BlazeRuntimeWrapper.this.lastResult = null;
                      resetOptions();
                      env.getEventBus().register(this);
                    }
                  }

                  @Subscribe
                  public void analysisPhaseComplete(AnalysisPhaseCompleteEvent e) {
                    topLevelTargets = ImmutableSet.copyOf(e.getTopLevelTargets());
                  }
                })
            .addBlazeModule(
                new BlazeModule() {
                  @Override
                  public void beforeCommand(CommandEnvironment env) {
                    BlazeRuntimeWrapper.this.events.initExternal(env.getReporter());
                  }
                })
            .build();
    runtime.initWorkspace(directories, binTools);
  }

  public final BlazeRuntime getRuntime() {
    return runtime;
  }

  /** Registers the given {@code subscriber} with the {@link EventBus} before each command. */
  public void registerSubscriber(Object subscriber) {
    eventBusSubscribers.add(subscriber);
  }

  public final CommandEnvironment newCommand() throws Exception {
    return newCommand(BuildCommand.class);
  }

  /** Creates a new command environment; executeBuild does this automatically if you do not. */
  public final CommandEnvironment newCommand(Class<? extends BlazeCommand> command)
      throws Exception {
    return newCommandWithExtensions(command, /* extensions= */ ImmutableList.of());
  }

  /**
   * Creates a new command environment with additional proto extensions as if they were passed to
   * the blaze server.
   */
  public final CommandEnvironment newCommandWithExtensions(
      Class<? extends BlazeCommand> command, List<Message> extensions) throws Exception {
    Command commandAnnotation =
        checkNotNull(
            command.getAnnotation(Command.class),
            "BlazeCommand %s missing command annotation",
            command);
    additionalOptionsClasses.addAll(
        BlazeCommandUtils.getOptions(
            command, runtime.getBlazeModules(), runtime.getRuleClassProvider()));
    initializeOptionsParser(commandAnnotation);
    commandCreated = true;
    if (env != null) {
      runtime.afterCommand(env, BlazeCommandResult.success());
    }

    checkNotNull(
        optionsParser,
        "The options parser must be initialized before creating a new command environment");
    optionsParser.setStarlarkOptions(starlarkOptions);

    env =
        runtime
            .getWorkspace()
            .initCommand(
                commandAnnotation,
                optionsParser,
                workspaceSetupWarnings,
                /* waitTimeInMs= */ 0L,
                /* commandStartTime= */ 0L,
                extensions.stream().map(Any::pack).collect(toImmutableList()),
                this.crashMessages::add,
                NO_OP_COMMAND_EXTENSION_REPORTER,
                /* attemptNumber= */ 1);
    return env;
  }

  /**
   * Returns the command environment. You must call {@link #newCommand()} before calling this
   * method.
   */
  public CommandEnvironment getCommandEnvironment() {
    return env;
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return runtime.getWorkspace().getSkyframeExecutor();
  }

  public void resetOptions() {
    optionsToParse.clear();
    starlarkOptions.clear();
  }

  public void addOptions(String... args) {
    addOptions(ImmutableList.copyOf(args));
  }

  public void addOptions(List<String> args) {
    optionsToParse.addAll(args);
  }

  public void addStarlarkOption(String label, Object value) {
    starlarkOptions.put(Label.parseCanonicalUnchecked(label).getCanonicalForm(), value);
  }

  public void addStarlarkOptions(Map<String, Object> starlarkOptions) {
    starlarkOptions.forEach(this::addStarlarkOption);
  }

  public ImmutableList<String> getOptions() {
    return ImmutableList.copyOf(optionsToParse);
  }

  public <O extends OptionsBase> O getOptions(Class<O> optionsClass) {
    return optionsParser.getOptions(optionsClass);
  }

  public ImmutableMap<String, Object> getStarlarkOptions() {
    return ImmutableMap.copyOf(starlarkOptions);
  }

  public void addOptionsClass(Class<? extends OptionsBase> optionsClass) {
    additionalOptionsClasses.add(optionsClass);
  }

  void finalizeBuildResult(@SuppressWarnings("unused") BuildResult request) {}

  /**
   * Initializes a new options parser, parsing all the options set by {@link
   * #addOptions(String...)}.
   */
  private void initializeOptionsParser(Command commandAnnotation) throws OptionsParsingException {
    // Create the options parser and parse all the options collected so far
    optionsParser = createOptionsParser(commandAnnotation);
    optionsParser.parse(optionsToParse);

    // Enforce the test invocation policy once the options have been added
    InvocationPolicyEnforcer optionsPolicyEnforcer =
        new InvocationPolicyEnforcer(
            runtime.getModuleInvocationPolicy(), Level.FINE, /* conversionContext= */ null);
    try {
      optionsPolicyEnforcer.enforce(optionsParser, commandAnnotation.name());
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
  }

  private OptionsParser createOptionsParser(Command commandAnnotation) {
    Set<Class<? extends OptionsBase>> options =
        new HashSet<>(
            ImmutableList.of(
                BuildRequestOptions.class,
                BuildEventProtocolOptions.class,
                ExecutionOptions.class,
                LocalExecutionOptions.class,
                CommonCommandOptions.class,
                ClientOptions.class,
                LoadingOptions.class,
                AnalysisOptions.class,
                KeepGoingOption.class,
                LoadingPhaseThreadsOption.class,
                PackageOptions.class,
                BuildLanguageOptions.class,
                UiOptions.class,
                SandboxOptions.class));
    options.addAll(additionalOptionsClasses);

    for (BlazeModule module : runtime.getBlazeModules()) {
      Iterables.addAll(options, module.getCommonCommandOptions());
      Iterables.addAll(options, module.getCommandOptions(commandAnnotation));
    }
    options.addAll(runtime.getRuleClassProvider().getFragmentRegistry().getOptionsClasses());
    return OptionsParser.builder().optionsClasses(options).build();
  }

  public void executeBuild(List<String> targets) throws Exception {
    if (!commandCreated) {
      // If you didn't create a command we do it for you
      newCommand();
    }
    commandCreated = false;
    BuildTool buildTool = new BuildTool(env);
    Reporter reporter = env.getReporter();
    try (OutErr.SystemPatcher systemOutErrPatcher = reporter.getOutErr().getSystemPatcher()) {
      Profiler.instance()
          .start(
              /* profiledTasks= */ ImmutableSet.of(),
              /* stream= */ null,
              /* format= */ null,
              /* outputBase= */ null,
              /* buildID= */ null,
              /* recordAllDurations= */ false,
              new JavaClock(),
              /* execStartTimeNanos= */ 42,
              /* slimProfile= */ false,
              /* includePrimaryOutput= */ false,
              /* includeTargetLabel= */ false,
              /* collectTaskHistograms= */ true,
              new CollectLocalResourceUsage(
                  runtime.getBugReporter(),
                  WorkerProcessMetricsCollector.instance(),
                  env.getLocalResourceManager(),
                  /* collectWorkerDataInProfiler= */ false,
                  /* collectLoadAverage= */ false,
                  /* collectSystemNetworkUsage= */ false,
                  /* collectResourceManagerEstimation= */ false,
                  /* collectPressureStallIndicators= */ false));

      StoredEventHandler storedEventHandler = new StoredEventHandler();
      reporter.addHandler(storedEventHandler);

      env.decideKeepIncrementalState();

      // This cannot go into newCommand, because we hook up the EventCollectionApparatus as a
      // module, and after that ran, further changes to the apparatus aren't reflected on the
      // reporter.
      for (BlazeModule module : runtime.getBlazeModules()) {
        module.beforeCommand(env);
      }
      reporter.removeHandler(storedEventHandler);

      EventBus eventBus = env.getEventBus();
      for (Object subscriber : eventBusSubscribers) {
        eventBus.register(subscriber);
      }

      // Replay events from decideKeepIncrementalState and beforeCommand, just as
      // BlazeCommandDispatcher does.
      storedEventHandler.replayOn(reporter);

      env.beforeCommand(InvocationPolicy.getDefaultInstance());

      lastRequest = createRequest(env.getCommandName(), targets);
      lastResult = new BuildResult(lastRequest.getStartTime());

      for (BlazeModule module : runtime.getBlazeModules()) {
        env.getSkyframeExecutor().injectExtraPrecomputedValues(module.getPrecomputedValues());
      }

      Crash crash = null;
      DetailedExitCode detailedExitCode = DetailedExitCode.of(createGenericDetailedFailure());
      try {
        try (SilentCloseable c = Profiler.instance().profile("syncPackageLoading")) {
          env.syncPackageLoading(lastRequest);
        }
        buildTool.buildTargets(lastRequest, lastResult, null);
        detailedExitCode = DetailedExitCode.success();
      } catch (RuntimeException | Error e) {
        crash = Crash.from(e);
        detailedExitCode = crash.getDetailedExitCode();
        throw e;
      } finally {
        env.getTimestampGranularityMonitor().waitForTimestampGranularity(lastRequest.getOutErr());
        this.configuration = lastResult.getBuildConfiguration();
        finalizeBuildResult(lastResult);
        buildTool.stopRequest(
            lastResult, crash != null ? crash.getThrowable() : null, detailedExitCode);
        getSkyframeExecutor().notifyCommandComplete(reporter);
        if (crash != null) {
          runtime
              .getBugReporter()
              .handleCrash(crash, CrashContext.keepAlive().reportingTo(reporter));
        }
      }
    } finally {
      Profiler.instance().stop();
    }
  }

  private static FailureDetail createGenericDetailedFailure() {
    return FailureDetail.newBuilder()
        .setSpawn(Spawn.newBuilder().setCode(Code.NON_ZERO_EXIT))
        .build();
  }

  private BuildRequest createRequest(String commandName, List<String> targets) {
    BuildRequest.Builder builder =
        BuildRequest.builder()
            .setCommandName(commandName)
            .setId(env.getCommandId())
            .setOptions(optionsParser)
            .setStartupOptions(null)
            .setOutErr(env.getReporter().getOutErr())
            .setTargets(targets)
            .setStartTimeMillis(runtime.getClock().currentTimeMillis());
    if ("test".equals(commandName)) {
      builder.setRunTests(true);
    }
    return builder.build();
  }

  @Nullable // Null if no build has been run.
  public BuildRequest getLastRequest() {
    return lastRequest;
  }

  @Nullable // Null if no build has been run.
  public BuildResult getLastResult() {
    return lastResult;
  }

  @Nullable // Null if no build has been run.
  public BuildConfigurationValue getConfiguration() {
    return configuration;
  }

  public List<String> getCrashMessages() {
    return crashMessages;
  }
}
