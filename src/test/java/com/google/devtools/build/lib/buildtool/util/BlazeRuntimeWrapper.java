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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
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
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.Any;
import com.google.protobuf.Message;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
  private BuildConfigurationCollection configurations;
  private ImmutableSet<ConfiguredTarget> topLevelTargets;

  private OptionsParser optionsParser;
  private ImmutableList.Builder<String> optionsToParse = new ImmutableList.Builder<>();
  private final List<Class<? extends OptionsBase>> additionalOptionsClasses = new ArrayList<>();

  private final List<Object> eventBusSubscribers = new ArrayList<>();

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
    optionsParser = createOptionsParser();
  }

  @Command(name = "build", builds = true, help = "", shortDescription = "")
  private static class DummyBuildCommand {}

  public OptionsParser createOptionsParser() {
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
      Iterables.addAll(
          options, module.getCommandOptions(DummyBuildCommand.class.getAnnotation(Command.class)));
    }
    options.addAll(runtime.getRuleClassProvider().getConfigurationOptions());
    return OptionsParser.builder().optionsClasses(options).build();
  }

  private void enforceTestInvocationPolicy(OptionsParser parser) {
    InvocationPolicyEnforcer optionsPolicyEnforcer =
        new InvocationPolicyEnforcer(runtime.getModuleInvocationPolicy());
    try {
      optionsPolicyEnforcer.enforce(parser);
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
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
    return newCommandWithExtensions(command, /*extensions=*/ ImmutableList.of());
  }

  /**
   * Creates a new command environment with additional proto extensions as if they were passed to
   * the blaze server.
   */
  public final CommandEnvironment newCommandWithExtensions(
      Class<? extends BlazeCommand> command, List<Message> extensions) throws Exception {
    additionalOptionsClasses.addAll(
        BlazeCommandUtils.getOptions(
            command, runtime.getBlazeModules(), runtime.getRuleClassProvider()));
    initializeOptionsParser();
    commandCreated = true;
    if (env != null) {
      runtime.afterCommand(env, BlazeCommandResult.success());
    }

    checkNotNull(
        optionsParser,
        "The options parser must be initialized before creating a new command environment");

    env =
        runtime
            .getWorkspace()
            .initCommand(
                command.getAnnotation(Command.class),
                optionsParser,
                new ArrayList<>(),
                0L,
                0L,
                extensions.stream().map(Any::pack).collect(toImmutableList()));
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
    optionsToParse = new ImmutableList.Builder<>();
  }

  public void addOptions(String... args) {
    addOptions(ImmutableList.copyOf(args));
  }

  public void addOptions(List<String> args) {
    optionsToParse.addAll(args);
  }

  public ImmutableList<String> getOptions() {
    return optionsToParse.build();
  }

  public <O extends OptionsBase> O getOptions(Class<O> optionsClass) {
    return optionsParser.getOptions(optionsClass);
  }

  public void addOptionsClass(Class<? extends OptionsBase> optionsClass) {
    additionalOptionsClasses.add(optionsClass);
  }

  void finalizeBuildResult(@SuppressWarnings("unused") BuildResult request) {}

  /**
   * Initializes a new options parser, parsing all the options set by {@link
   * #addOptions(String...)}.
   */
  public void initializeOptionsParser() throws OptionsParsingException {
    // Create the options parser and parse all the options collected so far
    optionsParser = createOptionsParser();
    optionsParser.parse(optionsToParse.build());
    // Enforce the test invocation policy once the options have been added
    enforceTestInvocationPolicy(optionsParser);
  }

  public void executeBuild(List<String> targets) throws Exception {
    if (!commandCreated) {
      // If you didn't create a command we do it for you
      newCommand();
    }
    commandCreated = false;
    BuildTool buildTool = new BuildTool(env);
    try (OutErr.SystemPatcher systemOutErrPatcher =
        env.getReporter().getOutErr().getSystemPatcher()) {
      Profiler.instance()
          .start(
              /*profiledTasks=*/ ImmutableSet.of(),
              /*stream=*/ null,
              /*format=*/ null,
              /*outputBase=*/ null,
              /*buildID=*/ null,
              /*recordAllDurations=*/ false,
              new JavaClock(),
              /*execStartTimeNanos=*/ 42,
              /*enabledCpuUsageProfiling=*/ false,
              /*slimProfile=*/ false,
              /*includePrimaryOutput=*/ false,
              /*includeTargetLabel=*/ false);

      // This cannot go into newCommand, because we hook up the EventCollectionApparatus as a
      // module, and after that ran, further changes to the apparatus aren't reflected on the
      // reporter.
      for (BlazeModule module : runtime.getBlazeModules()) {
        module.beforeCommand(env);
      }
      EventBus eventBus = env.getEventBus();
      for (Object subscriber : eventBusSubscribers) {
        eventBus.register(subscriber);
      }

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
        this.configurations = lastResult.getBuildConfigurationCollection();
        finalizeBuildResult(lastResult);
        buildTool.stopRequest(
            lastResult,
            crash != null ? crash.getThrowable() : null,
            detailedExitCode,
            /*startSuspendCount=*/ 0);
        getSkyframeExecutor().notifyCommandComplete(env.getReporter());
        if (crash != null) {
          runtime
              .getBugReporter()
              .handleCrash(crash, CrashContext.keepAlive().reportingTo(env.getReporter()));
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

  BuildRequest createRequest(String commandName, List<String> targets) {

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

  public BuildRequest getLastRequest() {
    return lastRequest;
  }

  public BuildResult getLastResult() {
    return lastResult;
  }

  public BuildConfigurationCollection getConfigurationCollection() {
    return configurations;
  }

  public ImmutableSet<ConfiguredTarget> getTopLevelTargets() {
    return topLevelTargets;
  }
}
