// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.SubscriberExceptionContext;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageValidator;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.TopDownActionCache;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DefaultHashFunctionNotSetException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.OptionsProvider;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * A module Bazel can load at the beginning of its execution. Modules are supplied with extension
 * points to augment the functionality at specific, well-defined places.
 *
 * <p>The constructors of individual Bazel modules should be empty. All work should be done in the
 * methods (e.g. {@link #blazeStartup}).
 */
public abstract class BlazeModule {

  /**
   * Returns the extra startup options this module contributes.
   *
   * <p>This method will be called at the beginning of Blaze startup (before {@link #globalInit}).
   * The startup options need to be parsed very early in the process, which requires this to be
   * separate from {@link #serverInit}.
   */
  public Iterable<Class<? extends OptionsBase>> getStartupOptions() {
    return ImmutableList.of();
  }

  /**
   * Called at the beginning of Bazel startup, before {@link #getFileSystem} and
   * {@link #blazeStartup}.
   *
   * @param startupOptions the server's startup options
   *
   * @throws AbruptExitException to shut down the server immediately
   */
  public void globalInit(OptionsParsingResult startupOptions) throws AbruptExitException {
  }

  /**
   * Returns the file system implementation used by Bazel. It is an error if more than one module
   * returns a file system. If all return null, the default unix file system is used.
   *
   * <p>This method will be called at the beginning of Bazel startup (in-between {@link #globalInit}
   * and {@link #blazeStartup}).
   *
   * @param startupOptions the server's startup options
   * @param realExecRootBase absolute path fragment of the actual, underlying execution root
   */
  public ModuleFileSystem getFileSystem(
      OptionsParsingResult startupOptions, PathFragment realExecRootBase)
      throws AbruptExitException, DefaultHashFunctionNotSetException {
    return null;
  }

  /**
   * Returns the {@link TopDownActionCache} used by Bazel. It is an error if more than one module
   * returns a top-down action cache. If all modules return null, there will be no top-down caching.
   *
   * <p>This method will be called at the beginning of Bazel startup (in-between {@link #globalInit}
   * and {@link #blazeStartup}).
   */
  public TopDownActionCache getTopDownActionCache() {
    return null;
  }

  /** Tuple returned by {@link #getFileSystem}. */
  @AutoValue
  public abstract static class ModuleFileSystem {
    public abstract FileSystem fileSystem();

    /** Non-null if this filesystem virtualizes the execroot folder. */
    @Nullable
    public abstract Path virtualExecRootBase();

    public static ModuleFileSystem create(
        FileSystem fileSystem, @Nullable Path virtualExecRootBase) {
      return new AutoValue_BlazeModule_ModuleFileSystem(fileSystem, virtualExecRootBase);
    }

    public static ModuleFileSystem create(FileSystem fileSystem) {
      return create(fileSystem, null);
    }
  }

  /**
   * Returns handler for {@link com.google.common.eventbus.EventBus} subscriber and async thread
   * exceptions. For async thread exceptions, {@link
   * SubscriberExceptionHandler#handleException(Throwable, SubscriberExceptionContext)} will be
   * called with null {@link SubscriberExceptionContext}. If all modules return null, a handler that
   * crashes on all async exceptions and files bug reports for all EventBus subscriber exceptions
   * will be used.
   */
  public SubscriberExceptionHandler getEventBusAndAsyncExceptionHandler() {
    return null;
  }

  /**
   * Called when Bazel starts up after {@link #getStartupOptions}, {@link #globalInit}, and {@link
   * #getFileSystem}.
   *
   * @param startupOptions the server's startup options
   * @param versionInfo the Bazel version currently running
   * @param instanceId the id of the current Bazel server
   * @param fileSystem
   * @param directories the install directory
   * @param clock the clock
   * @throws AbruptExitException to shut down the server immediately
   */
  public void blazeStartup(
      OptionsParsingResult startupOptions,
      BlazeVersionInfo versionInfo,
      UUID instanceId,
      FileSystem fileSystem,
      ServerDirectories directories,
      Clock clock)
      throws AbruptExitException {}

  /**
   * Called to initialize a new server ({@link BlazeRuntime}). Modules can override this method to
   * affect how the server is configured. This is called after the startup options have been
   * collected and parsed, and after the file system was setup.
   *
   * @param startupOptions the server startup options
   * @param builder builder class that collects the server configuration
   *
   * @throws AbruptExitException to shut down the server immediately
   */
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder)
      throws AbruptExitException {
  }

  /**
   * Sets up the configured rule class provider, which contains the built-in rule classes, aspects,
   * configuration fragments, and other things; called during Blaze startup (after {@link
   * #blazeStartup}).
   *
   * <p>Bazel only creates one provider per server, so it is not possible to have different contents
   * for different workspaces.
   *
   * @param builder the configured rule class provider builder
   */
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {}

  /**
   * Called when Bazel initializes a new workspace; this is only called after {@link #serverInit},
   * and only if the server initialization was successful. Modules can override this method to
   * affect how the workspace is configured.
   *
   * @param runtime the blaze runtime
   * @param directories the workspace directories
   * @param builder the workspace builder
   */
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
  }

  /**
   * Called to notify modules that the given command is about to be executed. This allows capturing
   * the {@link com.google.common.eventbus.EventBus}, {@link Command}, or {@link
   * OptionsParsingResult}.
   *
   * @param env the command
   * @throws AbruptExitException modules can throw this exception to abort the command
   */
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {}

  /**
   * Returns additional listeners to the console output stream. Called at the beginning of each
   * command (after #beforeCommand).
   */
  @SuppressWarnings("unused")
  @Nullable
  public OutErr getOutputListener() {
    return null;
  }

  /**
   * Returns the output service to be used. It is an error if more than one module returns an
   * output service.
   *
   * <p>This method will be called at the beginning of each command (after #beforeCommand).
   */
  @SuppressWarnings("unused")
  public OutputService getOutputService() throws AbruptExitException {
    return null;
  }

  /**
   * Returns extra options this module contributes to a specific command. Note that option
   * inheritance applies: if this method returns a non-empty list, then the returned options are
   * added to every command that depends on this command.
   *
   * <p>This method may be called at any time, and the returned value may be cached. Implementations
   * must be thread-safe and never return different lists for the same command object. Typical
   * implementations look like this:
   *
   * <pre>
   * return "build".equals(command.name())
   *     ? ImmutableList.<Class<? extends OptionsBase>>of(MyOptions.class)
   *     : ImmutableList.<Class<? extends OptionsBase>>of();
   * </pre>
   *
   * Note that this example adds options to all commands that inherit from the build command.
   *
   * <p>This method is also used to generate command-line documentation; in order to avoid
   * duplicated options descriptions, this method should never return the same options class for two
   * different commands if one of them inherits the other.
   *
   * <p>If you want to add options to all commands, override {@link #getCommonCommandOptions}
   * instead.
   *
   * @param command the command
   */
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of();
  }

  /**
   * Returns extra options this module contributes to all commands.
   */
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of();
  }

  /**
   * Returns an instance of BuildOptions to be used to create {@link
   * BuildOptions.OptionsDiffForReconstruction} with. Only one installed Module should override
   * this.
   */
  public BuildOptions getDefaultBuildOptions(BlazeRuntime runtime) {
    return null;
  }

  /**
   * Called after Bazel analyzes the build's top-level targets. This is called once per build if
   * --analyze is enabled. Modules can override this to perform extra checks on analysis results.
   *
   * @param env the command environment
   * @param request the build request
   * @param buildOptions the build's top-level options
   * @param configuredTargets the build's requested top-level targets as {@link ConfiguredTarget}s
   */
  public void afterAnalysis(
      CommandEnvironment env,
      BuildRequest request,
      BuildOptions buildOptions,
      Iterable<ConfiguredTarget> configuredTargets,
      ImmutableSet<AspectValue> aspects)
      throws InterruptedException, ViewCreationFailedException {}

  /**
   * Called when Bazel initializes the action execution subsystem. This is called once per build if
   * action execution is enabled. Modules can override this method to affect how execution is
   * performed.
   *
   * @param env the command environment
   * @param request the build request
   * @param builder the builder to add action context providers and consumers to
   */
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder)
      throws ExecutorInitException {}

  /**
   * Registers any action contexts this module provides with the execution phase. They will be
   * available for {@linkplain ActionContext.ActionContextRegistry#getContext querying} to actions
   * and other action contexts.
   *
   * <p>This method is invoked before actions are executed but after {@link #executorInit}.
   *
   * @param registryBuilder builder with which to register action contexts
   * @param env environment for the current command
   * @param buildRequest the current build request
   * @throws ExecutorInitException if there are fatal issues creating or registering action contexts
   */
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest)
      throws ExecutorInitException {}

  /**
   * Registers any spawn strategies this module provides with the execution phase.
   *
   * <p>This method is invoked before actions are executed but after {@link #executorInit}.
   *
   * @param registryBuilder builder with which to register strategies
   * @param env environment for the current command
   * @throws ExecutorInitException if there are fatal issues creating or registering strategies
   */
  public void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder, CommandEnvironment env)
      throws ExecutorInitException {}

  /**
   * Called after each command.
   *
   * @throws AbruptExitException modules can throw this exception to modify the command exit code
   */
  public void afterCommand() throws AbruptExitException {}

  /**
   * Called after {@link #afterCommand()}. This method can be used to close and cleanup resources
   * specific to the command.
   *
   * <p>This method must not throw any exceptions, report any errors or generate any stdout/stderr.
   * Any of the above will make Bazel crash occasionally. Please use {@link #afterCommand()}
   * instead.
   */
  public void commandComplete() {}

  /**
   * Called when Blaze shuts down.
   *
   * <p>If you are also implementing {@link #blazeShutdownOnCrash()}, consider putting the common
   * shutdown code in the latter and calling that other hook from here.
   */
  public void blazeShutdown() {
  }

  /**
   * Called when Blaze shuts down due to a crash.
   *
   * <p>Modules may use this to flush pending state, but they must be careful to only do a minimal
   * number of things. Keep in mind that we are crashing so who knows what state we are in. Modules
   * rarely need to implement this.
   */
  public void blazeShutdownOnCrash() {}

  /**
   * Returns true if the module will arrange for a {@code BuildMetricsEvent} to be posted after the
   * build completes.
   *
   * <p>The Blaze runtime ensures that it has exactly one module for which this method returns true,
   * substituting its own module if none is supplied explicitly.
   *
   * <p>It is an error if multiple modules return true.
   */
  public boolean postsBuildMetricsEvent() {
    return false;
  }

  /**
   * Returns a {@link QueryRuntimeHelper.Factory} that will be used by the query, cquery, and aquery
   * commands.
   *
   * <p>It is an error if multiple modules return non-null values.
   */
  public QueryRuntimeHelper.Factory getQueryRuntimeHelperFactory() {
    return null;
  }

  /**
   * Returns a helper that the {@link PackageFactory} will use during package loading, or null if
   * the module does not provide any helper.
   *
   * <p>Called once during server startup some time after {@link #serverInit}.
   *
   * <p>Note that only one helper per Bazel/Blaze runtime is allowed.
   */
  public Package.Builder.Helper getPackageBuilderHelper(
      ConfiguredRuleClassProvider ruleClassProvider, FileSystem fs) {
    return null;
  }

  /**
   * Returns a {@link PackageValidator} to be used to validate loaded packages, or null if the
   * module does not provide any validator.
   *
   * <p>Called once during server startup some time after {@link #serverInit}.
   *
   * <p>Note that only one helper per Bazel/Blaze runtime is allowed.
   */
  @Nullable
  public PackageValidator getPackageValidator() {
    return null;
  }

  /**
   * Optionally returns a provider for project files that can be used to bundle targets and
   * command-line options.
   */
  @Nullable
  public ProjectFile.Provider createProjectFileProvider() {
    return null;
  }

  /**
   * Optionally returns a factory to create coverage report actions; this is called once per build,
   * such that it can be affected by command options.
   *
   * <p>It is an error if multiple modules return non-null values.
   *
   * @param commandOptions the options for the current command
   */
  @Nullable
  public CoverageReportActionFactory getCoverageReportFactory(OptionsProvider commandOptions) {
    return null;
  }

  /**
   * Services provided for Blaze modules via BlazeRuntime.
   */
  public interface ModuleEnvironment {
    /**
     * Gets a file from the depot based on its label and returns the {@link Path} where it can be
     * found.
     *
     * <p>Returns null when the package designated by the label does not exist.
     */
    @Nullable
    Path getFileFromWorkspace(Label label);

    /**
     * Exits Blaze as early as possible by sending an interrupt to the command's main thread.
     */
    void exit(AbruptExitException exception);
  }

  /**
   * Provides additional precomputed values to inject into the skyframe graph. Called on every
   * command execution.
   */
  public ImmutableList<PrecomputedValue.Injected> getPrecomputedValues() {
    return ImmutableList.of();
  }

  @Override
  public String toString() {
    return this.getClass().getSimpleName();
  }
}
