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

import com.google.common.base.Predicate;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.rules.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

import javax.annotation.Nullable;

/**
 * A module Blaze can load at the beginning of its execution. Modules are supplied with extension
 * points to augment the functionality at specific, well-defined places.
 *
 * <p>The constructors of individual Blaze modules should be empty. All work should be done in the
 * methods (e.g. {@link #blazeStartup}).
 */
public abstract class BlazeModule {

  /**
   * Returns the extra startup options this module contributes.
   *
   * <p>This method will be called at the beginning of Blaze startup (before #blazeStartup).
   */
  public Iterable<Class<? extends OptionsBase>> getStartupOptions() {
    return ImmutableList.of();
  }

  /**
   * Called before {@link #getFileSystem} and {@link #blazeStartup}.
   *
   * <p>This method will be called at the beginning of Blaze startup.
   */
  @SuppressWarnings("unused")
  public void globalInit(OptionsProvider startupOptions) throws AbruptExitException {
  }

  /**
   * Returns the file system implementation used by Blaze. It is an error if more than one module
   * returns a file system. If all return null, the default unix file system is used.
   *
   * <p>This method will be called at the beginning of Blaze startup (in-between #globalInit and
   * #blazeStartup).
   */
  @SuppressWarnings("unused")
  public FileSystem getFileSystem(OptionsProvider startupOptions, PathFragment outputPath) {
    return null;
  }

  /**
   * Called when Blaze starts up.
   */
  @SuppressWarnings("unused")
  public void blazeStartup(OptionsProvider startupOptions,
      BlazeVersionInfo versionInfo, UUID instanceId, BlazeDirectories directories,
      Clock clock) throws AbruptExitException {
  }

  /**
   * May yield a supplier that provides factories for the Preprocessor to apply. Only one of the
   * configured modules may return non-null.
   *
   * <p>The factory yielded by the supplier will be checked with
   * {@link Preprocessor.Factory#isStillValid} at the beginning of each incremental build. This
   * allows modules to have preprocessors customizable by flags.
   *
   * <p>This method will be called during Blaze startup (after #blazeStartup).
   */
  public Preprocessor.Factory.Supplier getPreprocessorFactorySupplier() {
    return null;
  }

  /**
   * Adds the rule classes supported by this module.
   *
   * <p>This method will be called during Blaze startup (after #blazeStartup).
   */
  @SuppressWarnings("unused")
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
  }

  /**
   * Returns the list of commands this module contributes to Blaze.
   *
   * <p>This method will be called during Blaze startup (after #blazeStartup).
   */
  public Iterable<? extends BlazeCommand> getCommands() {
    return ImmutableList.of();
  }

  /**
   * Returns the list of query output formatters this module provides.
   *
   * <p>This method will be called during Blaze startup (after #blazeStartup).
   */
  public Iterable<OutputFormatter> getQueryOutputFormatters() {
    return ImmutableList.of();
  }

  /**
   * Returns the {@link DiffAwareness} strategies this module contributes. These will be used to
   * determine which files, if any, changed between Blaze commands.
   *
   * <p>This method will be called during Blaze startup (after #blazeStartup).
   */
  @SuppressWarnings("unused")
  public Iterable<? extends DiffAwareness.Factory> getDiffAwarenessFactories(boolean watchFS) {
    return ImmutableList.of();
  }

  /**
   * Returns the workspace status action factory contributed by this module.
   *
   * <p>There should always be exactly one of these in a Blaze instance.
   */
  public WorkspaceStatusAction.Factory getWorkspaceStatusActionFactory() {
    return null;
  }

  /**
   * PlatformSet is a group of platforms characterized by a regular expression.  For example, the
   * entry "oldlinux": "i[34]86-libc[345]-linux" might define a set of platforms representing
   * certain older linux releases.
   *
   * <p>Platform-set names are used in BUILD files in the third argument to <tt>vardef</tt>, to
   * define per-platform tweaks to variables such as CFLAGS.
   *
   * <p>vardef is a legacy mechanism: it needs explicit support in the rule implementations,
   * and cannot express conditional dependencies, only conditional attribute values. This
   * mechanism will be supplanted by configuration dependent attributes, and its effect can
   * usually also be achieved with abi_deps.
   *
   * <p>This method will be called during Blaze startup (after #blazeStartup).
   */
  public Map<String, String> getPlatformSetRegexps() {
    return ImmutableMap.<String, String>of();
  }

  public Iterable<SkyValueDirtinessChecker> getCustomDirtinessCheckers() {
    return ImmutableList.of();
  }

  /**
   * Services provided for Blaze modules via BlazeRuntime.
   */
  public interface ModuleEnvironment {
    /**
     * Gets a file from the depot based on its label and returns the {@link Path} where it can
     * be found.
     */
    Path getFileFromWorkspace(Label label)
        throws NoSuchThingException, InterruptedException, IOException;

    /**
     * Exits Blaze as early as possible. This is currently a hack and should only be called in
     * event handlers for {@code BuildStartingEvent}, {@code GotOptionsEvent} and
     * {@code LoadingPhaseCompleteEvent}.
     */
    void exit(AbruptExitException exception);
  }

  /**
   * Called before each command.
   */
  @SuppressWarnings("unused")
  public void beforeCommand(Command command, CommandEnvironment env) throws AbruptExitException {
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
   * Does any handling of options needed by the command.
   *
   * <p>This method will be called at the beginning of each command (after #beforeCommand).
   */
  @SuppressWarnings("unused")
  public void handleOptions(OptionsProvider optionsProvider) {
  }

  /**
   * Returns the extra options this module contributes to a specific command.
   *
   * <p>This method will be called at the beginning of each command (after #beforeCommand).
   */
  @SuppressWarnings("unused")
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of();
  }

  /**
   * Returns a map of option categories to descriptive strings. This is used by {@code HelpCommand}
   * to show a more readable list of flags.
   */
  public Map<String, String> getOptionCategories() {
    return ImmutableMap.of();
  }

  /**
   * A item that is returned by "blaze info".
   */
  public interface InfoItem {
    /**
     * The name of the info key.
     */
    String getName();

    /**
     * The help description of the info key.
     */
    String getDescription();

    /**
     * Whether the key is printed when "blaze info" is invoked without arguments.
     *
     * <p>This is usually true for info keys that take multiple lines, thus, cannot really be
     * included in the output of argumentless "blaze info".
     */
    boolean isHidden();

    /**
     * Returns the value of the info key. The return value is directly printed to stdout.
     */
    byte[] get(Supplier<BuildConfiguration> configurationSupplier) throws AbruptExitException;
  }

  /**
   * Returns the additional information this module provides to "blaze info".
   *
   * <p>This method will be called at the beginning of each "blaze info" command (after
   * #beforeCommand).
   */
  public Iterable<InfoItem> getInfoItems() {
    return ImmutableList.of();
  }

  /**
   * Returns the list of query functions this module provides to "blaze query".
   *
   * <p>This method will be called at the beginning of each "blaze query" command (after
   * #beforeCommand).
   */
  public Iterable<QueryFunction> getQueryFunctions() {
    return ImmutableList.of();
  }

  /**
   * Returns the action context providers the module contributes to Blaze, if any.
   *
   * <p>This method will be called at the beginning of the execution phase, e.g. of the
   * "blaze build" command.
   */
  public Iterable<ActionContextProvider> getActionContextProviders() {
    return ImmutableList.of();
  }

  /**
   * Returns the action context consumers that pulls in action contexts required by this module,
   * if any.
   *
   * <p>This method will be called at the beginning of the execution phase, e.g. of the
   * "blaze build" command.
   */
  public Iterable<ActionContextConsumer> getActionContextConsumers() {
    return ImmutableList.of();
  }

  /**
   * Called after each command.
   */
  public void afterCommand() {
  }

  /**
   * Called when Blaze shuts down.
   */
  public void blazeShutdown() {
  }

  /**
   * Action inputs are allowed to be missing for all inputs where this predicate returns true.
   */
  public Predicate<PathFragment> getAllowedMissingInputs() {
    return null;
  }

  /**
   * Perform module specific check of current command environment.
   */
  public void checkEnvironment(CommandEnvironment env) {
  }

  /**
   * Optionally specializes the cache that ensures source files are looked at just once during
   * a build. Only one module may do so.
   */
  public ActionInputFileCache createActionInputCache(String cwd, FileSystem fs) {
    return null;
  }

  /**
   * Returns the extensions this module contributes to the global namespace of the BUILD language.
   */
  public PackageFactory.EnvironmentExtension getPackageEnvironmentExtension() {
    return new PackageFactory.EmptyEnvironmentExtension();
  }

  /**
   * Returns a factory for creating {@link SkyframeExecutor} objects. If the module does not
   * provide any SkyframeExecutorFactory, it returns null. Note that only one factory per
   * Bazel/Blaze runtime is allowed.
   */
  public SkyframeExecutorFactory getSkyframeExecutorFactory() {
    return null;
  }

  /** Returns a map of "extra" SkyFunctions for SkyValues that this module may want to build. */
  public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(BlazeDirectories directories) {
    return ImmutableMap.of();
  }

  /**
   * Returns the extra precomputed values that the module makes available in Skyframe.
   *
   * <p>This method is called once per Blaze instance at the very beginning of its life.
   * If it creates the injected values by using a {@code com.google.common.base.Supplier},
   * that supplier is asked for the value it contains just before the loading phase begins. This
   * functionality can be used to implement precomputed values that are not constant during the
   * lifetime of a Blaze instance (naturally, they must be constant over the course of a build)
   *
   * <p>The following things must be done in order to define a new precomputed values:
   * <ul>
   * <li> Create a public static final variable of type
   * {@link com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed}
   * <li> Set its value by adding an {@link Injected} in this method (it can be created using the
   * aforementioned variable and the value or a supplier of the value)
   * <li> Reference the value in Skyframe functions by calling get {@code get} method on the
   * {@link com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed} variable. This
   * will never return null, because its value will have been injected before most of the Skyframe
   * values are computed.
   * </ul>
   */
  public Iterable<Injected> getPrecomputedSkyframeValues() {
    return ImmutableList.of();
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
   * Optionally returns a factory to create coverage report actions.
   */
  @Nullable
  public CoverageReportActionFactory getCoverageReportFactory() {
    return null;
  }

  /**
   * Optionally returns the invocation policy to override options in blaze.
   */
  @Nullable
  public InvocationPolicy getInvocationPolicy() {
    return null;
  }
}
