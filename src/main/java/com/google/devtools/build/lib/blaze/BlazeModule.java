// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.blaze;

import com.google.common.base.Predicate;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.packages.MakeEnvironment;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.PackageArgument;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.view.WorkspaceStatusAction;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

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
   * The factory yielded by the supplier will be checked with
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

  /**
   * Services provided for Blaze modules via BlazeRuntime.
   */
  public interface ModuleEnvironment {
    /**
     * Gets a file from the depot based on its label and returns the {@link Path} where it can
     * be found.
     */
    Path getFileFromDepot(Label label)
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
  public void beforeCommand(BlazeRuntime blazeRuntime, Command command)
      throws AbruptExitException {
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
   * Returns the action context provider the module contributes to Blaze, if any.
   *
   * <p>This method will be called at the beginning of the execution phase, e.g. of the
   * "blaze build" command.
   */
  public ActionContextProvider getActionContextProvider() {
    return null;
  }

  /**
   * Returns the action context consumer that pulls in action contexts required by this module,
   * if any.
   *
   * <p>This method will be called at the beginning of the execution phase, e.g. of the
   * "blaze build" command.
   */
  public ActionContextConsumer getActionContextConsumer() {
    return null;
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
    return new PackageFactory.EnvironmentExtension() {
      @Override
      public void update(
          Environment environment, MakeEnvironment.Builder pkgMakeEnv, Label buildFileLabel) {
      }

      @Override
      public Iterable<PackageArgument<?>> getPackageArguments() {
        return ImmutableList.of();
      }
    };
  }
}
