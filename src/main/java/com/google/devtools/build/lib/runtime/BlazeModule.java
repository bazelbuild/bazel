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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
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
  public void globalInit(OptionsProvider startupOptions) throws AbruptExitException {
  }

  /**
   * Returns the file system implementation used by Bazel. It is an error if more than one module
   * returns a file system. If all return null, the default unix file system is used.
   *
   * <p>This method will be called at the beginning of Bazel startup (in-between {@link #globalInit}
   * and {@link #blazeStartup}).
   *
   * @param startupOptions the server's startup options
   */
  public FileSystem getFileSystem(OptionsProvider startupOptions) {
    return null;
  }

  /**
   * Called when Bazel starts up after {@link #getStartupOptions}, {@link #globalInit}, and
   * {@link #getFileSystem}.
   *
   * @param startupOptions the server's startup options
   * @param versionInfo the Bazel version currently running
   * @param instanceId the id of the current Bazel server
   * @param directories the install directory
   * @param clock the clock
   *
   * @throws AbruptExitException to shut down the server immediately
   */
  public void blazeStartup(OptionsProvider startupOptions,
      BlazeVersionInfo versionInfo, UUID instanceId, ServerDirectories directories,
      Clock clock) throws AbruptExitException {
  }

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
  public void serverInit(OptionsProvider startupOptions, ServerBuilder builder)
      throws AbruptExitException {
  }

  /**
   * Sets up the configured rule class provider, which contains the built-in rule classes, aspects,
   * configuration fragments, and other things; called during Blaze startup (after
   * {@link #blazeStartup}).
   * 
   * <p>Bazel only creates one provider per server, so it is not possible to have different contents
   * for different workspaces.
   *
   * @param builder the configured rule class provider builder
   */
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
  }

  /**
   * Called when Bazel initializes a new workspace; this is only called after {@link #serverInit},
   * and only if the server initialization was successful. Modules can override this method to
   * affect how the workspace is configured.
   *
   * @param directories the workspace directories
   * @param builder the workspace builder
   */
  public void workspaceInit(BlazeDirectories directories, WorkspaceBuilder builder) {
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
  public void handleOptions(OptionsProvider optionsProvider) {}

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
   * Called when Bazel initializes the action execution subsystem. This is called once per build if
   * action execution is enabled. Modules can override this method to affect how execution is
   * performed.
   *
   * @param env the command environment
   * @param request the build request
   * @param builder the builder to add action context providers and consumers to
   */
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
  }

  /**
   * Called after each command.
   */
  public void afterCommand() {
  }

  /**
   * Called when Blaze shuts down.
   *
   * <p>If you are also implementing {@link #shutdownOnCrash()}, consider putting the common
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
   * Perform module specific check of current command environment.
   */
  public void checkEnvironment(CommandEnvironment env) {
  }

  /**
   * Returns a helper that the {@link PackageFactory} will use during package loading. If the module
   * does not provide any helper, it should return null. Note that only one helper per Bazel/Blaze
   * runtime is allowed.
   */
  public Package.Builder.Helper getPackageBuilderHelper(RuleClassProvider ruleClassProvider,
      FileSystem fs) {
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
  public CoverageReportActionFactory getCoverageReportFactory(OptionsClassProvider commandOptions) {
    return null;
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
     * Exits Blaze as early as possible by sending an interrupt to the command's main thread.
     */
    void exit(AbruptExitException exception);
  }
}
