// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.DefaultsPackage;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Encapsulates the state needed for a single command. The environment is dropped when the current
 * command is done and all corresponding objects are garbage collected.
 */
public final class CommandEnvironment {
  private final BlazeRuntime runtime;

  private final UUID commandId;  // Unique identifier for the command being run
  private final Reporter reporter;
  private final EventBus eventBus;
  private final BlazeModule.ModuleEnvironment blazeModuleEnvironment;
  private final Map<String, String> clientEnv = new HashMap<>();

  private String outputFileSystem;

  private AtomicReference<AbruptExitException> pendingException = new AtomicReference<>();

  private class BlazeModuleEnvironment implements BlazeModule.ModuleEnvironment {
    @Override
    public Path getFileFromWorkspace(Label label)
        throws NoSuchThingException, InterruptedException, IOException {
      Target target = getPackageManager().getTarget(reporter, label);
      return (runtime.getOutputService() != null)
          ? runtime.getOutputService().stageTool(target)
          : target.getPackage().getPackageDirectory().getRelative(target.getName());
    }

    @Override
    public void exit(AbruptExitException exception) {
      pendingException.compareAndSet(null, exception);
    }
  }

  public CommandEnvironment(BlazeRuntime runtime, UUID commandId, Reporter reporter,
      EventBus eventBus) {
    this.runtime = runtime;
    this.commandId = commandId;
    this.reporter = reporter;
    this.eventBus = eventBus;
    this.blazeModuleEnvironment = new BlazeModuleEnvironment();
  }

  public BlazeRuntime getRuntime() {
    return runtime;
  }

  public BlazeDirectories getDirectories() {
    return runtime.getDirectories();
  }

  /**
   * Returns the reporter for events.
   */
  public Reporter getReporter() {
    return reporter;
  }

  public EventBus getEventBus() {
    return eventBus;
  }

  public BlazeModule.ModuleEnvironment getBlazeModuleEnvironment() {
    return blazeModuleEnvironment;
  }

  /**
   * Return an unmodifiable view of the blaze client's environment when it invoked the current
   * command.
   */
  public Map<String, String> getClientEnv() {
    return Collections.unmodifiableMap(clientEnv);
  }

  @VisibleForTesting
  void updateClientEnv(List<Map.Entry<String, String>> clientEnvList, boolean ignoreClientEnv) {
    Preconditions.checkState(clientEnv.isEmpty());

    Collection<Map.Entry<String, String>> env =
        ignoreClientEnv ? System.getenv().entrySet() : clientEnvList;
    for (Map.Entry<String, String> entry : env) {
      clientEnv.put(entry.getKey(), entry.getValue());
    }
  }

  public PackageManager getPackageManager() {
    return runtime.getPackageManager();
  }

  public BuildView getView() {
    return runtime.getView();
  }

  /**
   * Returns the UUID that Blaze uses to identify everything logged from the current build command.
   * It's also used to invalidate Skyframe nodes that are specific to a certain invocation, such as
   * the build info.
   */
  public UUID getCommandId() {
    return commandId;
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return runtime.getSkyframeExecutor();
  }

  public Path getWorkingDirectory() {
    return runtime.getWorkingDirectory();
  }

  public ActionCache getPersistentActionCache() throws IOException {
    return runtime.getPersistentActionCache(reporter);
  }

  /**
   * This method only exists for the benefit of InfoCommand, which needs to construct a {@link
   * BuildConfigurationCollection} without running a full loading phase. Don't add any more clients;
   * instead, we should change info so that it doesn't need the configuration.
   */
  public BuildConfigurationCollection getConfigurations(OptionsProvider optionsProvider)
      throws InvalidConfigurationException, InterruptedException {
    BuildOptions buildOptions = runtime.createBuildOptions(optionsProvider);
    boolean keepGoing = optionsProvider.getOptions(BuildView.Options.class).keepGoing;
    boolean loadingSuccessful =
        runtime.getLoadingPhaseRunner().loadForConfigurations(reporter,
            ImmutableSet.copyOf(buildOptions.getAllLabels().values()),
            keepGoing);
    if (!loadingSuccessful) {
      throw new InvalidConfigurationException("Configuration creation failed");
    }
    return getSkyframeExecutor().createConfigurations(runtime.getConfigurationFactory(),
        buildOptions, runtime.getDirectories(), ImmutableSet.<String>of(), keepGoing);
  }

  /**
   * Hook method called by the BlazeCommandDispatcher right before the dispatch
   * of each command ends (while its outcome can still be modified).
   */
  ExitCode precompleteCommand(ExitCode originalExit) {
    eventBus.post(new CommandPrecompleteEvent(originalExit));
    // If Blaze did not suffer an infrastructure failure, check for errors in modules.
    ExitCode exitCode = originalExit;
    AbruptExitException exception = pendingException.get();
    if (!originalExit.isInfrastructureFailure() && exception != null) {
      exitCode = exception.getExitCode();
    }
    return exitCode;
  }

  /**
   * Throws the exception currently queued by a Blaze module.
   *
   * <p>This should be called as often as is practical so that errors are reported as soon as
   * possible. Ideally, we'd not need this, but the event bus swallows exceptions so we raise
   * the exception this way.
   */
  public void throwPendingException() throws AbruptExitException {
    AbruptExitException exception = pendingException.get();
    if (exception != null) {
      throw exception;
    }
  }

  /**
   * Initializes the package cache using the given options, and syncs the package cache. Also
   * injects a defaults package using the options for the {@link BuildConfiguration}.
   *
   * @see DefaultsPackage
   */
  public void setupPackageCache(PackageCacheOptions packageCacheOptions,
      String defaultsPackageContents) throws InterruptedException, AbruptExitException {
    runtime.setupPackageCache(packageCacheOptions, defaultsPackageContents, commandId);
  }

  public long getCommandStartTime() {
    return runtime.getCommandStartTime();
  }

  void setOutputFileSystem(String outputFileSystem) {
    this.outputFileSystem = outputFileSystem;
  }

  public String getOutputFileSystem() {
    return outputFileSystem;
  }
}
