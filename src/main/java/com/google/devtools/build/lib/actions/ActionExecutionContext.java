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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.common.options.OptionsClassProvider;
import java.io.Closeable;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A class that groups services in the scope of the action. Like the FileOutErr object.
 */
public class ActionExecutionContext implements Closeable {

  private final Executor executor;
  private final ActionInputFileCache actionInputFileCache;
  private final ActionInputPrefetcher actionInputPrefetcher;
  private final MetadataHandler metadataHandler;
  private final FileOutErr fileOutErr;
  private final ImmutableMap<String, String> clientEnv;
  private final ArtifactExpander artifactExpander;
  @Nullable
  private final Environment env;

  private ActionExecutionContext(
      Executor executor,
      ActionInputFileCache actionInputFileCache,
      ActionInputPrefetcher actionInputPrefetcher,
      MetadataHandler metadataHandler,
      FileOutErr fileOutErr,
      Map<String, String> clientEnv,
      @Nullable ArtifactExpander artifactExpander,
      @Nullable SkyFunction.Environment env) {
    this.actionInputFileCache = actionInputFileCache;
    this.actionInputPrefetcher = actionInputPrefetcher;
    this.metadataHandler = metadataHandler;
    this.fileOutErr = fileOutErr;
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
    this.executor = executor;
    this.artifactExpander = artifactExpander;
    this.env = env;
  }

  public ActionExecutionContext(
      Executor executor,
      ActionInputFileCache actionInputFileCache,
      ActionInputPrefetcher actionInputPrefetcher,
      MetadataHandler metadataHandler,
      FileOutErr fileOutErr,
      Map<String, String> clientEnv,
      ArtifactExpander artifactExpander) {
    this(
        executor,
        actionInputFileCache,
        actionInputPrefetcher,
        metadataHandler,
        fileOutErr,
        clientEnv,
        artifactExpander,
        null);
  }

  public static ActionExecutionContext forInputDiscovery(
      Executor executor,
      ActionInputFileCache actionInputFileCache,
      ActionInputPrefetcher actionInputPrefetcher,
      MetadataHandler metadataHandler,
      FileOutErr fileOutErr,
      Map<String, String> clientEnv,
      Environment env) {
    return new ActionExecutionContext(
        executor,
        actionInputFileCache,
        actionInputPrefetcher,
        metadataHandler,
        fileOutErr,
        clientEnv,
        null,
        env);
  }

  public ActionInputPrefetcher getActionInputPrefetcher() {
    return actionInputPrefetcher;
  }

  public ActionInputFileCache getActionInputFileCache() {
    return actionInputFileCache;
  }

  public MetadataHandler getMetadataHandler() {
    return metadataHandler;
  }

  public Path getExecRoot() {
    return executor.getExecRoot();
  }

  /**
   * Returns whether failures should have verbose error messages.
   */
  public boolean getVerboseFailures() {
    return executor.getVerboseFailures();
  }

  /**
   * Returns the command line options of the Blaze command being executed.
   */
  public OptionsClassProvider getOptions() {
    return executor.getOptions();
  }

  public Clock getClock() {
    return executor.getClock();
  }

  public EventBus getEventBus() {
    return executor.getEventBus();
  }

  public EventHandler getEventHandler() {
    return executor.getEventHandler();
  }

  /**
   * Looks up and returns an action context implementation of the given interface type.
   */
  public <T extends ActionContext> T getContext(Class<? extends T> type) {
    return executor.getContext(type);
  }

  /**
   * Returns the action context implementation for spawn actions with a given mnemonic.
   */
  public SpawnActionContext getSpawnActionContext(String mnemonic) {
    return executor.getSpawnActionContext(mnemonic);
  }

  /**
   * Whether this Executor reports subcommands. If not, reportSubcommand has no effect.
   * This is provided so the caller of reportSubcommand can avoid wastefully constructing the
   * subcommand string.
   */
  public boolean reportsSubcommands() {
    return executor.reportsSubcommands();
  }

  /**
   * Report a subcommand event to this Executor's Reporter and, if action
   * logging is enabled, post it on its EventBus.
   */
  public void reportSubcommand(Spawn spawn) {
    String reason;
    ActionOwner owner = spawn.getResourceOwner().getOwner();
    if (owner == null) {
      reason = spawn.getResourceOwner().prettyPrint();
    } else {
      reason = Label.print(owner.getLabel())
          + " [" + spawn.getResourceOwner().prettyPrint() + "]";
    }
    String message = Spawns.asShellCommand(spawn, getExecRoot());
    getEventHandler().handle(Event.of(EventKind.SUBCOMMAND, null, "# " + reason + "\n" + message));
  }

  public ImmutableMap<String, String> getClientEnv() {
    return clientEnv;
  }

  public ArtifactExpander getArtifactExpander() {
    return artifactExpander;
  }

  /**
   * Provide that {@code FileOutErr} that the action should use for redirecting the output and error
   * stream.
   */
  public FileOutErr getFileOutErr() {
    return fileOutErr;
  }

  /**
   * Provides a mechanism for the action to request values from Skyframe while it discovers inputs.
   */
  public Environment getEnvironmentForDiscoveringInputs() {
    return Preconditions.checkNotNull(env);
  }

  @Override
  public void close() throws IOException {
    fileOutErr.close();
  }

  /**
   * Allows us to create a new context that overrides the FileOutErr with another one. This is
   * useful for muting the output for example.
   */
  public ActionExecutionContext withFileOutErr(FileOutErr fileOutErr) {
    return new ActionExecutionContext(
        executor,
        actionInputFileCache,
        actionInputPrefetcher,
        metadataHandler,
        fileOutErr,
        clientEnv,
        artifactExpander,
        env);
  }
}
