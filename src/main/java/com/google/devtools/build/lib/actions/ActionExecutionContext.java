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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetExpander;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.common.options.OptionsProvider;
import java.io.Closeable;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/** A class that groups services in the scope of the action. Like the FileOutErr object. */
public class ActionExecutionContext implements Closeable, ActionContext.ActionContextRegistry {

  /** Enum for --subcommands flag */
  public enum ShowSubcommands {
    TRUE(true, false), PRETTY_PRINT(true, true), FALSE(false, false);

    private final boolean shouldShowSubcommands;
    private final boolean prettyPrintArgs;

    ShowSubcommands(boolean shouldShowSubcommands, boolean prettyPrintArgs) {
      this.shouldShowSubcommands = shouldShowSubcommands;
      this.prettyPrintArgs = prettyPrintArgs;
    }
  }

  private final Executor executor;
  private final MetadataProvider actionInputFileCache;
  private final ActionInputPrefetcher actionInputPrefetcher;
  private final ActionKeyContext actionKeyContext;
  private final MetadataHandler metadataHandler;
  private final boolean rewindingEnabled;
  private final LostInputsCheck lostInputsCheck;
  private final FileOutErr fileOutErr;
  private final ExtendedEventHandler eventHandler;
  private final ImmutableMap<String, String> clientEnv;
  private final ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets;
  @Nullable private final ArtifactExpander artifactExpander;
  @Nullable private final Environment env;

  @Nullable private final FileSystem actionFileSystem;
  @Nullable private final Object skyframeDepsResult;

  @Nullable private ImmutableList<FilesetOutputSymlink> outputSymlinks;

  private final ArtifactPathResolver pathResolver;
  private final NestedSetExpander nestedSetExpander;

  private ActionExecutionContext(
      Executor executor,
      MetadataProvider actionInputFileCache,
      ActionInputPrefetcher actionInputPrefetcher,
      ActionKeyContext actionKeyContext,
      MetadataHandler metadataHandler,
      boolean rewindingEnabled,
      LostInputsCheck lostInputsCheck,
      FileOutErr fileOutErr,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets,
      @Nullable ArtifactExpander artifactExpander,
      @Nullable Environment env,
      @Nullable FileSystem actionFileSystem,
      @Nullable Object skyframeDepsResult,
      NestedSetExpander nestedSetExpander) {
    this.actionInputFileCache = actionInputFileCache;
    this.actionInputPrefetcher = actionInputPrefetcher;
    this.actionKeyContext = actionKeyContext;
    this.metadataHandler = metadataHandler;
    this.rewindingEnabled = rewindingEnabled;
    this.lostInputsCheck = lostInputsCheck;
    this.fileOutErr = fileOutErr;
    this.eventHandler = eventHandler;
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
    this.topLevelFilesets = topLevelFilesets;
    this.executor = executor;
    this.artifactExpander = artifactExpander;
    this.env = env;
    this.actionFileSystem = actionFileSystem;
    this.skyframeDepsResult = skyframeDepsResult;
    this.pathResolver = ArtifactPathResolver.createPathResolver(actionFileSystem,
        // executor is only ever null in testing.
        executor == null ? null : executor.getExecRoot());
    this.nestedSetExpander = nestedSetExpander;
  }

  public ActionExecutionContext(
      Executor executor,
      MetadataProvider actionInputFileCache,
      ActionInputPrefetcher actionInputPrefetcher,
      ActionKeyContext actionKeyContext,
      MetadataHandler metadataHandler,
      boolean rewindingEnabled,
      LostInputsCheck lostInputsCheck,
      FileOutErr fileOutErr,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets,
      ArtifactExpander artifactExpander,
      @Nullable FileSystem actionFileSystem,
      @Nullable Object skyframeDepsResult,
      NestedSetExpander nestedSetExpander) {
    this(
        executor,
        actionInputFileCache,
        actionInputPrefetcher,
        actionKeyContext,
        metadataHandler,
        rewindingEnabled,
        lostInputsCheck,
        fileOutErr,
        eventHandler,
        clientEnv,
        topLevelFilesets,
        artifactExpander,
        /*env=*/ null,
        actionFileSystem,
        skyframeDepsResult,
        nestedSetExpander);
  }

  public static ActionExecutionContext forInputDiscovery(
      Executor executor,
      MetadataProvider actionInputFileCache,
      ActionInputPrefetcher actionInputPrefetcher,
      ActionKeyContext actionKeyContext,
      MetadataHandler metadataHandler,
      boolean rewindingEnabled,
      LostInputsCheck lostInputsCheck,
      FileOutErr fileOutErr,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      Environment env,
      @Nullable FileSystem actionFileSystem,
      NestedSetExpander nestedSetExpander) {
    return new ActionExecutionContext(
        executor,
        actionInputFileCache,
        actionInputPrefetcher,
        actionKeyContext,
        metadataHandler,
        rewindingEnabled,
        lostInputsCheck,
        fileOutErr,
        eventHandler,
        clientEnv,
        ImmutableMap.of(),
        /*artifactExpander=*/ null,
        env,
        actionFileSystem,
        /*skyframeDepsResult=*/ null,
        nestedSetExpander);
  }

  public ActionInputPrefetcher getActionInputPrefetcher() {
    return actionInputPrefetcher;
  }

  public MetadataProvider getMetadataProvider() {
    return actionInputFileCache;
  }

  public MetadataHandler getMetadataHandler() {
    return metadataHandler;
  }

  public FileSystem getFileSystem() {
    if (actionFileSystem != null) {
      return actionFileSystem;
    }
    return executor.getFileSystem();
  }

  public Path getExecRoot() {
    return actionFileSystem != null
        ? actionFileSystem.getPath(executor.getExecRoot().asFragment())
        : executor.getExecRoot();
  }

  @Nullable
  public FileSystem getActionFileSystem() {
    return actionFileSystem;
  }

  public boolean isRewindingEnabled() {
    return rewindingEnabled;
  }

  public void checkForLostInputs() throws LostInputsActionExecutionException {
    lostInputsCheck.checkForLostInputs();
  }

  /**
   * Returns the path for an ActionInput.
   *
   * <p>Notably, in the future, we want any action-scoped artifacts to resolve paths using this
   * method instead of {@link Artifact#getPath} because that does not allow filesystem injection.
   *
   * <p>TODO(shahan): cleanup {@link Action}-scoped references to {@link Artifact#getPath} and
   * {@link Artifact#getRoot}.
   */
  public Path getInputPath(ActionInput input) {
    return pathResolver.toPath(input);
  }

  public Root getRoot(Artifact artifact) {
    return pathResolver.transformRoot(artifact.getRoot().getRoot());
  }

  public ArtifactPathResolver getPathResolver() {
    return pathResolver;
  }

  /** Returns whether failures for {@code failedLabel} should have verbose error messages. */
  public boolean showVerboseFailures(Label failedLabel) {
    return executor.getVerboseFailuresPredicate().test(failedLabel);
  }

  /**
   * Returns the command line options of the Blaze command being executed.
   */
  public OptionsProvider getOptions() {
    return executor.getOptions();
  }

  public Clock getClock() {
    return executor.getClock();
  }

  public ExtendedEventHandler getEventHandler() {
    return eventHandler;
  }

  public ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> getTopLevelFilesets() {
    return topLevelFilesets;
  }

  @Nullable
  public ImmutableList<FilesetOutputSymlink> getOutputSymlinks() {
    return outputSymlinks;
  }

  public void setOutputSymlinks(ImmutableList<FilesetOutputSymlink> outputSymlinks) {
    Preconditions.checkState(
        this.outputSymlinks == null,
        "Unexpected reassignment of the outputSymlinks of a Fileset from\n:%s to:\n%s",
        this.outputSymlinks,
        outputSymlinks);
    this.outputSymlinks = outputSymlinks;
  }

  @Override
  @Nullable
  public <T extends ActionContext> T getContext(Class<T> type) {
    return executor.getContext(type);
  }

  /**
   * Report a subcommand event to this Executor's Reporter and, if action
   * logging is enabled, post it on its EventBus.
   */
  public void maybeReportSubcommand(Spawn spawn) {
    ShowSubcommands showSubcommands = executor.reportsSubcommands();
    if (!showSubcommands.shouldShowSubcommands) {
      return;
    }

    StringBuilder reason = new StringBuilder();
    ActionOwner owner = spawn.getResourceOwner().getOwner();
    if (owner == null) {
      reason.append(spawn.getResourceOwner().prettyPrint());
    } else {
      reason.append(Label.print(owner.getLabel()));
      reason.append(" [");
      reason.append(spawn.getResourceOwner().prettyPrint());
      reason.append(", configuration: ");
      reason.append(owner.getConfigurationChecksum());
      if (owner.getExecutionPlatform() != null) {
        reason.append(", execution platform: ");
        reason.append(owner.getExecutionPlatform().label());
      }
      reason.append("]");
    }
    String message = Spawns.asShellCommand(spawn, getExecRoot(), showSubcommands.prettyPrintArgs);
    getEventHandler().handle(Event.of(EventKind.SUBCOMMAND, null, "# " + reason + "\n" + message));
  }

  public ImmutableMap<String, String> getClientEnv() {
    return clientEnv;
  }

  public ArtifactExpander getArtifactExpander() {
    return artifactExpander;
  }

  @Nullable
  public Object getSkyframeDepsResult() {
    return skyframeDepsResult;
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

  public ActionKeyContext getActionKeyContext() {
    return actionKeyContext;
  }

  public NestedSetExpander getNestedSetExpander() {
    return nestedSetExpander;
  }

  @Override
  public void close() throws IOException {
    fileOutErr.close();
    if (actionFileSystem instanceof Closeable) {
      ((Closeable) actionFileSystem).close();
    }
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
        actionKeyContext,
        metadataHandler,
        rewindingEnabled,
        lostInputsCheck,
        fileOutErr,
        eventHandler,
        clientEnv,
        topLevelFilesets,
        artifactExpander,
        env,
        actionFileSystem,
        skyframeDepsResult,
        nestedSetExpander);
  }

  /**
   * A way of checking whether any lost inputs have been detected during the execution of this
   * action.
   */
  public interface LostInputsCheck {

    LostInputsCheck NONE = () -> {};

    /** Throws if inputs have been lost. */
    void checkForLostInputs() throws LostInputsActionExecutionException;
  }
}
