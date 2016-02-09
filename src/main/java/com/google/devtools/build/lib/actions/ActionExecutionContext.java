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

import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;

import javax.annotation.Nullable;

/**
 * A class that groups services in the scope of the action. Like the FileOutErr object.
 */
public class ActionExecutionContext {

  private final Executor executor;
  private final ActionInputFileCache actionInputFileCache;
  private final MetadataHandler metadataHandler;
  private final FileOutErr fileOutErr;
  private final ArtifactExpander artifactExpander;
  @Nullable
  private final Environment env;

  private ActionExecutionContext(Executor executor, ActionInputFileCache actionInputFileCache,
      MetadataHandler metadataHandler, FileOutErr fileOutErr,
      @Nullable ArtifactExpander artifactExpander,
      @Nullable SkyFunction.Environment env) {
    this.actionInputFileCache = actionInputFileCache;
    this.metadataHandler = metadataHandler;
    this.fileOutErr = fileOutErr;
    this.executor = executor;
    this.artifactExpander = artifactExpander;
    this.env = env;
  }

  public ActionExecutionContext(Executor executor, ActionInputFileCache actionInputFileCache,
      MetadataHandler metadataHandler, FileOutErr fileOutErr, ArtifactExpander artifactExpander) {
    this(executor, actionInputFileCache, metadataHandler, fileOutErr, artifactExpander, null);
  }

  public static ActionExecutionContext normal(Executor executor,
      ActionInputFileCache actionInputFileCache, MetadataHandler metadataHandler,
      FileOutErr fileOutErr, ArtifactExpander artifactExpander) {
    return new ActionExecutionContext(executor, actionInputFileCache, metadataHandler, fileOutErr,
        artifactExpander, null);
  }

  public static ActionExecutionContext forInputDiscovery(Executor executor,
      ActionInputFileCache actionInputFileCache, MetadataHandler metadataHandler,
      FileOutErr fileOutErr, Environment env) {
    return new ActionExecutionContext(executor, actionInputFileCache, metadataHandler, fileOutErr,
        null, env);
  }

  public ActionInputFileCache getActionInputFileCache() {
    return actionInputFileCache;
  }

  public MetadataHandler getMetadataHandler() {
    return metadataHandler;
  }

  public Executor getExecutor() {
    return executor;
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

  /**
   * Allows us to create a new context that overrides the FileOutErr with another one. This is
   * useful for muting the output for example.
   */
  public ActionExecutionContext withFileOutErr(FileOutErr fileOutErr) {
    return new ActionExecutionContext(executor, actionInputFileCache, metadataHandler, fileOutErr,
        artifactExpander, env);
  }
}
