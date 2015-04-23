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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;

import javax.annotation.Nullable;

/**
 * Returns information about executables produced by a target and the files needed to run it.
 */
@Immutable
public final class FilesToRunProvider implements TransitiveInfoProvider {

  private final Label label;
  private final ImmutableList<Artifact> filesToRun;
  @Nullable private final RunfilesSupport runfilesSupport;
  @Nullable private final Artifact executable;

  public FilesToRunProvider(Label label, ImmutableList<Artifact> filesToRun,
      @Nullable RunfilesSupport runfilesSupport, @Nullable Artifact executable) {
    this.label = label;
    this.filesToRun = filesToRun;
    this.runfilesSupport = runfilesSupport;
    this.executable  = executable;
  }

  /**
   * Creates an instance that contains one single executable and no other files.
   */
  public static FilesToRunProvider fromSingleArtifact(Label label, Artifact artifact) {
    return new FilesToRunProvider(label, ImmutableList.of(artifact), null, artifact);
  }

  /**
   * Returns the label that is associated with this piece of information.
   *
   * <p>This is usually the label of the target that provides the information.
   */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns artifacts needed to run the executable for this target.
   */
  public ImmutableList<Artifact> getFilesToRun() {
    return filesToRun;
  }

  /**
   * Returns the {@RunfilesSupport} object associated with the target or null if it does not exist.
   */
  @Nullable public RunfilesSupport getRunfilesSupport() {
    return runfilesSupport;
  }

  /**
   * Returns the Executable or null if it does not exist.
   */
  @Nullable public Artifact getExecutable() {
    return executable;
  }

  /**
   * Returns the RunfilesManifest or null if it does not exist. It is a shortcut to
   * getRunfilesSupport().getRunfilesManifest().
   */
  @Nullable public Artifact getRunfilesManifest() {
    return runfilesSupport != null ? runfilesSupport.getRunfilesManifest() : null;
  }
}
