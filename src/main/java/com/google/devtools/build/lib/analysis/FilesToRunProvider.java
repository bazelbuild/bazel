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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import javax.annotation.Nullable;

/** Returns information about executables produced by a target and the files needed to run it. */
@Immutable
@SkylarkModule(name = "FilesToRunProvider", doc = "", category = SkylarkModuleCategory.PROVIDER)
public final class FilesToRunProvider implements TransitiveInfoProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String SKYLARK_NAME = "files_to_run";

  public static final FilesToRunProvider EMPTY =
      new FilesToRunProvider(ImmutableList.<Artifact>of(), null, null);

  private final ImmutableList<Artifact> filesToRun;
  @Nullable private final RunfilesSupport runfilesSupport;
  @Nullable private final Artifact executable;

  public FilesToRunProvider(ImmutableList<Artifact> filesToRun,
      @Nullable RunfilesSupport runfilesSupport, @Nullable Artifact executable) {
    this.filesToRun = filesToRun;
    this.runfilesSupport = runfilesSupport;
    this.executable  = executable;
  }

  /**
   * Creates an instance that contains one single executable and no other files.
   */
  public static FilesToRunProvider fromSingleExecutableArtifact(Artifact artifact) {
    return new FilesToRunProvider(ImmutableList.of(artifact), null, artifact);
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

  /** Returns the Executable or null if it does not exist. */
  @SkylarkCallable(
    name = "executable",
    doc = "The main executable or None if it does not exist.",
    structField = true,
    allowReturnNones = true
  )
  @Nullable
  public Artifact getExecutable() {
    return executable;
  }

  /**
   * Returns the RunfilesManifest or null if it does not exist. It is a shortcut to
   * getRunfilesSupport().getRunfilesManifest().
   */
  @SkylarkCallable(
    name = "runfiles_manifest",
    doc = "The runfiles manifest or None if it does not exist.",
    structField = true,
    allowReturnNones = true
  )
  @Nullable
  public Artifact getRunfilesManifest() {
    return runfilesSupport != null ? runfilesSupport.getRunfilesManifest() : null;
  }

  /** Return a {@link RunfilesSupplier} encapsulating runfiles for this tool. */
  public RunfilesSupplier getRunfilesSupplier() {
    if (executable != null && runfilesSupport != null) {
      return new RunfilesSupplierImpl(executable, runfilesSupport.getRunfiles());
    } else {
      return EmptyRunfilesSupplier.INSTANCE;
    }
  }
}
