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

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.common.base.Preconditions.checkArgument;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import javax.annotation.Nullable;

/** Returns information about executables produced by a target and the files needed to run it. */
@Immutable
public class FilesToRunProvider implements TransitiveInfoProvider, FilesToRunProviderApi<Artifact> {

  /** The name of the field in Starlark used to access a {@link FilesToRunProvider}. */
  public static final String STARLARK_NAME = "files_to_run";

  public static final FilesToRunProvider EMPTY =
      new FilesToRunProvider(NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  public static FilesToRunProvider create(
      NestedSet<Artifact> filesToRun,
      @Nullable RunfilesSupport runfilesSupport,
      @Nullable Artifact executable) {
    if (filesToRun.isEmpty()) {
      checkArgument(runfilesSupport == null, "No files to run with runfiles: %s", runfilesSupport);
      checkArgument(executable == null, "No files to run with executable: %s", executable);
      return EMPTY;
    }
    if (runfilesSupport == null && executable == null) {
      return new FilesToRunProvider(filesToRun);
    }
    if (filesToRun.isSingleton()
        && runfilesSupport == null
        && filesToRun.getSingleton().equals(executable)) {
      return new SingleExecutableFilesToRunProvider(filesToRun);
    }
    return new FullFilesToRunProvider(filesToRun, runfilesSupport, executable);
  }

  private final NestedSet<Artifact> filesToRun;

  private FilesToRunProvider(NestedSet<Artifact> filesToRun) {
    this.filesToRun = filesToRun;
  }

  @Override
  public final boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /** Returns artifacts needed to run the executable for this target. */
  public final NestedSet<Artifact> getFilesToRun() {
    return filesToRun;
  }

  @Override
  @Nullable
  public Artifact getExecutable() {
    return null;
  }

  /**
   * Returns the {@link RunfilesSupport} object associated with the target or null if it does not
   * exist.
   */
  @Nullable
  public RunfilesSupport getRunfilesSupport() {
    return null;
  }

  @Override
  @Nullable
  public final Artifact getRunfilesManifest() {
    var runfilesSupport = getRunfilesSupport();
    return runfilesSupport != null ? runfilesSupport.getRunfilesManifest() : null;
  }

  /** Returns a {@link RunfilesSupplier} encapsulating runfiles for this tool. */
  public final RunfilesSupplier getRunfilesSupplier() {
    return firstNonNull(getRunfilesSupport(), EmptyRunfilesSupplier.INSTANCE);
  }

  /** A single executable. */
  private static final class SingleExecutableFilesToRunProvider extends FilesToRunProvider {

    private SingleExecutableFilesToRunProvider(NestedSet<Artifact> filesToRun) {
      super(filesToRun);
    }

    @Override
    public Artifact getExecutable() {
      return getFilesToRun().getSingleton();
    }
  }

  /** A {@link FilesToRunProvider} possible with {@link RunfilesSupport} and/or an executable. */
  private static final class FullFilesToRunProvider extends FilesToRunProvider {
    @Nullable private final RunfilesSupport runfilesSupport;
    @Nullable private final Artifact executable;

    private FullFilesToRunProvider(
        NestedSet<Artifact> filesToRun,
        @Nullable RunfilesSupport runfilesSupport,
        @Nullable Artifact executable) {
      super(filesToRun);
      this.runfilesSupport = runfilesSupport;
      this.executable = executable;
    }

    @Override
    @Nullable
    public RunfilesSupport getRunfilesSupport() {
      return runfilesSupport;
    }

    @Override
    @Nullable
    public Artifact getExecutable() {
      return executable;
    }
  }
}
