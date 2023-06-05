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
public interface FilesToRunProvider
    extends TransitiveInfoProvider, FilesToRunProviderApi<Artifact> {

  /** The name of the field in Starlark used to access a {@link FilesToRunProvider}. */
  String STARLARK_NAME = "files_to_run";

  FilesToRunProvider EMPTY =
      new BasicFilesToRunProvider(NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  /** Creates an instance that contains one single executable and no other files. */
  static FilesToRunProvider fromSingleExecutableArtifact(Artifact artifact) {
    return new SingleExecutableFilesToRunProvider(
        NestedSetBuilder.create(Order.STABLE_ORDER, artifact));
  }

  static FilesToRunProvider create(
      NestedSet<Artifact> filesToRun,
      @Nullable RunfilesSupport runfilesSupport,
      @Nullable Artifact executable) {
    if (filesToRun.isEmpty()) {
      checkArgument(runfilesSupport == null, "No files to run with runfiles: %s", runfilesSupport);
      checkArgument(executable == null, "No files to run with executable: %s", executable);
      return EMPTY;
    }
    if (runfilesSupport == null && executable == null) {
      return new BasicFilesToRunProvider(filesToRun);
    }
    if (filesToRun.isSingleton()
        && runfilesSupport == null
        && filesToRun.getSingleton().equals(executable)) {
      return new SingleExecutableFilesToRunProvider(filesToRun);
    }
    return new FullFilesToRunProvider(filesToRun, runfilesSupport, executable);
  }

  @Override
  default boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /** Returns artifacts needed to run the executable for this target. */
  NestedSet<Artifact> getFilesToRun();

  @Override
  @Nullable
  default Artifact getExecutable() {
    return null;
  }

  /**
   * Returns the {@link RunfilesSupport} object associated with the target or null if it does not
   * exist.
   */
  @Nullable
  default RunfilesSupport getRunfilesSupport() {
    return null;
  }

  @Override
  @Nullable
  default Artifact getRunfilesManifest() {
    var runfilesSupport = getRunfilesSupport();
    return runfilesSupport != null ? runfilesSupport.getRunfilesManifest() : null;
  }

  /** Returns a {@link RunfilesSupplier} encapsulating runfiles for this tool. */
  default RunfilesSupplier getRunfilesSupplier() {
    return firstNonNull(getRunfilesSupport(), EmptyRunfilesSupplier.INSTANCE);
  }

  /** A {@link FilesToRunProvider} with no {@link RunfilesSupport} or executable. */
  class BasicFilesToRunProvider implements FilesToRunProvider {
    private final NestedSet<Artifact> filesToRun;

    private BasicFilesToRunProvider(NestedSet<Artifact> filesToRun) {
      this.filesToRun = filesToRun;
    }

    @Override
    public NestedSet<Artifact> getFilesToRun() {
      return filesToRun;
    }
  }

  /** A single executable. */
  final class SingleExecutableFilesToRunProvider extends BasicFilesToRunProvider {

    private SingleExecutableFilesToRunProvider(NestedSet<Artifact> filesToRun) {
      super(filesToRun);
    }

    @Override
    public Artifact getExecutable() {
      return getFilesToRun().getSingleton();
    }
  }

  /** A {@link FilesToRunProvider} possible with {@link RunfilesSupport} and/or an executable. */
  final class FullFilesToRunProvider extends BasicFilesToRunProvider {
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
