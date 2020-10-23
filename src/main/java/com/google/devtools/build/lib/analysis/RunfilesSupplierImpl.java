// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import javax.annotation.Nullable;

/** {@link RunfilesSupplier} implementation wrapping a single {@link Runfiles} directory mapping. */
// TODO(bazel-team): Consider renaming to SingleRunfilesSupplierImpl.
@AutoCodec
public class RunfilesSupplierImpl implements RunfilesSupplier {
  private final PathFragment runfilesDir;
  private final Runfiles runfiles;
  @Nullable private final Artifact manifest;
  private final boolean buildRunfileLinks;
  private final boolean runfileLinksEnabled;

  /**
   * Create an instance for runfiles directory.
   *
   * @param runfilesDir the path of the runfiles directory relative to the exec root
   * @param runfiles the associated runfiles
   * @param buildRunfileLinks whether runfile symlinks are created during build
   * @param runfileLinksEnabled whether it's allowed to create runfile symlinks
   */
  public RunfilesSupplierImpl(
      PathFragment runfilesDir,
      Runfiles runfiles,
      boolean buildRunfileLinks,
      boolean runfileLinksEnabled) {
    this(runfilesDir, runfiles, /* manifest= */ null, buildRunfileLinks, runfileLinksEnabled);
  }

  /**
   * Create an instance mapping {@code runfiles} to {@code runfilesDir}.
   *
   * @param runfilesDir the desired runfiles directory. Should be relative.
   * @param runfiles the runfiles for runilesDir.
   * @param manifest runfiles' associated runfiles manifest artifact, if present. Important: this
   *     parameter will be used to filter the resulting spawn's inputs to not poison downstream
   *     caches.
   * @param buildRunfileLinks whether runfile symlinks are created during build
   * @param runfileLinksEnabled whether it's allowed to create runfile symlinks
   */
  @AutoCodec.Instantiator
  public RunfilesSupplierImpl(
      PathFragment runfilesDir,
      Runfiles runfiles,
      @Nullable Artifact manifest,
      boolean buildRunfileLinks,
      boolean runfileLinksEnabled) {
    Preconditions.checkArgument(!runfilesDir.isAbsolute());
    this.runfilesDir = Preconditions.checkNotNull(runfilesDir);
    this.runfiles = Preconditions.checkNotNull(runfiles);
    this.manifest = manifest;
    this.buildRunfileLinks = buildRunfileLinks;
    this.runfileLinksEnabled = runfileLinksEnabled;
  }

  /** Use this constructor in tests only. */
  @VisibleForTesting
  public RunfilesSupplierImpl(PathFragment runfilesDir, Runfiles runfiles) {
    this(
        runfilesDir,
        runfiles,
        /* manifest= */ null,
        /* buildRunfileLinks= */ false,
        /* runfileLinksEnabled= */ false);
  }

  /** Creates a runfiles supplier */
  public static RunfilesSupplier create(RunfilesSupport runfilesSupport) {
    return new RunfilesSupplierImpl(
        runfilesSupport.getRunfilesDirectoryExecPath(),
        runfilesSupport.getRunfiles(),
        runfilesSupport.isBuildRunfileLinks(),
        runfilesSupport.isRunfilesEnabled());
  }

  /** creates a runfiles supplier */
  public static RunfilesSupplier create(
      PathFragment runfilesDir,
      Runfiles runfiles,
      @Nullable Artifact manifest,
      BuildConfiguration configuration) {
    return new RunfilesSupplierImpl(
        runfilesDir,
        runfiles,
        manifest,
        configuration.buildRunfileLinks(),
        configuration.buildRunfilesManifests());
  }

  @Override
  public NestedSet<Artifact> getArtifacts() {
    return runfiles.getAllArtifacts();
  }

  @Override
  public ImmutableSet<PathFragment> getRunfilesDirs() {
    return ImmutableSet.of(runfilesDir);
  }

  @Override
  public ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings() {
    return ImmutableMap.of(
        runfilesDir, runfiles.getRunfilesInputs(/*eventHandler=*/ null, /*location=*/ null));
  }

  @Override
  public ImmutableList<Artifact> getManifests() {
    return manifest != null ? ImmutableList.of(manifest) : ImmutableList.<Artifact>of();
  }

  @Override
  public boolean isBuildRunfileLinks(PathFragment runfilesDir) {
    return buildRunfileLinks && this.runfilesDir.equals(runfilesDir);
  }

  @Override
  public boolean isRunfileLinksEnabled(PathFragment runfilesDir) {
    return runfileLinksEnabled && this.runfilesDir.equals(runfilesDir);
  }
}
