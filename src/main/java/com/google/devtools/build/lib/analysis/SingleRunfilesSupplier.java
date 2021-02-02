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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import javax.annotation.Nullable;

/** {@link RunfilesSupplier} implementation wrapping a single {@link Runfiles} directory mapping. */
public class SingleRunfilesSupplier implements RunfilesSupplier {
  private final PathFragment runfilesDir;
  private final Runfiles runfiles;
  @Nullable private final Artifact manifest;
  private final boolean buildRunfileLinks;
  private final boolean runfileLinksEnabled;

  /**
   * Creates a no-manifest {@link SingleRunfilesSupplier} from the given {@link RunfilesSupport}.
   */
  public static SingleRunfilesSupplier create(RunfilesSupport runfilesSupport) {
    return new SingleRunfilesSupplier(
        runfilesSupport.getRunfilesDirectoryExecPath(),
        runfilesSupport.getRunfiles(),
        /*manifest=*/ null,
        runfilesSupport.isBuildRunfileLinks(),
        runfilesSupport.isRunfilesEnabled());
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
  public SingleRunfilesSupplier(
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
    return manifest != null ? ImmutableList.of(manifest) : ImmutableList.of();
  }

  @Override
  public boolean isBuildRunfileLinks(PathFragment runfilesDir) {
    return buildRunfileLinks && this.runfilesDir.equals(runfilesDir);
  }

  @Override
  public boolean isRunfileLinksEnabled(PathFragment runfilesDir) {
    return runfileLinksEnabled && this.runfilesDir.equals(runfilesDir);
  }

  /**
   * Returns a {@link SingleRunfilesSupplier} identical to this one, but with the given runfiles
   * directory.
   */
  public SingleRunfilesSupplier withOverriddenRunfilesDir(PathFragment newRunfilesDir) {
    return newRunfilesDir.equals(runfilesDir)
        ? this
        : new SingleRunfilesSupplier(
            newRunfilesDir, runfiles, manifest, buildRunfileLinks, runfileLinksEnabled);
  }
}
