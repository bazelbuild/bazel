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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.lang.ref.SoftReference;
import java.util.Map;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** {@link RunfilesSupplier} implementation wrapping a single {@link Runfiles} directory mapping. */
@AutoCodec
public final class SingleRunfilesSupplier implements RunfilesSupplier {

  private final PathFragment runfilesDir;
  private final Runfiles runfiles;
  private final Supplier<Map<PathFragment, Artifact>> runfilesInputs;
  @Nullable private final Artifact repoMappingManifest;
  private final RunfileSymlinksMode runfileSymlinksMode;
  private final boolean buildRunfileLinks;

  /**
   * Same as {@link SingleRunfilesSupplier#SingleRunfilesSupplier(PathFragment, Runfiles, boolean,
   * Artifact, RunfileSymlinksMode, boolean)}, except adds caching for {@linkplain
   * Runfiles#getRunfilesInputs runfiles inputs}.
   *
   * <p>The runfiles inputs are computed lazily and softly cached. Caching is shared across
   * instances created via {@link #withOverriddenRunfilesDir}.
   */
  public static SingleRunfilesSupplier createCaching(
      PathFragment runfilesDir,
      Runfiles runfiles,
      @Nullable Artifact repoMappingManifest,
      RunfileSymlinksMode runfileSymlinksMode,
      boolean buildRunfileLinks) {
    return new SingleRunfilesSupplier(
        runfilesDir,
        runfiles,
        /* runfilesCachingEnabled= */ true,
        repoMappingManifest,
        runfileSymlinksMode,
        buildRunfileLinks);
  }

  /**
   * Create an instance mapping {@code runfiles} to {@code runfilesDir}.
   *
   * @param runfilesDir the desired runfiles directory. Should be relative.
   * @param runfiles the runfiles for runilesDir.
   * @param runfileSymlinksMode how to create runfile symlinks
   * @param buildRunfileLinks whether runfile symlinks should be created during the build
   */
  @AutoCodec.Instantiator
  public SingleRunfilesSupplier(
      PathFragment runfilesDir,
      Runfiles runfiles,
      @Nullable Artifact repoMappingManifest,
      RunfileSymlinksMode runfileSymlinksMode,
      boolean buildRunfileLinks) {
    this(
        runfilesDir,
        runfiles,
        /* runfilesCachingEnabled= */ false,
        repoMappingManifest,
        runfileSymlinksMode,
        buildRunfileLinks);
  }

  private SingleRunfilesSupplier(
      PathFragment runfilesDir,
      Runfiles runfiles,
      boolean runfilesCachingEnabled,
      @Nullable Artifact repoMappingManifest,
      RunfileSymlinksMode runfileSymlinksMode,
      boolean buildRunfileLinks) {
    this(
        runfilesDir,
        runfiles,
        runfilesCachingEnabled
            ? new RunfilesCacher(runfiles, repoMappingManifest)
            : () ->
                runfiles.getRunfilesInputs(
                    /* eventHandler= */ null, /* location= */ null, repoMappingManifest),
        repoMappingManifest,
        runfileSymlinksMode,
        buildRunfileLinks);
  }

  private SingleRunfilesSupplier(
      PathFragment runfilesDir,
      Runfiles runfiles,
      Supplier<Map<PathFragment, Artifact>> runfilesInputs,
      @Nullable Artifact repoMappingManifest,
      RunfileSymlinksMode runfileSymlinksMode,
      boolean buildRunfileLinks) {
    checkArgument(!runfilesDir.isAbsolute());
    this.runfilesDir = checkNotNull(runfilesDir);
    this.runfiles = checkNotNull(runfiles);
    this.runfilesInputs = checkNotNull(runfilesInputs);
    this.repoMappingManifest = repoMappingManifest;
    this.runfileSymlinksMode = runfileSymlinksMode;
    this.buildRunfileLinks = buildRunfileLinks;
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
    return ImmutableMap.of(runfilesDir, runfilesInputs.get());
  }

  @Override
  @Nullable
  public RunfileSymlinksMode getRunfileSymlinksMode(PathFragment runfilesDir) {
    if (this.runfilesDir.equals(runfilesDir)) {
      return runfileSymlinksMode;
    }
    return null;
  }

  @Override
  public boolean isBuildRunfileLinks(PathFragment runfilesDir) {
    return buildRunfileLinks && this.runfilesDir.equals(runfilesDir);
  }

  @Override
  public SingleRunfilesSupplier withOverriddenRunfilesDir(PathFragment newRunfilesDir) {
    return newRunfilesDir.equals(runfilesDir)
        ? this
        : new SingleRunfilesSupplier(
            newRunfilesDir,
            runfiles,
            runfilesInputs,
            repoMappingManifest,
            runfileSymlinksMode,
            buildRunfileLinks);
  }

  @Override
  public Map<PathFragment, RunfilesTree> getRunfilesTreesForLogging() {
    return ImmutableMap.of(
        runfilesDir,
        new RunfilesTree(
            runfilesDir,
            runfiles.getArtifacts(),
            runfiles.getEmptyFilenames(),
            runfiles.getRootSymlinks(),
            runfiles.getSymlinks(),
            repoMappingManifest,
            runfiles.isLegacyExternalRunfiles()));
  }

  /** Softly caches the result of {@link Runfiles#getRunfilesInputs}. */
  private static final class RunfilesCacher implements Supplier<Map<PathFragment, Artifact>> {

    private final Runfiles runfiles;
    @Nullable private final Artifact repoMappingManifest;
    private volatile SoftReference<Map<PathFragment, Artifact>> ref = new SoftReference<>(null);

    RunfilesCacher(Runfiles runfiles, @Nullable Artifact repoMappingManifest) {
      this.runfiles = runfiles;
      this.repoMappingManifest = repoMappingManifest;
    }

    @Override
    public Map<PathFragment, Artifact> get() {
      Map<PathFragment, Artifact> result = ref.get();
      if (result != null) {
        return result;
      }
      synchronized (this) {
        result = ref.get();
        if (result == null) {
          result =
              runfiles.getRunfilesInputs(
                  /*eventHandler=*/ null, /*location=*/ null, repoMappingManifest);
          ref = new SoftReference<>(result);
        }
      }
      return result;
    }
  }
}
