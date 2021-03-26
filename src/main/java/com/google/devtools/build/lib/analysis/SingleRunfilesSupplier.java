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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
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
   * Same as {@link SingleRunfilesSupplier#SingleRunfilesSupplier(PathFragment, Runfiles, Artifact,
   * boolean, boolean)}, except adds caching for {@linkplain Runfiles#getRunfilesInputs runfiles
   * inputs}.
   *
   * <p>The runfiles inputs are computed lazily and softly cached. Caching is shared across
   * instances created via {@link #withOverriddenRunfilesDir}.
   */
  public static SingleRunfilesSupplier createCaching(
      PathFragment runfilesDir,
      Runfiles runfiles,
      boolean buildRunfileLinks,
      boolean runfileLinksEnabled) {
    return new SingleRunfilesSupplier(
        runfilesDir,
        runfiles,
        new RunfilesCacher(runfiles),
        /*manifest=*/ null,
        buildRunfileLinks,
        runfileLinksEnabled);
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
  public SingleRunfilesSupplier(
      PathFragment runfilesDir,
      Runfiles runfiles,
      @Nullable Artifact manifest,
      boolean buildRunfileLinks,
      boolean runfileLinksEnabled) {
    this(
        runfilesDir,
        runfiles,
        () -> runfiles.getRunfilesInputs(/*eventHandler=*/ null, /*location=*/ null),
        manifest,
        buildRunfileLinks,
        runfileLinksEnabled);
  }

  private SingleRunfilesSupplier(
      PathFragment runfilesDir,
      Runfiles runfiles,
      Supplier<Map<PathFragment, Artifact>> runfilesInputs,
      @Nullable Artifact manifest,
      boolean buildRunfileLinks,
      boolean runfileLinksEnabled) {
    checkArgument(!runfilesDir.isAbsolute());
    this.runfilesDir = checkNotNull(runfilesDir);
    this.runfiles = checkNotNull(runfiles);
    this.runfilesInputs = checkNotNull(runfilesInputs);
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
    return ImmutableMap.of(runfilesDir, runfilesInputs.get());
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
            newRunfilesDir,
            runfiles,
            runfilesInputs,
            manifest,
            buildRunfileLinks,
            runfileLinksEnabled);
  }

  /** Softly caches the result of {@link Runfiles#getRunfilesInputs}. */
  private static final class RunfilesCacher implements Supplier<Map<PathFragment, Artifact>> {
    private final Runfiles runfiles;
    private volatile SoftReference<Map<PathFragment, Artifact>> ref = new SoftReference<>(null);

    RunfilesCacher(Runfiles runfiles) {
      this.runfiles = runfiles;
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
          result = runfiles.getRunfilesInputs(/*eventHandler=*/ null, /*location=*/ null);
          ref = new SoftReference<>(result);
        }
      }
      return result;
    }
  }
}
