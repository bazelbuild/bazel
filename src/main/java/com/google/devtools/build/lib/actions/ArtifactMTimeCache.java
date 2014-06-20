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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.cache.ArtifactMetadataCache;
import com.google.devtools.build.lib.actions.cache.MetadataCache;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;

import java.util.Objects;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A persistent in-memory cache to store the last seen modification time of
 * artifacts. It also serves the purpose of determining whether a build is
 * incremental or not based on the top-level artifacts to be built.
 *
 * <p>This class' correctness is conditional on the previous build having been called on
 * the same set of root artifacts, with Blaze not having been shut down or otherwise cleared
 * in the meantime. Given these conditions, it can be asserted that the only actions needing
 * to be executed are those that are marked as not having succeeded last run, or that have a
 * modified input. A modified input is an artifact that either had its generating action
 * re-executed for whatever reason, or has its mtime (as given by a stat) different
 * than the one cached from the previous build, or was detected as modified in the previous
 * build, but whose dependent actions were not successfully marked as stale in that build
 * (presumably because the build was interrupted before that stage). See documentation of
 * {@link DependentActionGraph#staleActions} for more details on this last case.
 */
public class ArtifactMTimeCache implements ArtifactMetadataRetriever {

  /** Reference to the {@code ArtifactMetadataCache} to use for stat calls. */
  private ArtifactMetadataCache cache;

  /**
   * Set of top-level artifacts built during the last build.
   */
  private Set<Artifact> previousArtifactsToBuild;

  /**
   * Boolean to record whether the last build was successful. It is set to {@code false}
   * at the time {@code buildStarted} is called, and will only be set to {@code true}
   * when {@code buildSuccessful} is called.
   */
  private boolean lastBuildSuccessful;

  /**
   * The workspace ID used by the last build, or null if an output service wasn't used.
   * A non-changing output file workspace is a requirement for an incremental build, since we need
   * to know that <i>blaze-out/some/file</i> references the same file as it did before.
   */
  private String lastWorkspace;

  /**
   * Creates a new ArtifactMTimeCache with the default options.
   */
  public ArtifactMTimeCache() {
    clear();
  }

  /**
   * Reverts the cache to its initial state.
   */
  public void clear() {
    previousArtifactsToBuild = null;
    lastBuildSuccessful = false;
    lastWorkspace = null;
    cache = null;
  }

  /**
   * Checks if the current cache information is applicable to a build of the artifacts in
   * {@code artifactsToBuild}. Also checks that the output file workspace hasn't changed
   * from the last build. If this method returns {@code true} then the {@code changedArtifacts}
   * method will work.
   *
   * Note: This method will return {@code false} if called after the {@code buildStarted}
   * method. It is only intended to be run before a build starts, but must be called on
   * every build.
   *
   * @param useIncrementalDependencyChecker whether incremental_builder is enabled.
   * @param artifactsToBuild set of artifacts in a build to check for applicability.
   * @param mdCache the metadataCache
   * @param workspace workspace ID for the current build, or null if none.
   *        Primarily treated as an opaque identifier: It if changes, the
   *        build is not incremental.
   * @return boolean relevance of this cache to the set of artifacts given.
   */
  public boolean isApplicableToBuild(boolean useIncrementalDependencyChecker,
      Set<Artifact> artifactsToBuild, MetadataCache mdCache, String workspace,
      boolean hasStaleActionData) {
    boolean sameWorkspace = Objects.equals(lastWorkspace, workspace);
    boolean isApplicable = useIncrementalDependencyChecker &&
                           (lastBuildSuccessful || hasStaleActionData) &&
                           mdCache == cache.getMetadataCache() &&
                           sameWorkspace && artifactsToBuild.equals(previousArtifactsToBuild);
    this.lastBuildSuccessful = false;
    this.lastWorkspace = workspace;
    this.previousArtifactsToBuild = artifactsToBuild;
    return isApplicable;
  }

  /**
   * Marks the last build as successful, and updates the list of artifacts the cache is relevant to,
   * and the ArtifactMetadataCache to be held over to the next time the {@code ArtifactMTimeCache}
   * is needed.
   *
   * @param artifactsToBuild set of artifacts successfully built. Note that this is not the full
   *                         list of all built artifacts. Rather it is only the ones that were
   *                         passed to the builder.
   */
  public void markBuildSuccessful(Set<Artifact> artifactsToBuild) {
    this.lastBuildSuccessful = true;
  }

  /**
   * Returns the dirty subset of the given artifact collection.
   *
   * <p>These are the files that have changed since the last successful build. The set includes
   * artifacts whose cached file status is obsolete, as well as those that were already dirty in the
   * last build but we failed to build them.
   *
   * <p>This is a wrapper method around the {@link ArtifactMetadataCache#updateCache} method, where
   * the actual checking is done.
   *
   * @param artifacts the set of artifacts to be checked for changes.
   * @param modified the modified files, or null if unknown.
   * @return subset of all changed artifacts.
   */
  public Set<Artifact> changedArtifacts(Set<Artifact> artifacts,
      ModifiedFileSet modified, Set<Artifact> artifactsKnownBad)
      throws InterruptedException {
    return cache.updateCache(artifacts, modified, artifactsKnownBad);
  }

  public void setArtifactMetadataCache(ArtifactMetadataCache cache) {
    this.cache = cache;
  }

  public ArtifactMetadataCache getArtifactMetadataCache() {
    return cache;
  }

  @Override
  @Nullable
  public byte[] getDigest(Artifact artifact) {
    Preconditions.checkState(cache != null);
    return cache.getDigestMaybe(artifact);
  }

  public void beforeBuild() {
    cache.beforeBuild();
  }

  public void afterBuild() {
    cache.afterBuild();
  }

  public void clearChangedArtifacts() {
    cache.clearChangedArtifacts();
  }
}
