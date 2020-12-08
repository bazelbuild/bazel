// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.includescanning;

import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScannerSupplier;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;

/** IncludeScannerSupplier implementation. */
public class IncludeScannerSupplierImpl implements IncludeScannerSupplier {
  private static class IncludeScannerParams {
    final List<PathFragment> quoteIncludePaths;
    final List<PathFragment> includePaths;
    final List<PathFragment> frameworkIncludePaths;

    IncludeScannerParams(
        List<PathFragment> quoteIncludePaths,
        List<PathFragment> includePaths,
        List<PathFragment> frameworkIncludePaths) {
      this.quoteIncludePaths = quoteIncludePaths;
      this.includePaths = includePaths;
      this.frameworkIncludePaths = frameworkIncludePaths;
    }

    @Override
    public int hashCode() {
      return Objects.hash(quoteIncludePaths, includePaths, frameworkIncludePaths);
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof IncludeScannerParams)) {
        return false;
      }
      IncludeScannerParams that = (IncludeScannerParams) other;
      return this.quoteIncludePaths.equals(that.quoteIncludePaths)
          && this.includePaths.equals(that.includePaths)
          && this.frameworkIncludePaths.equals(that.frameworkIncludePaths);
    }
  }

  private final BlazeDirectories directories;
  private final ExecutorService includePool;
  private final ArtifactFactory artifactFactory;

  private IncludeParser includeParser;

  /**
   * Cache of include scan results mapping source paths to sets of scanned inclusions. Shared by all
   * scanner instances.
   */
  private final ConcurrentMap<Artifact, ListenableFuture<Collection<Inclusion>>> includeParseCache =
      new ConcurrentHashMap<>();

  /** Map of grepped include files from input (.cc or .h) to a header-grepped file. */
  private final PathExistenceCache pathCache;

  private final Supplier<SpawnIncludeScanner> spawnIncludeScannerSupplier;
  private final Path execRoot;

  /** Cache of include scanner instances mapped by include-path hashes. */
  private final LoadingCache<IncludeScannerParams, IncludeScanner> scanners =
      CacheBuilder.newBuilder()
          .build(
              new CacheLoader<IncludeScannerParams, IncludeScanner>() {
                @Override
                public IncludeScanner load(IncludeScannerParams key) {
                  return new LegacyIncludeScanner(
                      includeParser,
                      includePool,
                      includeParseCache,
                      pathCache,
                      key.quoteIncludePaths,
                      key.includePaths,
                      key.frameworkIncludePaths,
                      directories.getOutputPath(execRoot.getBaseName()),
                      execRoot,
                      artifactFactory,
                      spawnIncludeScannerSupplier);
                }
              });

  public IncludeScannerSupplierImpl(
      BlazeDirectories directories,
      ExecutorService includePool,
      ArtifactFactory artifactFactory,
      Supplier<SpawnIncludeScanner> spawnIncludeScannerSupplier,
      Path execRoot) {
    this.directories = directories;
    this.includePool = includePool;
    this.artifactFactory = artifactFactory;
    this.spawnIncludeScannerSupplier = spawnIncludeScannerSupplier;
    this.execRoot = execRoot;
    this.pathCache = new PathExistenceCache(execRoot, artifactFactory);
  }

  @Override
  public IncludeScanner scannerFor(
      List<PathFragment> quoteIncludePaths,
      List<PathFragment> includePaths,
      List<PathFragment> frameworkPaths) {
    Preconditions.checkNotNull(includeParser);
    return scanners.getUnchecked(
        new IncludeScannerParams(quoteIncludePaths, includePaths, frameworkPaths));
  }

  public void init(IncludeParser includeParser) {
    Preconditions.checkState(
        this.includeParser == null,
        "Must only be initialized once: %s %s",
        this.includeParser,
        includeParser);
    Preconditions.checkState(includeParseCache.isEmpty(), includeParseCache);
    Preconditions.checkState(scanners.asMap().isEmpty(), scanners);
    this.includeParser = Preconditions.checkNotNull(includeParser);
    if (this.includeParser.getHints() != null) {
      // The Hints object lives across the lifetime of the Blaze server, but its cached hints may
      // be stale.
      this.includeParser.getHints().clearCachedLegacyHints();
    }
  }
}
