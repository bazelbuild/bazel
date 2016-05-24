// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Function;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.Set;

import javax.annotation.Nullable;

/**
 * Provider of transitively available dex archives corresponding to Jars.  A dex archive is a zip of
 * {@code .dex} files that each encode exactly one {@code .class} file in an Android-readable form.
 * The file names in a dex archive should match the file names in the originating Jar file, except
 * with {@code .dex} appended, i.e., {@code <package/for/ClassName[$Inner].class.dex}.
 *
 * <p>For convenience this class implements a {@link Function} to map from Jars to dex archives if
 * available (returns the given Jar otherwise).
 */
@Immutable
public class DexArchiveProvider implements TransitiveInfoProvider {

  /**
   * Provider that doesn't provide any dex archives, which is what any neverlink target should use.
   * It's not strictly necessary to handle neverlink specially, but doing so reduces the amount
   * of processing done for targets that won't be used for dexing anyway.
   */
  public static final DexArchiveProvider NEVERLINK = new DexArchiveProvider.Builder().build();

  /**
   * Builder for {@link DexArchiveProvider}.
   */
  public static class Builder {

    private final Table<ImmutableSet<String>, Artifact, Artifact> dexArchives =
        HashBasedTable.create();

    public Builder() {
    }

    /**
     * Adds all dex archives from the given providers, which is useful to aggregate providers from
     * dependencies.
     */
    public Builder addTransitiveProviders(Iterable<DexArchiveProvider> providers) {
      for (DexArchiveProvider provider : providers) {
        dexArchives.putAll(provider.dexArchives);
      }
      return this;
    }

    /**
     * Adds the given dex archive as a replacement for the given Jar.
     * @param dexopts
     */
    public Builder addDexArchive(Set<String> dexopts, Artifact dexArchive, Artifact dexedJar) {
      checkArgument(dexArchive.getFilename().endsWith(".dex.zip"),
          "Doesn't look like a dex archive: %s", dexArchive);
      // Adding this artifact will fail iff dexArchive already appears as the value of another jar.
      // It's ok and expected to put the same pair multiple times. Note that ImmutableBiMap fails
      // in that situation, which is why we're not using it here.
      // It's weird to put a dexedJar that's already in the map with a different value so we fail.
      Artifact old =
          dexArchives.put(
              ImmutableSet.copyOf(dexopts), checkNotNull(dexedJar, "dexedJar"), dexArchive);
      checkArgument(old == null || old.equals(dexedJar),
          "We already had mapping %s-%s for dexopts %s, so we don't also need %s",
          dexedJar, old, dexopts, dexArchive);
      return this;
    }

    /**
     * Returns the finished {@link DexArchiveProvider}.
     */
    public DexArchiveProvider build() {
      return new DexArchiveProvider(ImmutableTable.copyOf(dexArchives));
    }
  }

  /** Map from Jar artifacts to the corresponding dex archives. */
  private final ImmutableTable<ImmutableSet<String>, Artifact, Artifact> dexArchives;

  private DexArchiveProvider(ImmutableTable<ImmutableSet<String>, Artifact, Artifact> dexArchives) {
    this.dexArchives = dexArchives;
  }

  public Function<Artifact, Artifact> archivesForDexopts(ImmutableSet<String> dexopts) {
    final ImmutableMap<Artifact, Artifact> dexArchivesForDexopts = dexArchives.row(dexopts);
    return new Function<Artifact, Artifact>() {
      /** Maps Jars to available dex archives and returns the given Jar otherwise. */
      @Override
      @Nullable
      public Artifact apply(@Nullable Artifact jar) {
        Artifact dexArchive = dexArchivesForDexopts.get(jar);
        return dexArchive != null ? dexArchive : jar; // return null iff input == null
      }
    };
  }

  @Override
  public int hashCode() {
    return dexArchives.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    DexArchiveProvider other = (DexArchiveProvider) obj;
    return dexArchives.equals(other.dexArchives);
  }
}
