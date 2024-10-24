// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.Globber.Operation;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/** {@link SkyValue} corresponding to the computation result of the {@link GlobsFunction}. */
public class GlobsValue implements SkyValue {

  // TODO: b/290998109 - Storing the matches seem unnecessary except for tests. Consider only
  // storing `matches` when testing.
  private final ImmutableSet<PathFragment> matches;

  public GlobsValue(ImmutableSet<PathFragment> matches) {
    this.matches = matches;
  }

  public ImmutableSet<PathFragment> getMatches() {
    return matches;
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }
    if (!(other instanceof GlobsValue)) {
      return false;
    }

    return getMatches().equals(((GlobsValue) other).getMatches());
  }

  @Override
  public int hashCode() {
    return matches.hashCode();
  }

  /**
   * Representation of individual glob inside a package, including its expression and Globber
   * operation type.
   */
  public static class GlobRequest {

    private final String pattern;
    private final Globber.Operation globOperation;

    public String getPattern() {
      return pattern;
    }

    public Operation getGlobOperation() {
      return globOperation;
    }

    private GlobRequest(String pattern, Globber.Operation globOperation) {
      this.pattern = pattern;
      this.globOperation = globOperation;
    }

    @Override
    public String toString() {
      return String.format("GlobRequest: %s %s", pattern, globOperation);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof GlobRequest other)) {
        return false;
      }

      return pattern.equals(other.pattern) && globOperation.equals(other.globOperation);
    }

    @Override
    public int hashCode() {
      return Objects.hash(pattern, globOperation);
    }

    /**
     * Creates {@link GlobRequest} object iff pattern is a valid glob expression.
     *
     * <p>@throws InvalidGlobPatternException if the pattern is not valid.
     */
    public static GlobRequest create(String pattern, Globber.Operation globOperation)
        throws InvalidGlobPatternException {
      if (pattern.indexOf('?') != -1) {
        throw new InvalidGlobPatternException(pattern, "wildcard ? forbidden");
      }

      String error = UnixGlob.checkPatternForError(pattern);
      if (error != null) {
        throw new InvalidGlobPatternException(pattern, error);
      }
      return new GlobRequest(pattern, globOperation);
    }
  }

  /**
   * Returns the interned {@link GlobsValue.Key} object which contains all glob deps of a package.
   *
   * @param packageIdentifier packageId the name of the owner package (must be an existing package)
   * @param packageRoot the package root of {@code packageId}
   * @param globRequests container of all glob expressions and types of Globber operations, all
   *     input glob expressions are expected to be valid.
   */
  public static Key key(
      PackageIdentifier packageIdentifier,
      Root packageRoot,
      ImmutableSet<GlobRequest> globRequests) {
    return Key.create(packageIdentifier, packageRoot, globRequests);
  }

  /**
   * {@link SkyKey} type for {@link GlobsValue}, serving as the input to {@link GlobsFunction}.
   *
   * <p>Expects all glob expressions inside {@link Key#globRequests} are valid, as indicated by
   * {@code UnixGlob#checkPatternForError}.
   */
  @VisibleForSerialization
  @AutoCodec
  public static class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private final PackageIdentifier packageIdentifier;
    private final Root packageRoot;
    private final ImmutableSet<GlobRequest> globRequests;

    private static Key create(
        PackageIdentifier packageIdentifier,
        Root packageRoot,
        ImmutableSet<GlobRequest> globRequests) {
      return interner.intern(new Key(packageIdentifier, packageRoot, globRequests));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    private Key(
        PackageIdentifier packageIdentifier,
        Root packageRoot,
        ImmutableSet<GlobRequest> globRequests) {
      this.packageIdentifier = packageIdentifier;
      this.packageRoot = packageRoot;
      this.globRequests = globRequests;
    }

    /**
     * Returns the package that "owns" all globs.
     *
     * <p>The globs evaluation code ensures that the boundaries of this package are not crossed.
     */
    public PackageIdentifier getPackageIdentifier() {
      return packageIdentifier;
    }

    /** Returns the package root of {@link #packageIdentifier}. */
    public Root getPackageRoot() {
      return packageRoot;
    }

    /**
     * Returns an {@link ImmutableSet} containing all globs inside the package, including each glob
     * expression and operation.
     */
    public ImmutableSet<GlobRequest> getGlobRequests() {
      return globRequests;
    }

    @Override
    public boolean skipsBatchPrefetch() {
      return true;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.GLOBS;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Key other)) {
        return false;
      }
      return packageIdentifier.equals(other.packageIdentifier)
          && packageRoot.equals(other.packageRoot)
          && globRequests.equals(other.globRequests);
    }

    @Override
    public int hashCode() {
      return Objects.hash(packageIdentifier, packageRoot, globRequests);
    }

    @Override
    public String toString() {
      return String.format(
          "<GlobsKey packageRoot = %s, packageIdentifier = %s, globRequests = [%s]>",
          packageRoot,
          packageIdentifier,
          globRequests.stream().map(GlobRequest::toString).sorted().collect(joining(",")));
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
