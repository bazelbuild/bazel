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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.Objects;

/**
 * Parameters to be passed to the linker.
 *
 * <p>The parameters concerned are the link options (strings) passed to the linker, linkstamps, a
 * list of libraries to be linked in, and a list of libraries to build at link time.
 *
 * <p>Items in the collections are stored in nested sets. Link options and libraries are stored in
 * link order (preorder) and linkstamps are sorted.
 */
@AutoCodec
public final class CcLinkParams {
  /**
   * A list of link options contributed by a single configured target.
   *
   * <p><b>WARNING:</b> Do not implement {@code #equals()} in the obvious way. This class must be
   * checked for equality by object identity because otherwise if two configured targets contribute
   * the same link options, they will be de-duplicated, which is not the desirable behavior.
   */
  @AutoCodec
  @Immutable
  public static final class LinkOptions {
    private final ImmutableList<String> linkOptions;

    @VisibleForSerialization
    LinkOptions(Iterable<String> linkOptions) {
      this.linkOptions = ImmutableList.copyOf(linkOptions);
    }

    public ImmutableList<String> get() {
      return linkOptions;
    }

    public static LinkOptions of(Iterable<String> linkOptions) {
      return new LinkOptions(linkOptions);
    }

    @Override
    public String toString() {
      return '[' + Joiner.on(",").join(linkOptions) + ']';
    }
  }

  /**
   * A linkstamp that also knows about its declared includes.
   *
   * <p>This object is required because linkstamp files may include other headers which will have to
   * be provided during compilation.
   */
  @AutoCodec
  public static final class Linkstamp {
    private final Artifact artifact;
    private final NestedSet<Artifact> declaredIncludeSrcs;

    @VisibleForSerialization
    Linkstamp(Artifact artifact, NestedSet<Artifact> declaredIncludeSrcs) {
      this.artifact = Preconditions.checkNotNull(artifact);
      this.declaredIncludeSrcs = Preconditions.checkNotNull(declaredIncludeSrcs);
    }

    /**
     * Returns the linkstamp artifact.
     */
    public Artifact getArtifact() {
      return artifact;
    }

    /**
     * Returns the declared includes.
     */
    public NestedSet<Artifact> getDeclaredIncludeSrcs() {
      return declaredIncludeSrcs;
    }

    @Override
    public int hashCode() {
      return Objects.hash(artifact, declaredIncludeSrcs);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Linkstamp)) {
        return false;
      }
      Linkstamp other = (Linkstamp) obj;
      return artifact.equals(other.artifact)
          && declaredIncludeSrcs.equals(other.declaredIncludeSrcs);
    }
  }
}
