// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.graph.MutableGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.skyframe.ExecutionPhaseSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.HashSet;
import java.util.Set;

/**
 * {@link SkyKey} for requesting all artifacts in a {@link NestedSet}.
 *
 * <p>{@link com.google.devtools.build.lib.skyframe.ArtifactNestedSetFunction} requests the keys
 * returned by {@link #getDirectDepKeys} to ensure that the Skyframe graph mirrors the {@link
 * NestedSet} structure. This class is only declared in the {@code nestedset} package to give it
 * low-level access to the backing {@code Object[]}.
 *
 * <p>An {@link ArtifactNestedSetKey} must only be created from a {@link NestedSet} with multiple
 * elements. Singletons should be translated into a direct request for the corresponding {@link
 * Artifact#key}.
 *
 * <p>Instances are compared using identity equality on the corresponding {@code Object[]} (note
 * that to save memory, this class does not retain a {@link NestedSet.Node}). This means that two
 * instances are not guaranteed to compare equal even if they contain the same set of artifacts.
 */
public final class ArtifactNestedSetKey implements ExecutionPhaseSkyKey {

  private static final SkyKeyInterner<ArtifactNestedSetKey> interner = SkyKey.newInterner();

  public static ArtifactNestedSetKey create(NestedSet<Artifact> set) {
    Object children = set.getChildren();
    checkArgument(
        children instanceof Object[],
        "ArtifactNestedSetKey cannot represent empty or singleton set: %s",
        set);
    return createInternal((Object[]) children);
  }

  private static ArtifactNestedSetKey createInternal(Object[] children) {
    return interner.intern(new ArtifactNestedSetKey(children));
  }

  private final Object[] children;

  private ArtifactNestedSetKey(Object[] children) {
    this.children = children;
  }

  /**
   * Returns a list of this key's direct dependencies, including {@link Artifact#key} for leaves and
   * {@link ArtifactNestedSetKey} for non-leaves.
   */
  public ImmutableList<SkyKey> getDirectDepKeys() {
    ImmutableList.Builder<SkyKey> depKeys = ImmutableList.builderWithExpectedSize(children.length);
    for (Object child : children) {
      if (child instanceof Artifact artifact) {
        depKeys.add(Artifact.key(artifact));
      } else {
        depKeys.add(createInternal((Object[]) child));
      }
    }
    return depKeys.build();
  }

  /** Applies a consumer function to the direct artifacts of this nested set. */
  public void applyToDirectArtifacts(DirectArtifactConsumer function) throws InterruptedException {
    for (Object child : children) {
      if (child instanceof Artifact artifact) {
        function.accept(artifact);
      }
    }
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.ARTIFACT_NESTED_SET;
  }

  /**
   * Like {@link NestedSet#toList}, returns a duplicate-free list of the artifacts contained beneath
   * the node represented by this key.
   *
   * <p>Order of the returned list is undefined.
   */
  public ImmutableList<Artifact> expandToArtifacts() {
    return new NestedSet<Artifact>(
            Order.STABLE_ORDER,
            /* depth= */ 3, // Not accurate, but doesn't matter.
            children,
            /* memo= */ null)
        .toList();
  }

  /**
   * Augments the given rewind graph with paths from {@code failedKey} to {@code lostArtifacts}
   * discoverable by following the {@link ArtifactNestedSetKey} nodes in {@code failedKeyDeps}.
   *
   * <p>{@code rewindGraph} must not contain any {@link ArtifactNestedSetKey} nodes prior to calling
   * this method.
   */
  public static void addNestedSetPathsToRewindGraph(
      MutableGraph<SkyKey> rewindGraph,
      SkyKey failedKey,
      Set<SkyKey> failedKeyDeps,
      Set<? extends Artifact> lostArtifacts) {
    var seen = new HashSet<ArtifactNestedSetKey>();
    for (var nestedSetDep : Iterables.filter(failedKeyDeps, ArtifactNestedSetKey.class)) {
      if (searchForLostArtifacts(nestedSetDep, rewindGraph, lostArtifacts, seen)) {
        rewindGraph.putEdge(failedKey, nestedSetDep);
      }
    }
  }

  private static boolean searchForLostArtifacts(
      ArtifactNestedSetKey node,
      MutableGraph<SkyKey> rewindGraph,
      Set<? extends Artifact> lostArtifacts,
      Set<ArtifactNestedSetKey> seen) {
    if (rewindGraph.nodes().contains(node)) {
      return true;
    }
    if (!seen.add(node)) {
      return false;
    }
    boolean anyFound = false;
    for (Object child : node.children) {
      if (child instanceof Artifact artifact) {
        if (lostArtifacts.contains(artifact)) {
          rewindGraph.putEdge(node, Artifact.key(artifact));
          anyFound = true;
        }
      } else {
        ArtifactNestedSetKey nextNode = createInternal((Object[]) child);
        if (searchForLostArtifacts(nextNode, rewindGraph, lostArtifacts, seen)) {
          rewindGraph.putEdge(node, nextNode);
          anyFound = true;
        }
      }
    }
    return anyFound;
  }

  @Override
  public boolean valueIsShareable() {
    // ArtifactNestedSetValue is just a promise that data is available in memory. Not meant for
    // cross-server sharing.
    return false;
  }

  @Override
  public SkyKeyInterner<?> getSkyKeyInterner() {
    return interner;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(children);
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }
    return that instanceof ArtifactNestedSetKey artifactNestedSetKey
        && children == artifactNestedSetKey.children;
  }

  @Override
  public String toString() {
    return String.format("ArtifactNestedSetKey[%s]@%s", children.length, hashCode());
  }

  /** A consumer to be applied to each direct artifact. */
  @FunctionalInterface
  public interface DirectArtifactConsumer {
    void accept(Artifact artifact) throws InterruptedException;
  }
}
