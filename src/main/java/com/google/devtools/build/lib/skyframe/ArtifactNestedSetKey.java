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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.MapMaker;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collections;
import java.util.Map;

/** SkyKey for {@code NestedSet<Artifact>}. */
@AutoCodec
public class ArtifactNestedSetKey implements SkyKey {
  // Use a map instead of a classic Interner to avoid creating a wrapper object just to compare
  // equality. This is valid because rawChildren equality totally governs ArtifactNestedSetKey
  // equality. If that changes, this must change.
  private static final Map<Object, ArtifactNestedSetKey> interner =
      new MapMaker().weakValues().makeMap();

  private final Object rawChildren;

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.ARTIFACT_NESTED_SET;
  }

  private ArtifactNestedSetKey(Object rawChildren) {
    checkState(rawChildren instanceof Object[] || rawChildren instanceof Artifact);
    this.rawChildren = rawChildren;
  }

  /**
   * We use Object instead of NestedSet to store children as a measure to save memory, and to be
   * consistent with the implementation of NestedSet.
   *
   * @param rawChildren the underlying members of the nested set.
   */
  @AutoCodec.Instantiator
  public static ArtifactNestedSetKey create(Object rawChildren) {
    return interner.computeIfAbsent(rawChildren, ArtifactNestedSetKey::new);
  }

  @VisibleForTesting
  public Object getRawChildrenForTesting() {
    return rawChildren;
  }

  @Override
  public int hashCode() {
    return rawChildren.hashCode();
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }

    if (!(that instanceof ArtifactNestedSetKey)) {
      return false;
    }

    Object theirRawChildren = ((ArtifactNestedSetKey) that).rawChildren;
    if (rawChildren == theirRawChildren) {
      return true;
    }

    if (rawChildren instanceof Artifact && theirRawChildren instanceof Artifact) {
      return rawChildren.equals(theirRawChildren);
    }

    return false;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("rawChildren", NestedSet.childrenToString(rawChildren))
        .toString();
  }

  /**
   * Return the set of transitive members.
   *
   * <p>This refers to the transitive members after any inlining that might have happened at
   * construction of the nested set.
   *
   * <p>TODO(b/142232950) Investigate the potential additional load on GC.
   */
  Iterable<Object> transitiveMembers() {
    if (!(rawChildren instanceof Object[])) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Object> listBuilder = new ImmutableList.Builder<>();
    for (Object c : (Object[]) rawChildren) {
      if (c instanceof Object[]) {
        listBuilder.add(c);
      }
    }
    return listBuilder.build();
  }

  /**
   * Return the set of direct members.
   *
   * <p>This refers to the direct members after any inlining that might have happened at
   * construction of the nested set.
   *
   * <p>TODO(b/142232950) Investigate the potential additional load on GC.
   */
  Iterable<SkyKey> directKeys() {
    if (!(rawChildren instanceof Object[])) {
      return Collections.singletonList(Artifact.key((Artifact) rawChildren));
    }
    ImmutableList.Builder<SkyKey> listBuilder = new ImmutableList.Builder<>();
    for (Object c : (Object[]) rawChildren) {
      if (!(c instanceof Object[])) {
        listBuilder.add(Artifact.key((Artifact) c));
      }
    }
    return listBuilder.build();
  }
}
