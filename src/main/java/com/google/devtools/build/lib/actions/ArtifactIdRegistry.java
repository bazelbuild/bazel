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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.MapMaker;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * A registry that keeps a map of artifacts to unique integer ids.
 */
class ArtifactIdRegistry implements ArtifactSerializer, ArtifactDeserializer {

  /**
   * A sequence of registered artifacts. The position in the list is the artifact's id.
   *
   * <p>Synchronized using {@link #artifactIdsLock}.
   */
  private final List<Artifact> serializedArtifactList = new ArrayList<>();

  /**
   * A map of artifacts to unique integer ids.
   *
   * <p>Writes to this map must be synchronized using {@link #artifactIdsLock}, in order to
   * maintain consistency with {@link #serializedArtifactList}.
   */
  private final ConcurrentMap<Artifact, Integer> serializedArtifactIds =
      new MapMaker().concurrencyLevel(1).makeMap();

  /**
   * A lock for keeping {@code serializedArtifactList} and {@code serializedArtifactIds} in sync.
   */
  private ReadWriteLock artifactIdsLock = new ReentrantReadWriteLock();

  ArtifactIdRegistry() {
  }

  @Override
  public int getArtifactId(Artifact artifact) {
    Integer artifactId = serializedArtifactIds.get(artifact);
    if (artifactId == null) {
      artifactId = assignArtifactId(artifact);
    }
    return artifactId;
  }

  private Integer assignArtifactId(Artifact artifact) {
    artifactIdsLock.writeLock().lock();
    try {
      Integer artifactId = serializedArtifactIds.get(artifact);
      if (artifactId == null) {
        artifactId = serializedArtifactList.size();
        serializedArtifactList.add(artifact);
        serializedArtifactIds.put(artifact, artifactId);
      }
      return artifactId;
    } finally {
      artifactIdsLock.writeLock().unlock();
    }
  }

  @Override
  public Artifact lookupArtifactById(int artifactId) {
    artifactIdsLock.readLock().lock();
    try {
      return serializedArtifactList.get(artifactId);
    } finally {
      artifactIdsLock.readLock().unlock();
    }
  }

  @Override
  public ImmutableList<Artifact> lookupArtifactsByIds(Iterable<Integer> artifactIds) {
    int size = Iterables.size(artifactIds);
    Artifact[] result = new Artifact[size];

    int i = 0;

    artifactIdsLock.readLock().lock();
    try {
      for (int artifactId : artifactIds) {
        result[i] = serializedArtifactList.get(artifactId);
        i++;
      }
    } finally {
      artifactIdsLock.readLock().unlock();
    }

    return ImmutableList.copyOf(result);
  }
}
