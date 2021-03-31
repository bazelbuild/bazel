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
package com.google.devtools.build.lib.runtime;

import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import javax.annotation.concurrent.ThreadSafe;

/** Conversion of paths to URIs. */
@ThreadSafe
public class CountingArtifactGroupNamer implements ArtifactGroupNamer {

  private final ConcurrentMap<NestedSet.Node, LatchedGroupName> nodeNames =
      new ConcurrentHashMap<>();

  @Override
  public NamedSetOfFilesId apply(NestedSet.Node id) {
    LatchedGroupName name = nodeNames.get(id);
    if (name == null) {
      return null;
    }
    return NamedSetOfFilesId.newBuilder().setId(name.getName()).build();
  }

  /**
   * If the {@link NestedSet} has no name already, return a new name for it. Return null otherwise.
   */
  public LatchedGroupName maybeName(NestedSet<?> set) {
    NestedSet.Node id = set.toNode();
    LatchedGroupName existingGroupName;
    LatchedGroupName newGroupName;
    // synchronized necessary only to ensure node names are chosen uniquely and compactly.
    // TODO(adgar): consider dropping compactness and unconditionally increment an AtomicLong to
    // pick unique node names.
    synchronized (this) {
      newGroupName = new LatchedGroupName(nodeNames.size());
      existingGroupName = nodeNames.putIfAbsent(id, newGroupName);
    }
    if (existingGroupName != null) {
      existingGroupName.waitUntilWritten();
      return null;
    }
    return newGroupName;
  }

  /**
   * A name for a {@code NestedSet<?>} that the constructor must {@link #close()} after the set is
   * written, allowing all other consumers to {@link #waitUntilWritten()}.
   */
  public static class LatchedGroupName implements AutoCloseable {
    private final CountDownLatch latch;
    private final int name;

    public LatchedGroupName(int name) {
      this.latch = new CountDownLatch(1);
      this.name = name;
    }

    @Override
    public void close() {
      latch.countDown();
    }

    String getName() {
      return Integer.toString(name);
    }

    private void waitUntilWritten() {
      Uninterruptibles.awaitUninterruptibly(latch);
    }
  }
}
