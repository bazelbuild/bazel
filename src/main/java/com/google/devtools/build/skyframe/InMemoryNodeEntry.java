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
package com.google.devtools.build.skyframe;

import java.util.Collection;
import javax.annotation.Nullable;

/**
 * A {@link NodeEntry} that is stored in memory.
 *
 * <p>Supports several {@link NodeEntry} methods without throwing {@link InterruptedException}.
 */
public interface InMemoryNodeEntry extends NodeEntry {

  /** Returns the {@link SkyKey} associated with this node. */
  SkyKey getKey();

  /** Whether this node stores edges (deps and rdeps). */
  boolean keepsEdges();

  /**
   * Returns the compressed {@link GroupedDeps} of direct deps. Can only be called if this node
   * {@link #isDone} and {@link #keepsEdges}.
   */
  @GroupedDeps.Compressed
  Object getCompressedDirectDepsForDoneEntry();

  @Override // Remove InterruptedException.
  SkyValue getValue();

  @Override // Remove InterruptedException.
  @Nullable
  SkyValue getValueMaybeWithMetadata();

  @Override // Remove InterruptedException.
  @Nullable
  SkyValue toValue();

  @Override // Remove InterruptedException.
  @Nullable
  ErrorInfo getErrorInfo();

  @Override // Remove InterruptedException.
  Iterable<SkyKey> getDirectDeps();

  @Override // Remove InterruptedException.
  Collection<SkyKey> getReverseDepsForDoneEntry();

  @Override // Remove InterruptedException.
  DependencyState addReverseDepAndCheckIfDone(@Nullable SkyKey reverseDep);

  @Override // Remove InterruptedException.
  @Nullable
  MarkedDirtyResult markDirty(DirtyType dirtyType);
}
