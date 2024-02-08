// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.metrics;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import java.time.Instant;

/** Contains a snapshot of the resource usage of multiple processes. */
@AutoValue
public abstract class ResourceSnapshot {
  /** Overall memory consumption by all descendant processes including initial process. */
  public abstract ImmutableMap<Long, Integer> getPidToMemoryInKb();

  /** Time when this snapshot was collected. */
  public abstract Instant getCollectionTime();

  public static ResourceSnapshot create(
      ImmutableMap<Long, Integer> pidToMemoryInKb, Instant collectionTime) {
    return new AutoValue_ResourceSnapshot(pidToMemoryInKb, collectionTime);
  }

  public static ResourceSnapshot createEmpty(Instant collectionTime) {
    return new AutoValue_ResourceSnapshot(ImmutableMap.of(), collectionTime);
  }
}
