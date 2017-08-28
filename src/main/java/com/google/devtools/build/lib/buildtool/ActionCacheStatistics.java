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
package com.google.devtools.build.lib.buildtool;

import com.google.auto.value.AutoValue;
import java.time.Duration;

/** Statistics about the action cache during a single build. */
@AutoValue
public abstract class ActionCacheStatistics {
  /** Gets the time it took to save the action cache to disk. */
  public abstract Duration saveTime();

  /** Gets the size of the action cache in bytes as persisted on disk. */
  public abstract long sizeInBytes();

  /** Returns a new builder. */
  static Builder builder() {
    return new AutoValue_ActionCacheStatistics.Builder();
  }

  @AutoValue.Builder
  abstract static class Builder {
    /** Sets the time it took to save the action cache to disk. */
    abstract Builder setSaveTime(Duration duration);

    /** Sets the size of the action cache in bytes as persisted to disk. */
    abstract Builder setSizeInBytes(long sizeInBytes);

    /** Constructs and returns the {@link ActionCacheStatistics}. */
    abstract ActionCacheStatistics build();
  }
}
