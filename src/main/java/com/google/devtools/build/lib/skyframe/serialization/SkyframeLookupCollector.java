// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.PeerFailedException;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.SkyframeLookup;
import java.util.ArrayDeque;
import javax.annotation.concurrent.GuardedBy;

/**
 * Tracks state pertaining to Skyframe lookups from deserialization.
 *
 * <p>The future completes once it can be determined that all Skyframe lookups are known.
 *
 * <p>This is shared across {@link SharedValueDeserializationContext} and transitive inner contexts
 * created by {@link SharedValueDeserializationContext#readValueForFingerprint}.
 */
final class SkyframeLookupCollector extends QuiescingFuture<ArrayDeque<SkyframeLookup<?>>> {
  /** Skyframe lookups required for deserialization. */
  private final ArrayDeque<SkyframeLookup<?>> skyframeLookups = new ArrayDeque<>();

  @GuardedBy("this")
  private PeerFailedException cause;

  /**
   * A notification that balances the pre-increment of {@link QuiescingFuture}.
   *
   * <p>The client must call this once. Must not be called before all initial calls to {@link
   * SharedValueDeserializationContext#readValueForFingerprint} occur.
   *
   * <p>{@link SharedValueDeserializationContext#getSharedValue} calls may recursively trigger more
   * fetches asynchronously which is fine as long as the parent child notification ordering
   * described in {@link QuiescingFuture} is followed.
   */
  void notifyFetchesInitialized() {
    decrement();
  }

  void notifyFetchStarting() {
    increment();
  }

  void notifyFetchDone() {
    decrement();
  }

  void notifyFetchException(Throwable t) {
    synchronized (this) {
      if (cause == null) {
        // If this is the first failure, captures it and abandons any previously collected lookups.
        cause = new PeerFailedException(t);
        for (SkyframeLookup<?> lookup : skyframeLookups) {
          lookup.abandon(cause);
        }
        skyframeLookups.clear();
      }
    }
    // The future fails fast here. Any lookups that are added after the failure are immediately
    // abandoned.
    notifyException(t);
  }

  @Override
  protected ArrayDeque<SkyframeLookup<?>> getValue() {
    return skyframeLookups;
  }

  synchronized void addLookup(SkyframeLookup<?> lookup) {
    if (cause != null) {
      lookup.abandon(cause); // Abandons any lookups added after the first error.
      return;
    }
    skyframeLookups.addLast(lookup);
  }
}
