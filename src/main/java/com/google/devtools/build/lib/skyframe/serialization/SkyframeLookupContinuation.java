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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.LookupAbandonedException;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.PeerFailedException;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.SkyframeLookup;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayDeque;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/**
 * A partial deserialization result that may require one or more Skyframe lookups to complete.
 *
 * <p>This class is designed to reside in {@link SkyKeyComputeState}. In particular, note that
 * {@link #abandon} should be called.
 */
public final class SkyframeLookupContinuation {
  private final ArrayDeque<SkyframeLookup<?>> skyframeLookups;
  private final ListenableFuture<?> result;

  private State state = State.LOOKUP;

  SkyframeLookupContinuation(
      ArrayDeque<SkyframeLookup<?>> skyframeLookups, ListenableFuture<?> result) {
    this.skyframeLookups = skyframeLookups;
    this.result = result;
  }

  private static enum State {
    /** Start state that performs initial Skyframe lookups for any keys. */
    LOOKUP,
    /**
     * State that is ready to resume from restart.
     *
     * <p>If this state is reached, all values should be present in Skyframe.
     */
    RESUME,
    /**
     * Marker indicating completion.
     *
     * <p>It's an error to call {@link #process} from this state.
     */
    ENDED;
  }

  /**
   * Performs the next deserialization processing step.
   *
   * <p>Clients may need to call this twice, with the 2nd call being after a Skyframe restart.
   *
   * @return a future containing the deserialization result (which could be pending deserialization
   *     occurring in other threads) or null if a Skyframe restart is needed
   */
  @Nullable
  public ListenableFuture<?> process(LookupEnvironment env)
      throws InterruptedException, SkyframeDependencyException {
    return switch (state) {
      case LOOKUP -> doLookup(env);
      case RESUME -> resume(env);
      case ENDED -> throw new IllegalStateException("already ended: " + result);
    };
  }

  /**
   * Performs state cleanup.
   *
   * <p>This must be called if the lookups cannot be completed, for example, if {@link
   * SkyKeyComputeState#close} is called on any containing compute state or if there's an error.
   */
  public void abandon(LookupAbandonedException exception) {
    for (SkyframeLookup<?> lookup : skyframeLookups) {
      lookup.abandon(exception);
    }
    skyframeLookups.clear();
  }

  @VisibleForTesting
  ArrayDeque<SkyframeLookup<?>> getSkyframeLookupsForTesting() {
    return skyframeLookups;
  }

  /**
   * Performs any needed Skyframe lookups.
   *
   * @return a future containing the deserialization result or null if a Skyframe restart is needed
   */
  @Nullable
  private ListenableFuture<?> doLookup(LookupEnvironment env)
      throws InterruptedException, SkyframeDependencyException {
    if (skyframeLookups.isEmpty()) {
      this.state = State.ENDED;
      return result;
    }

    // TODO: b/335901349 - consider implementing an optimized codepath for unary lookups.
    SkyframeLookupResult lookupResult;
    try {
      // This is the only method that can throw InterruptedException.
      lookupResult =
          env.getValuesAndExceptions(Iterables.transform(skyframeLookups, SkyframeLookup::getKey));
    } catch (InterruptedException e) {
      abandon(new LookupAbandonedException(e));
      Thread.currentThread().interrupt(); // Restores the interrupted status.
      throw e;
    }
    int lookupCount = skyframeLookups.size();
    for (int i = 0; i < lookupCount; i++) {
      SkyframeLookup<?> lookup = skyframeLookups.pollFirst();
      if (lookupResult.queryDep(lookup.getKey(), lookup)) {
        throwDependencyExceptionIfFailed(lookup);
      } else {
        // Consumes lookups from the front of the queue and keeps any that are not available by
        // appending them to the back. The `lookupCount` loop bound ensures the re-appended lookups
        // won't be consumed. Reusing `skyframeLookups` to store the lookups to perform after
        // Skyframe restart reduces churn.
        skyframeLookups.addLast(lookup); // value not available in Skyframe
      }
    }
    if (skyframeLookups.isEmpty()) { // all lookups succeeded
      this.state = State.ENDED;
      return result;
    }
    this.state = State.RESUME;
    return null; // Skyframe restart needed
  }

  /**
   * Resumes deserialization after a Skyframe restart by consuming pending values.
   *
   * @return a future containing the deserialization result
   */
  private ListenableFuture<?> resume(LookupEnvironment env) throws SkyframeDependencyException {
    // There was a Skyframe restart. Everything that was requested should be available now. This
    // method should not be reachable by error bubbling because it can only be reached by
    // pre-existing SkyKeyComputeState, which is evicted before error bubbling.

    SkyframeLookupResult lookupResult = env.getLookupHandleForPreviouslyRequestedDeps();
    for (SkyframeLookup<?> lookup : skyframeLookups) {
      SkyKey key = lookup.getKey();
      checkState(
          lookupResult.queryDep(key, lookup),
          "previously requested key %s missing from Skyframe after restart",
          key);
      throwDependencyExceptionIfFailed(lookup);
    }
    skyframeLookups.clear();
    this.state = State.ENDED;
    return result;
  }

  private void throwDependencyExceptionIfFailed(SkyframeLookup<?> lookup)
      throws SkyframeDependencyException {
    if (!lookup.isFailed()) {
      return;
    }
    this.state = State.ENDED;
    try {
      var unused = Futures.getDone(lookup);
    } catch (ExecutionException e) {
      // In general, SkyframeLookups can contain either SkyframeDependencyExceptions or
      // LookupAbandonedExceptions. This is only reachable before any LookupAbandonedExceptions can
      // be propagated.
      var cause = (SkyframeDependencyException) e.getCause();
      abandon(new PeerFailedException(cause));
      throw cause;
    }
    throw new IllegalStateException("should have thrown an exception: " + lookup);
  }
}
