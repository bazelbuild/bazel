// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import java.io.IOException;
import java.util.NoSuchElementException;

/**
 * A cache that can lookup a {@link SpawnResult} given a {@link Spawn}, and can also upload the
 * results of an executed spawn to the cache.
 *
 * <p>This is an experimental interface to implement caching with sandboxed local execution.
 */
public interface SpawnCache extends ActionContext {
  /** A no-op implementation that has no result, and performs no upload. */
  public static CacheHandle NO_RESULT_NO_STORE =
      new CacheHandle() {
        @Override
        public boolean hasResult() {
          return false;
        }

        @Override
        public SpawnResult getResult() {
          throw new NoSuchElementException();
        }

        @Override
        public boolean willStore() {
          return false;
        }

        @Override
        public void store(SpawnResult result) throws InterruptedException, IOException {
          // Do nothing.
        }
      };

  /**
   * Helper method to create a {@link CacheHandle} from a successful {@link SpawnResult} instance.
   */
  public static CacheHandle success(SpawnResult result) {
    return new CacheHandle() {
      @Override
      public boolean hasResult() {
        return true;
      }

      @Override
      public SpawnResult getResult() {
        return result;
      }

      @Override
      public boolean willStore() {
        return false;
      }

      @Override
      public void store(SpawnResult result) throws InterruptedException, IOException {
        throw new IllegalStateException();
      }
    };
  }

  /** A no-op spawn cache. */
  public static class NoSpawnCache implements SpawnCache {
    @Override
    public CacheHandle lookup(Spawn spawn, SpawnExecutionContext context) {
      return SpawnCache.NO_RESULT_NO_STORE;
    }

    private NoSpawnCache() {}
  }

  /** A no-op implementation that has no results and performs no stores. */
  public static SpawnCache NO_CACHE = new NoSpawnCache();

  /**
   * This object represents both a successful and an unsuccessful cache lookup.
   *
   * <p>If {@link #hasResult} returns true, then {@link #getResult} returns a non-null instance.
   * Otherwise, if {@link #hasResult} returns false, then {@link #getResult} throws an {@link
   * IllegalStateException}.
   *
   * <p>If {@link #willStore} returns true, then {@link #store} may be called to upload the result
   * to the cache after successful execution. Otherwise, if {@link #willStore} returns false, then
   * {@link #store} throws an {@link IllegalStateException}.
   */
  interface CacheHandle {
    /** Returns whether the cache lookup was successful. */
    boolean hasResult();

    /**
     * Returns the cached result.
     *
     * @throws NoSuchElementException if there is no result in this cache entry
     */
    SpawnResult getResult();

    /**
     * Returns true if the store call will actually do work. Use this to avoid unnecessary work
     * before store if it won't do anything.
     */
    boolean willStore();

    /**
     * Called after successful {@link Spawn} execution, which may or may not store the result in the
     * cache.
     *
     * <p>A cache may silently return from a failed store operation. We recommend to err on the side
     * of raising an exception rather than returning silently, and to offer command-line flags to
     * tweak this default policy as needed.
     *
     * <p>If the current thread is interrupted, then this method should return as quickly as
     * possible with an {@link InterruptedException}.
     */
    void store(SpawnResult result) throws ExecException, InterruptedException, IOException;
  }

  /**
   * Perform a spawn lookup. This method is similar to {@link SpawnRunner#exec}, taking the same
   * parameters and being allowed to throw the same exceptions. The intent for this method is to
   * compute a cache lookup key for the given spawn, looking it up in an implementation-dependent
   * cache (can be either on the local or remote machine), and returning a non-null {@link
   * CacheHandle} instance.
   *
   * <p>If the lookup was successful, this method should write the cached outputs to their
   * corresponding output locations in the output tree, as well as stdout and stderr, after
   * notifying {@link SpawnExecutionContext#lockOutputFiles}.
   *
   * <p>If the lookup was unsuccessful, this method can return a {@link CacheHandle} instance that
   * has no result, but uploads the results of the execution to the cache. The reason for a callback
   * object is for the cache to store expensive intermediate values (such as the cache key) that are
   * needed both for the lookup and the subsequent store operation.
   *
   * <p>The lookup must not succeed for non-cachable spawns. See {@link Spawns#mayBeCached()} and
   * {@link Spawns#mayBeCachedRemotely}.
   *
   * <p>Note that cache stores may be disabled, in which case the returned {@link CacheHandle}
   * instance's {@link CacheHandle#store} is a no-op.
   */
  CacheHandle lookup(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, IOException, InterruptedException, ForbiddenActionInputException;

  /**
   * Returns whether this cache implementation makes sense to use together with dynamic execution.
   *
   * <p>A cache that's part of the remote system used for dynamic execution should not also be used
   * for the local speculative execution. However, a local cache or a separate remote cache-only
   * system would be.
   */
  default boolean usefulInDynamicExecution() {
    return true;
  }
}
