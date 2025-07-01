// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.util.AbruptExitException;
import javax.annotation.Nullable;

/**
 * Supplies external services used by remote analysis caching.
 *
 * <p>This interface exists so its implementation can be injected, at the workspace level.
 *
 * <p>The services themselves depend only on command options. Clients must call {@link #configure}
 * prior to calling either of the service getters.
 *
 * <p>Updating parameters is not thread safe. This class assumes that such updates are performed
 * synchronously. Subsequent service get calls are thread safe.
 */
public interface RemoteAnalysisCachingServicesSupplier {

  /**
   * Service definitions and parameters depend on {@code options}, which are allowed to vary
   * per-command.
   *
   * <p>This method updates the services and parameters when the relevant flags change.
   */
  default void configure(RemoteAnalysisCachingOptions cachingOptions, @Nullable ClientId clientId)
      throws AbruptExitException {
    // Does nothing by default.
  }

  /**
   * Gets or creates the {@link FingerprintValueService},
   *
   * <p>This may entail I/O so it is wrapped in a future.
   */
  @Nullable // null if remote analysis caching is not enabled
  ListenableFuture<FingerprintValueService> getFingerprintValueService();

  /**
   * Gets or creates the analysis cache service interface.
   *
   * <p>This may entail I/O so it is wrapped in a future.
   */
  @Nullable // null if frontier-style invalidation is used instead of the cache service
  default ListenableFuture<RemoteAnalysisCacheClient> getAnalysisCacheClient() {
    return null;
  }
}
