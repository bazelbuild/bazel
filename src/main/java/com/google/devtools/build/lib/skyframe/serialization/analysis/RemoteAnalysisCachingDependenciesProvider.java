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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions.RemoteAnalysisCacheMode;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Set;
import java.util.function.Supplier;

/**
 * An interface providing the functionalities used for analysis caching serialization and
 * deserialization.
 */
public interface RemoteAnalysisCachingDependenciesProvider {
  RemoteAnalysisCacheMode mode();

  void queryMetadataAndMaybeBailout() throws InterruptedException;

  /**
   * Returns the set of SkyKeys to be invalidated.
   *
   * <p>May call the remote analysis cache to get the set of keys to invalidate.
   */
  Set<SkyKey> lookupKeysToInvalidate(
      Supplier<ImmutableSet<SkyKey>> keysToLookupSupplier,
      RemoteAnalysisCachingServerState remoteAnalysisCachingState)
      throws InterruptedException;

  default boolean bailedOut() {
    return false;
  }

  void computeSelectionAndMinimizeMemory(InMemoryGraph graph);

  boolean shouldMinimizeMemory();



}
