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
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions.RemoteAnalysisCacheMode;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An interface providing the functionalities used for analysis caching serialization and
 * deserialization.
 */
public interface RemoteAnalysisCachingDependenciesProvider
    extends RemoteAnalysisCacheReaderDepsProvider {
  void setTopLevelBuildOptions(BuildOptions buildOptions);

  void queryMetadataAndMaybeBailout() throws InterruptedException;

  /**
   * Returns the set of SkyKeys to be invalidated.
   *
   * <p>May call the remote analysis cache to get the set of keys to invalidate.
   */
  Set<SkyKey> lookupKeysToInvalidate(
      ImmutableSet<SkyKey> keysToLookup,
      RemoteAnalysisCachingServerState remoteAnalysisCachingState)
      throws InterruptedException;

  default boolean bailedOut() {
    return false;
  }

  void computeSelectionAndMinimizeMemory(InMemoryGraph graph);

  Collection<Label> getTopLevelTargets();

  boolean shouldMinimizeMemory();

  /** A stub dependencies provider for when analysis caching is disabled. */
  final class DisabledDependenciesProvider
      implements RemoteAnalysisCachingDependenciesProvider, RemoteAnalysisCacheReaderDepsProvider {

    public static final DisabledDependenciesProvider INSTANCE = new DisabledDependenciesProvider();

    private DisabledDependenciesProvider() {}

    @Override
    public RemoteAnalysisCacheMode mode() {
      return RemoteAnalysisCacheMode.OFF;
    }

    @Override
    public boolean isRetrievalEnabled() {
      return false;
    }

    @Override
    public FrontierNodeVersion getSkyValueVersion() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ObjectCodecs getObjectCodecs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public FingerprintValueService getFingerprintValueService() {
      throw new UnsupportedOperationException();
    }

    @Override
    public RemoteAnalysisCacheClient getAnalysisCacheClient() {
      throw new UnsupportedOperationException();
    }

    @Override
    @Nullable
    public RemoteAnalysisJsonLogWriter getJsonLogWriter() {
      return null;
    }

    @Override
    public void recordRetrievalResult(RetrievalResult retrievalResult, SkyKey key) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void recordSerializationException(SerializationException e, SkyKey key) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void setTopLevelBuildOptions(BuildOptions buildOptions) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void queryMetadataAndMaybeBailout() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableSet<SkyKey> lookupKeysToInvalidate(
        ImmutableSet<SkyKey> keysToLookup,
        RemoteAnalysisCachingServerState remoteAnalysisCachingState) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void computeSelectionAndMinimizeMemory(InMemoryGraph graph) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Collection<Label> getTopLevelTargets() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean shouldMinimizeMemory() {
      throw new UnsupportedOperationException();
    }
  }
}
