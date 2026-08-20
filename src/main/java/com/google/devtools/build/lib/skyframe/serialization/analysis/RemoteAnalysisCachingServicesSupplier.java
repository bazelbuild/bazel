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

import static java.util.Objects.requireNonNull;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.devtools.build.lib.runtime.BlazeService;
import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.SkycacheMetadataParams;
import com.google.devtools.build.lib.util.SerializedAbruptExitException;
import com.google.devtools.common.options.OptionsProvider;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
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
 *
 * <p>Skybridge: this is the main boundary between the SC and the LC for Skycache.
 */
@SkybridgeInterface
public interface RemoteAnalysisCachingServicesSupplier extends BlazeService {

  /**
   * Service definitions and parameters depend on the command options, which are allowed to vary
   * per-command.
   *
   * <p>This method updates the services and parameters when the relevant flags change.
   */
  default void configure(
      OptionsProvider optionsProvider,
      RemoteAnalysisCacheMode mode,
      @Nullable ClientId clientId,
      String buildId)
      throws SerializedAbruptExitException {
    // Does nothing by default.
  }

  /** A specialized version of {@link #configure} for the dump command. */
  default void configureForDebugging(
      String remoteAnalysisDebugEntries,
      RemoteAnalysisCacheMode mode,
      ClientId clientId,
      String buildId)
      throws SerializedAbruptExitException {
    // Does nothing by default.
  }

  /**
   * Gets or creates the {@link FingerprintValueStore},
   *
   * <p>This may entail I/O so it is wrapped in a future.
   */
  @Nullable // null if remote analysis caching is not enabled
  ListenableFuture<? extends FingerprintValueStore> getFingerprintValueStore();

  /**
   * Gets or creates the analysis cache service interface.
   *
   * <p>This may entail I/O so it is wrapped in a future.
   */
  @Nullable // null if frontier-style invalidation is used instead of the cache service
  default ListenableFuture<? extends RemoteAnalysisCacheClient> getAnalysisCacheClient() {
    return null;
  }

  @Nullable
  default ListenableFuture<? extends RemoteAnalysisMetadataWriter> getMetadataWriter() {
    return null;
  }

  @Nullable
  SafeExecutor getCommandExecutor();

  /**
   * Gets the parameters for querying and updating Skycache metadata.
   *
   * <p>Returns null if metadata queries are not enabled.
   */
  @Nullable
  default SkycacheMetadataParams getSkycacheMetadataParams() {
    return null;
  }

  /** Represents a remote service peer. */
  public static final class Peer {
    private final String serviceName;
    private final String id;

    public Peer(String serviceName, String id) {
      this.serviceName = requireNonNull(serviceName);
      this.id = requireNonNull(id);
    }

    public String serviceName() {
      return serviceName;
    }

    public String id() {
      return id;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Peer peer)) {
        return false;
      }
      return serviceName.equals(peer.serviceName) && id.equals(peer.id);
    }

    @Override
    public int hashCode() {
      return Objects.hash(serviceName, id);
    }

    @Override
    public String toString() {
      return "Peer[serviceName=" + serviceName + ", id=" + id + "]";
    }
  }

  /**
   * Returns the map of backend peers and request counts connected during the current command, if
   * any.
   */
  @Nullable
  default Map<Peer, AtomicLong> getPeers() {
    return null;
  }

  /** Relinquishes any underlying resource that is scoped to the current command. */
  void resetCommandState();

  /** Relinquishes any global, server-lifetime resources (like cached channels). */
  default void blazeShutdown() {}
}
