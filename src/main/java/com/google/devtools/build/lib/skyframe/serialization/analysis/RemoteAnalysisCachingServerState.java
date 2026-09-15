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

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * State to track information pertinent to Skyframe nodes that were deserialized from the remote
 * analysis cache through the lifetime of the Bazel server (until clean/shutdown), that is, across
 * multiple invocations.
 */
public final class RemoteAnalysisCachingServerState {

  /** The {@link FrontierNodeVersion} */
  @Nullable private FrontierNodeVersion latestInvocationVersion;

  @Nullable private ClientId latestInvocationClientId;

  public RemoteAnalysisCachingServerState(
      @Nullable FrontierNodeVersion version, @Nullable ClientId clientId) {
    this.latestInvocationVersion = version;
    this.latestInvocationClientId = clientId;
  }

  /** Returns a {@link RemoteAnalysisCachingServerState} with empty/null fields. */
  public static RemoteAnalysisCachingServerState initializeEmpty() {
    return new RemoteAnalysisCachingServerState(/* version= */ null, /* clientId= */ null);
  }

  /** Returns {@link FrontierNodeVersion} of the latest (previous) invocation, if any. */
  @Nullable
  public FrontierNodeVersion version() {
    return latestInvocationVersion;
  }

  /**
   * Sets the {@link FrontierNodeVersion} of the remote analysis cache keys used in the current
   * invocation.
   *
   * <p>This will be used to determine invalidation during the next invocation.
   */
  public void setVersion(FrontierNodeVersion version) {
    this.latestInvocationVersion = version;
  }

  /** Returns the {@link ClientId} of the latest (previous) invocation, if any. */
  @Nullable
  public ClientId clientId() {
    return latestInvocationClientId;
  }

  /** Sets the {@link ClientId} of the current invocation. */
  public void setClientId(ClientId clientId) {
    this.latestInvocationClientId = clientId;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof RemoteAnalysisCachingServerState that)) {
      return false;
    }
    return Objects.equals(latestInvocationVersion, that.latestInvocationVersion)
        && Objects.equals(latestInvocationClientId, that.latestInvocationClientId);
  }

  @Override
  public int hashCode() {
    return Objects.hash(latestInvocationVersion, latestInvocationClientId);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("version", latestInvocationVersion)
        .add("clientId", latestInvocationClientId)
        .toString();
  }
}
