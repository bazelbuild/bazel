// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.collect.Comparators.max;
import static com.google.common.collect.Comparators.min;

import build.bazel.remote.execution.v2.ServerCapabilities;
import javax.annotation.Nullable;

/** Represents a range of the Remote Execution API that client supports. */
public class ClientApiVersion {
  private final ApiVersion low;
  private final ApiVersion high;

  public static final ClientApiVersion current =
      new ClientApiVersion(ApiVersion.low, ApiVersion.high);

  public ClientApiVersion(ApiVersion low, ApiVersion high) {
    this.low = low;
    this.high = high;
  }

  public ApiVersion getLow() {
    return low;
  }

  public ApiVersion getHigh() {
    return high;
  }

  public boolean isSupported(ApiVersion version) {
    return low.compareTo(version) <= 0 && high.compareTo(version) >= 0;
  }

  static class ServerSupportedStatus {
    private enum State {
      SUPPORTED,
      UNSUPPORTED,
      DEPRECATED,
    }

    private final String message;
    private final State state;
    private final ApiVersion highestSupportedVersion;

    private ServerSupportedStatus(State state, String message, ApiVersion highestSupportedVersion) {
      this.state = state;
      this.message = message;
      this.highestSupportedVersion = highestSupportedVersion;
    }

    public static ServerSupportedStatus supported(ApiVersion highestSupportedVersion) {
      return new ServerSupportedStatus(State.SUPPORTED, "", highestSupportedVersion);
    }

    public static ServerSupportedStatus unsupported(
        ApiVersion clientLow, ApiVersion clientHigh, ApiVersion serverLow, ApiVersion serverHigh) {
      return new ServerSupportedStatus(
          State.UNSUPPORTED,
          String.format(
              "The client supported API versions, %s to %s, is not supported by the server, %s to"
                  + " %s. Please switch to a different server or upgrade Bazel.",
              clientLow, clientHigh, serverLow, serverHigh),
          null);
    }

    public static ServerSupportedStatus deprecated(
        ApiVersion clientHigh, ApiVersion serverLow, ApiVersion serverHigh) {
      return new ServerSupportedStatus(
          State.DEPRECATED,
          String.format(
              "The highest API version Bazel support %s is deprecated by the server. "
                  + "Please upgrade to server's recommended version: %s to %s.",
              clientHigh, serverLow, serverHigh),
          clientHigh);
    }

    public String getMessage() {
      return message;
    }

    public ApiVersion getHighestSupportedVersion() {
      return highestSupportedVersion;
    }

    public boolean isSupported() {
      return state == State.SUPPORTED;
    }

    public boolean isDeprecated() {
      return state == State.DEPRECATED;
    }

    public boolean isUnsupported() {
      return state == State.UNSUPPORTED;
    }
  }

  // highestSupportedVersion compares the client's supported versions against the input low and high
  // versions and returns the highest supported version. If the client's supported versions are not
  // supported by the server, it returns null.
  @Nullable
  private ApiVersion highestSupportedVersion(ApiVersion serverLow, ApiVersion serverHigh) {
    var higestLow = max(this.low, serverLow);
    var lowestHigh = min(this.high, serverHigh);

    return higestLow.compareTo(lowestHigh) <= 0 ? lowestHigh : null;
  }

  public ServerSupportedStatus checkServerSupportedVersions(ServerCapabilities cap) {
    var serverLow = new ApiVersion(cap.getLowApiVersion());
    var serverHigh = new ApiVersion(cap.getHighApiVersion());

    var highest = highestSupportedVersion(serverLow, serverHigh);
    if (highest != null) {
      return ServerSupportedStatus.supported(highest);
    }

    var deprecated =
        cap.hasDeprecatedApiVersion() ? new ApiVersion(cap.getDeprecatedApiVersion()) : null;
    if (deprecated == null) {
      return ServerSupportedStatus.unsupported(this.low, this.high, serverLow, serverHigh);
    }

    highest = highestSupportedVersion(deprecated, serverHigh);
    if (highest != null) {
      return ServerSupportedStatus.deprecated(highest, serverLow, serverHigh);
    }

    return ServerSupportedStatus.unsupported(this.low, this.high, serverLow, serverHigh);
  }
}
