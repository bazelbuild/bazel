// Copyright 2018 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.ServerCapabilities;
import build.bazel.semver.SemVer;

/**
 * Represents a version of the Remote Execution API.
 */
public class ApiVersion implements Comparable<ApiVersion> {
  public final int major;
  public final int minor;
  public final int patch;
  public final String prerelease;

  // The current version of the Remote Execution API. This field will need to be updated
  // together with all version changes.
  public static final ApiVersion current = new ApiVersion(SemVer.newBuilder().setMajor(2).build());

  public ApiVersion(int major, int minor, int patch, String prerelease) {
    this.major = major;
    this.minor = minor;
    this.patch = patch;
    this.prerelease = prerelease;
  }

  public ApiVersion(SemVer semver) {
    this(semver.getMajor(), semver.getMinor(), semver.getPatch(), semver.getPrerelease());
  }

  @Override
  public String toString() {
    if (!prerelease.isEmpty()) {
      return prerelease;
    }
    StringBuilder builder = new StringBuilder();
    builder.append(major);
    builder.append(".");
    builder.append(minor);
    if (patch != 0) {
      builder.append(".");
      builder.append(patch);
    }
    return builder.toString();
  }

  public SemVer toSemVer() {
    return SemVer.newBuilder()
        .setMajor(major)
        .setMinor(minor)
        .setPatch(patch)
        .setPrerelease(prerelease)
        .build();
  }

  /**
   * Compares the current API version to another API version.
   *
   * @param other the API version to compare to.
   * @return 0 if the API versions are equal, a number less than 0 if the current version is earlier
   *     than other, and a number greater than 0 if the current version is later than other. It is
   *     assumed that all prerelease versions are earlier than all released versions.
   */
  @Override
  public int compareTo(ApiVersion other) {
    if (!prerelease.isEmpty()) {
      if (other.prerelease.isEmpty()) {
        return -1;
      }
      return prerelease.compareTo(other.prerelease);
    }
    if (!other.prerelease.isEmpty()) {
      return 1;
    }
    if (major != other.major) {
      return Integer.compare(major, other.major);
    }
    if (minor != other.minor) {
      return Integer.compare(minor, other.minor);
    }
    return Integer.compare(patch, other.patch);
  }

  static class ServerSupportedStatus {
    private enum State {
      SUPPORTED,
      UNSUPPORTED,
      DEPRECATED,
    }
    private final String message;
    private final State state;

    private ServerSupportedStatus(State state, String message) {
      this.state = state;
      this.message = message;
    }

    public static ServerSupportedStatus supported() {
      return new ServerSupportedStatus(State.SUPPORTED, "");
    }

    public static ServerSupportedStatus unsupported(
        ApiVersion curr, ApiVersion lowApiVersion, ApiVersion highApiVersion) {
      return new ServerSupportedStatus(
          State.UNSUPPORTED,
          String.format(
              "The API version %s is not supported by the server. "
                  + "Please switch to a supported version: %s to %s.",
              curr, lowApiVersion, highApiVersion));
    }

    public static ServerSupportedStatus deprecated(
        ApiVersion curr, ApiVersion lowApiVersion, ApiVersion highApiVersion) {
      return new ServerSupportedStatus(
          State.DEPRECATED,
          String.format(
              "The API version %s is deprecated by the server. "
                  + "Please upgrade to a recommended version: %s to %s.",
              curr, lowApiVersion, highApiVersion));
    }

    public String getMessage() {
      return message;
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

  public ServerSupportedStatus checkServerSupportedVersions(ServerCapabilities cap) {
    ApiVersion deprecated =
        cap.hasDeprecatedApiVersion() ? new ApiVersion(cap.getDeprecatedApiVersion()) : null;
    ApiVersion low = new ApiVersion(cap.getLowApiVersion());
    ApiVersion high = new ApiVersion(cap.getHighApiVersion());
    if (deprecated != null && compareTo(deprecated) >= 0 && compareTo(low) < 0) {
      return ServerSupportedStatus.deprecated(this, low, high);
    }
    if (compareTo(low) < 0 || compareTo(high) > 0) {
      return ServerSupportedStatus.unsupported(this, low, high);
    }
    return ServerSupportedStatus.supported();
  }
}
