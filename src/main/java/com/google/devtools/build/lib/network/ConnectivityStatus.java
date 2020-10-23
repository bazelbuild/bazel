// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.network;

/** Defines connectivity problem types and their short warning messages. */
public final class ConnectivityStatus {
  /** Enumerates common connectivity statuses and their generic short warnings. */
  public enum Status {
    NO_CREDENTIALS("No credentials"),
    NO_NETWORK("No internet connection"),
    NOT_REACHABLE("Service not reachable"),
    OK("");

    /** Generic warning associated with this status. */
    public final String shortWarning;

    Status(String shortWarning) {
      this.shortWarning = shortWarning;
    }
  }

  /**
   * A string containing the names of the failing statuses, so that places that refer to them in a
   * static context (ie. help for options that take these values as inputs) can stay in sync if
   * {@link Status} is changed.
   */
  public static final String FAILING_STATUSES =
      "'NO_CREDENTIALS', 'NO_NETWORK', and 'NOT_REACHABLE'";

  /** Service-specific information for this status. */
  public String serviceInfo;

  /** Generic category type for this status, which contains a generic warning. */
  public final Status status;

  @Override
  public String toString() {
    return serviceInfo.isEmpty() ? status.shortWarning : status.shortWarning + ": " + serviceInfo;
  }

  /**
   * Constructs a connectivity status with a service-specific warning.
   *
   * @param serviceInfo service-specific information displayed or logged in addition to the status's
   *     short warning when this connectivityStatus is present.
   */
  public ConnectivityStatus(Status status, String serviceInfo) {
    this.status = status;
    this.serviceInfo = serviceInfo;
  }
}
