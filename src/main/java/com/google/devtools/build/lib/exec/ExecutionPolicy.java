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
package com.google.devtools.build.lib.exec;

/**
 * Determines whether a Spawn is executable locally, remotely, or both.
 */
public final class ExecutionPolicy {
  private enum Locality {
    LOCAL_ONLY,
    REMOTE_ONLY,
    BOTH;
  }

  public static final ExecutionPolicy LOCAL_EXECUTION_ONLY =
      new ExecutionPolicy(Locality.LOCAL_ONLY);

  public static final ExecutionPolicy REMOTE_EXECUTION_ONLY =
      new ExecutionPolicy(Locality.REMOTE_ONLY);

  public static final ExecutionPolicy ANYWHERE = new ExecutionPolicy(Locality.BOTH);

  private final Locality locality;

  private ExecutionPolicy(Locality locality) {
    this.locality = locality;
  }

  public boolean canRunRemotelyOnly() {
    return locality == Locality.REMOTE_ONLY;
  }

  public boolean canRunRemotely() {
    return locality != Locality.LOCAL_ONLY;
  }

  public boolean canRunLocallyOnly() {
    return locality == Locality.LOCAL_ONLY;
  }

  public boolean canRunLocally() {
    return locality != Locality.REMOTE_ONLY;
  }

  @Override
  public String toString() {
    return locality.toString();
  }
}
