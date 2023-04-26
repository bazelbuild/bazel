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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.clock.Clock;

/** A {@link RemoteArtifactChecker} that checks the TTL of remote metadata. */
public class RemoteOutputChecker implements RemoteArtifactChecker {
  private final Clock clock;

  public RemoteOutputChecker(Clock clock) {
    this.clock = clock;
  }

  @Override
  public boolean shouldTrustRemoteArtifact(Artifact file, RemoteFileArtifactValue metadata) {
    return metadata.isAlive(clock.now());
  }
}
