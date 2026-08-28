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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.actions.Artifact;
import java.util.Collection;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** Tracks whether an artifact was "consumed" by any action in the build. */
public final class ConsumedArtifactsTracker implements EphemeralCheckIfOutputConsumed {

  private final Set<Artifact> consumed = ConcurrentHashMap.newKeySet(524288);

  public ConsumedArtifactsTracker() {}

  /**
   * This method guarantees the best estimate before the execution of the action that would generate
   * this artifact. The return value is undefined afterwards.
   */
  @Override
  public boolean test(Artifact artifact) {
    return consumed.contains(artifact);
  }

  void unregisterOutputsAfterExecutionDone(Collection<Artifact> outputs) {
    for (Artifact artifact : outputs) {
      consumed.remove(representative(artifact));
    }
  }

  /** Register the provided artifact as "consumed". */
  void registerConsumedArtifact(Artifact artifact) {
    if (artifact.isSourceArtifact()) {
      return; // Source artifacts are never orphaned, so we don't track them.
    }
    consumed.add(representative(artifact));
  }

  private static Artifact representative(Artifact artifact) {
    // Tree artifact children are checked through the parent artifact.
    return artifact.hasParent() ? artifact.getParent() : artifact;
  }
}
