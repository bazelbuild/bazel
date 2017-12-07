// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetFingerprintCache;
import com.google.devtools.build.lib.util.Fingerprint;

/** Contains state that aids in action key computation via {@link AbstractAction#computeKey}. */
public class ActionKeyContext {
  private static final class ArtifactNestedSetFingerprintCache
      extends NestedSetFingerprintCache<Artifact> {
    @Override
    protected void addItemFingerprint(Fingerprint fingerprint, Artifact item) {
      fingerprint.addPath(item.getExecPath());
    }
  }

  private final ArtifactNestedSetFingerprintCache artifactNestedSetFingerprintCache =
      new ArtifactNestedSetFingerprintCache();

  public void addArtifactsToFingerprint(Fingerprint fingerprint, NestedSet<Artifact> artifacts) {
    artifactNestedSetFingerprintCache.addNestedSetToFingerprint(fingerprint, artifacts);
  }

  public void clear() {
    artifactNestedSetFingerprintCache.clear();
  }
}
