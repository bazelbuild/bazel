// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Collection;

/** Helper functions for dealing with {@link Artifacts} */
public final class Artifacts {

  private Artifacts() {}

  public static void addToFingerprint(Fingerprint fp, Artifact artifact) {
    fp.addString(artifact.getExecPathString());
  }

  // TODO(bazel-team): Add option to sort collection of artifacts before adding to fingerprint?
  /** Appends a description of a complete collection of {@link Artifact} to the fingerprint. */
  public static void addToFingerprint(Fingerprint fp, Collection<Artifact> artifacts) {
    fp.addInt(artifacts.size());
    for (Artifact artifact : artifacts) {
      addToFingerprint(fp, artifact);
    }
  }
}
