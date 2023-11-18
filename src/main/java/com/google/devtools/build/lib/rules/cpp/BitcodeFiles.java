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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Wrapper around a map of bitcode files for purposes of caching its fingerprint. */
final class BitcodeFiles {

  private final NestedSet<Artifact> files;
  @Nullable private volatile byte[] fingerprint = null;

  BitcodeFiles(NestedSet<Artifact> files) {
    this.files = files;
  }

  NestedSet<Artifact> getFiles() {
    return files;
  }

  /** Helper function to get a map from path to artifact */
  ImmutableMap<PathFragment, Artifact> getFilesArtifactPathMap() {
    return getFiles().toList().stream()
        .collect(toImmutableMap(Artifact::getExecPath, artifact -> artifact));
  }

  void addToFingerprint(Fingerprint fp) {
    if (fingerprint == null) {
      synchronized (this) {
        if (fingerprint == null) {
          fingerprint = computeFingerprint();
        }
      }
    }
    fp.addBytes(fingerprint);
  }

  private byte[] computeFingerprint() {
    Fingerprint fp = new Fingerprint();
    for (Artifact path : files.toList()) {
      fp.addPath(path.getExecPath());
    }
    return fp.digestAndReset();
  }
}
