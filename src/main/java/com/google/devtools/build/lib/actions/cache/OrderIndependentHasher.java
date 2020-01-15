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
package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.util.Fingerprint;
import javax.annotation.Nullable;

/** Utility class for combining a set of (path, metadata) pairs into a single digest. */
public final class OrderIndependentHasher {
  private final Fingerprint fp = new Fingerprint();
  private final byte[] tmp = new byte[fp.getDigestLength()];
  private final byte[] digest = new byte[fp.getDigestLength()];

  /**
   * Add a artifact's path and metadata to the digest. This method must never be called twice with
   * the same arguments.
   */
  public void addArtifact(String execPath, @Nullable FileArtifactValue md) {
    fp.addString(execPath);
    if (md == null) {
      // Move along, nothing to see here.
    } else if (md.getDigest() != null) {
      fp.addBytes(md.getDigest());
    } else {
      // Use the timestamp if the digest is not present, but not both. Modifying a timestamp while
      // keeping the contents of a file the same should not cause rebuilds.
      fp.addLong(md.getModifiedTime());
    }
    fp.digestAndReset(tmp, 0, tmp.length);
    for (int i = 0; i < digest.length; i++) {
      digest[i] ^= tmp[i];
    }
  }

  /** Return the final combined digest. */
  public byte[] finish() {
    return digest;
  }
}
