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

package com.google.devtools.build.lib.actionsketch;

import static java.lang.Math.max;

import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import java.math.BigInteger;
import java.nio.ByteBuffer;

/** Utilities for dealing with {@link ActionSketch} sketches. */
public class Sketches {
  public static BigInteger fromHashCode(HashCode hashCode) {
    return new BigInteger(/*signum=*/ 1, hashCode.asBytes());
  }

  /**
   * Compute the hash of the direct action key for the given action, including the names of its
   * output files.
   */
  public static BigInteger computeActionKey(
      ActionAnalysisMetadata action, ActionKeyContext keyContext) throws InterruptedException {
    Hasher hasher =
        newHashOnlyHasher()
            .putUnencodedChars(action.getKey(keyContext, /*artifactExpander=*/ null));
    for (Artifact output : action.getOutputs()) {
      hasher.putUnencodedChars(output.getExecPath().getPathString());
    }
    return fromHashCode(hasher.hash());
  }

  private static Hasher newHashOnlyHasher() {
    return Hashing.murmur3_128().newHasher();
  }

  public static HashAndVersionTracker newHasher() {
    return new HashAndVersionTrackerImpl(newHashOnlyHasher());
  }

  /** Simple interface for accumulating elements for a hash+version pair. */
  public interface HashAndVersionTracker {
    HashAndVersionTracker putUnencodedChars(CharSequence chars);

    HashAndVersionTracker putVersion(long l);

    HashAndVersionTracker putBytes(ByteBuffer bytes);

    HashAndVersion hashAndVersion();
  }

  private static class HashAndVersionTrackerImpl implements HashAndVersionTracker {
    private final Hasher hasher;
    private long version = 0;

    private HashAndVersionTrackerImpl(Hasher hasher) {
      this.hasher = hasher;
    }

    @Override
    public HashAndVersion hashAndVersion() {
      return HashAndVersion.create(new BigInteger(1, hasher.hash().asBytes()), version);
    }

    @Override
    public HashAndVersionTracker putUnencodedChars(CharSequence charSequence) {
      hasher.putUnencodedChars(charSequence);
      return this;
    }

    @Override
    public HashAndVersionTracker putVersion(long v) {
      hasher.putLong(v);
      if (v != -1 && version != -1) {
        version = max(version, v);
      } else {
        version = -1;
      }
      return this;
    }

    @Override
    public HashAndVersionTracker putBytes(ByteBuffer bytes) {
      hasher.putBytes(bytes);
      return this;
    }
  }

  private Sketches() {}
}
