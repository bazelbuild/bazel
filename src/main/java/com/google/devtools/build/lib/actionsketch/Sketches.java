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

import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import java.math.BigInteger;

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
        newHasher().putUnencodedChars(action.getKey(keyContext, /*artifactExpander=*/ null));
    for (Artifact output : action.getOutputs()) {
      hasher.putUnencodedChars(output.getExecPath().getPathString());
    }
    return fromHashCode(hasher.hash());
  }

  public static Hasher newHasher() {
    return Hashing.murmur3_128().newHasher();
  }

  private Sketches() {}
}
