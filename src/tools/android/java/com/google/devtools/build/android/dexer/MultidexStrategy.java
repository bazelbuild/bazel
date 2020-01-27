// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

/**
 * Strategies for outputting multiple {@code .dex} files supported by {@link DexFileMerger}.
 */
public enum MultidexStrategy {
  /** Create exactly one .dex file.  The operation will fail if .dex limits are exceeded. */
  OFF,
  /** Create exactly one &lt;prefixN&gt;.dex file with N taken from the (single) input archive. */
  GIVEN_SHARD,
  /**
   * Assemble .dex files similar to {@link com.android.dx.command.dexer.Main dx}, with all but one
   * file as large as possible.
   */
  MINIMAL,
  /**
   * Allow some leeway and sometimes use additional .dex files to speed up processing.  This option
   * exists to give flexibility but it often (or always) may be identical to {@link #MINIMAL}.
   */
  BEST_EFFORT;

  public boolean isMultidexAllowed() {
    switch (this) {
      case OFF:
      case GIVEN_SHARD:
        return false;
      case MINIMAL:
      case BEST_EFFORT:
        return true;
    }
    throw new AssertionError("Unknown: " + this);
  }
}
