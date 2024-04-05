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

package com.google.devtools.build.android.r8;

import javax.annotation.Nullable;

/** Simple collectors that don't collect any information. */
public enum NoWriteCollectors implements DependencyCollector {
  /** Singleton instance that does nothing. */
  NOOP,
  /**
   * Singleton instance that does nothing besides throwing if {@link #missingImplementedInterface}
   * is called.
   */
  FAIL_ON_MISSING {
    @Override
    public void missingImplementedInterface(String origin, String target) {
      throw new IllegalStateException(
          String.format(
              "Couldn't find interface %s on the classpath for desugaring %s", target, origin));
    }
  };

  @Override
  @Nullable
  public final byte[] toByteArray() {
    return null;
  }
}
