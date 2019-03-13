// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.nativedeps;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import javax.annotation.Nullable;

/**
 * Container for the artifacts needed to load a native deps shared library. These
 * are the library itself as well as any symlinks needed to resolve its
 * dynamic link dependencies.
 */
@Immutable @ThreadSafe
public class NativeDepsRunfiles {
  /** An object that represents no native deps. */
  public static final NativeDepsRunfiles EMPTY = new NativeDepsRunfiles(null);

  @Nullable
  private final Artifact library;

  public NativeDepsRunfiles(@Nullable Artifact library) {
    this.library = library;
  }

  /**
   * Returns the native deps library (which may itself be a symlink to another library), may be
   * null.
   */
  @Nullable
  public Artifact getLibrary() {
    return library;
  }
}
