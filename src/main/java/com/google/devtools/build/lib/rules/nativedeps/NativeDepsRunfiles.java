// Copyright 2015 Google Inc. All rights reserved.
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

import java.util.List;

import javax.annotation.Nullable;

/**
 * Container for the artifacts needed to load a native deps shared library. These
 * are the library itself as well as any symlinks needed to resolve its
 * dynamic link dependencies.
 */
@Immutable @ThreadSafe
public class NativeDepsRunfiles {
  /** An object that represents no native deps. */
  public static final NativeDepsRunfiles EMPTY = new NativeDepsRunfiles(null, null);

  @Nullable
  private final Artifact library;

  @Nullable
  private final List<Artifact> runtimeSymlinks;

  public NativeDepsRunfiles(@Nullable Artifact library, @Nullable List<Artifact> runtimeSymlinks) {
    this.library = library;
    this.runtimeSymlinks = runtimeSymlinks;
  }

  /**
   * Returns the native deps library (which may itself be a symlink to another library), may be
   * null.
   */
  @Nullable
  public Artifact getLibrary() {
    return library;
  }

  /**
   * Returns the symlinks to the dynamic C++ runtime libraries needed by this library and findable
   * via this library's RPATH, may be null.
   */
  @Nullable
  public List<Artifact> getRuntimeSymlinks() {
    return runtimeSymlinks;
  }
}
