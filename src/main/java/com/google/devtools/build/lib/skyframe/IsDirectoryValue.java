// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static java.util.Objects.requireNonNull;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Whether a path is a directory, following symlinks.
 *
 * <p>Depending on this rather than the path's {@code FileValue} lets Skyframe stop invalidation
 * when the path's contents change but its type does not.
 */
public enum IsDirectoryValue implements SkyValue {
  DIRECTORY,
  NOT_DIRECTORY;

  public static IsDirectoryValue of(boolean isDirectory) {
    return isDirectory ? DIRECTORY : NOT_DIRECTORY;
  }

  public boolean isDirectory() {
    return this == DIRECTORY;
  }

  public static Key key(RootedPath path) {
    return Key.create(path);
  }

  /** Key for {@link IsDirectoryValue}. */
  @AutoCodec
  public record Key(RootedPath path) implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    public Key {
      requireNonNull(path, "path");
    }

    @AutoCodec.Instantiator
    static Key create(RootedPath path) {
      return interner.intern(new Key(path));
    }

    @Override
    public RootedPath argument() {
      return path;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.IS_DIRECTORY;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
