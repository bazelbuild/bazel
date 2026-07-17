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
//
package com.google.devtools.build.lib.vfs;

import com.google.devtools.build.lib.skyframe.serialization.EnumCodec;

/**
 * Indicates the file type at the other end of a symlink.
 *
 * <p>Required by some filesystems (notably on Windows) to correctly create a symlink when its
 * target does not yet exist, as a different kind of filesystem object might be required depending
 * on the target type.
 */
public enum SymlinkTargetType {
  /** The target is of unspecified type. */
  UNSPECIFIED,
  /** The target is a regular file. */
  FILE,
  /** The target is a directory. */
  DIRECTORY;

  /**
   * Codec for {@link SymlinkTargetType}.
   *
   * <p>{@link com.google.devtools.build.lib.skyframe.serialization.AutoRegistry} excludes the
   * entire com.google.devtools.build.lib.vfs java package from having DynamicCodec support.
   * Therefore, we need to provide our own codec.
   */
  @SuppressWarnings("unused") // found by CLASSPATH-scanning magic
  private static final class Codec extends EnumCodec<SymlinkTargetType> {
    Codec() {
      super(SymlinkTargetType.class);
    }
  }
}
