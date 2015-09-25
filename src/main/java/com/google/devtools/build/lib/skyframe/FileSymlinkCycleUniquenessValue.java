// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A value for ensuring that a file symlink cycle is reported exactly once. This is achieved by
 * forcing the same value key for two logically equivalent cycles (e.g. ['a' -> 'b' -> 'c' -> 'a']
 * and ['b' -> 'c' -> 'a' -> 'b']), and letting Skyframe do its magic.
 */
class FileSymlinkCycleUniquenessValue extends AbstractChainUniquenessValue {
  static final FileSymlinkCycleUniquenessValue INSTANCE = new FileSymlinkCycleUniquenessValue();

  private FileSymlinkCycleUniquenessValue() {
  }

  static SkyKey key(ImmutableList<RootedPath> cycle) {
    return AbstractChainUniquenessValue.key(SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, cycle);
  }
}
