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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * This interface represents a symbolic link to an absolute or relative path, stored in an
 * InMemoryFileSystem.
 */
@ThreadSafe
@Immutable
final class InMemoryLinkInfo extends InMemoryContentInfo {

  private final PathFragment linkContent;
  private final PathFragment normalizedLinkContent;

  InMemoryLinkInfo(Clock clock, PathFragment linkContent) {
    super(clock);
    this.linkContent = linkContent;
    this.normalizedLinkContent = linkContent;
  }

  @Override
  public boolean isDirectory() {
    return false;
  }

  @Override
  public boolean isSymbolicLink() {
    return true;
  }

  @Override
  public boolean isFile() {
    return false;
  }

  @Override
  public boolean isSpecialFile() {
    return false;
  }

  @Override
  public long getSize() {
    return linkContent.getSafePathString().length();
  }

  /**
   * Returns the content of the symbolic link.
   */
  PathFragment getLinkContent() {
    return linkContent;
  }

  /**
   * Returns the content of the symbolic link, with ".." and "." removed
   * (except for the possibility of necessary ".." segments at the beginning).
   */
  PathFragment getNormalizedLinkContent() {
    return normalizedLinkContent;
  }

  @Override
  public String toString() {
    return super.toString() + " -> " + linkContent;
  }
}
