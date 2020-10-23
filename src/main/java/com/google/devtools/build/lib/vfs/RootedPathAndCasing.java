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
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.Serializable;
import java.util.Objects;

/**
 * A {@link RootedPath} and its String representation.
 *
 * <p>This object stores not just a path but also the particular casing of it. On case-sensitive
 * filesystems, the paths "foo/Bar.txt" and "FOO/bar.TXT" are different files, but on
 * case-insensitive (or case-ignoring) they refer to the same file.
 *
 * <p>The {@code RootedPathAndCasing} class therefore allows comparing equal {@link RootedPath}s
 * that might use different casing.
 */
@AutoCodec
public final class RootedPathAndCasing implements Serializable {
  private final RootedPath path;
  private final String casing;
  private final int hash;

  /** Constructs a {@link RootedPath} from a {@link Root} and path fragment relative to the root. */
  @AutoCodec.Instantiator
  @AutoCodec.VisibleForSerialization
  RootedPathAndCasing(RootedPath path, String casing) {
    this.path = Preconditions.checkNotNull(path);
    this.casing = Preconditions.checkNotNull(casing);
    this.hash = Objects.hash(path, casing);
  }

  public static RootedPathAndCasing create(RootedPath path) {
    return new RootedPathAndCasing(path, path.getRootRelativePath().getPathString());
  }

  public RootedPath getPath() {
    return path;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RootedPathAndCasing)) {
      return false;
    }
    RootedPathAndCasing other = (RootedPathAndCasing) obj;
    return hash == other.hash // fast track for unequal objects
        && Objects.equals(path, other.path)
        && Objects.equals(casing, other.casing);
  }

  @Override
  public int hashCode() {
    return hash;
  }

  @Override
  public String toString() {
    return "[" + path.getRoot() + "]/[" + casing + "]";
  }
}
