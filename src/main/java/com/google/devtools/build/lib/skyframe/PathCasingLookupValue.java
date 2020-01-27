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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.RootedPathAndCasing;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Value that represents whether a certain path is correctly cased.
 *
 * <p>Most filesystems preserve uppercase and lowercase letters when creating entries: {@code
 * mkdir("Abc1")} creates the directory "Abc1", and not "ABC1" nor "abc1".
 *
 * <p>Some filesystems differentiate casing when looking up entries, but some don't. Suppose we have
 * an empty directory and create the file "Abc1" underneath it. Then, calling {@code exists("Abc1")}
 * succeeds and {@code exists("ABC1")} fails on ext4 (Linux) but both calls succeed on APFS (macOS)
 * and NTFS (Windows). Reason for this difference is that ext4 is case-sensitive, so "Abc1" and
 * "ABC1" mean different files, but APFS and NTFS are case-ignoring (or case-insensitive), so both
 * paths mean the same file.
 *
 * <p>This object represents whether an existing path on a case-ignoring filesystem (APFS and NTFS)
 * is correctly or incorrectly cased, i.e. whether the exact use of upper and lower case letters
 * matches the entry on disk. In the previous example, "Abc1" is correctly cased while "ABC1" is
 * incorrectly cased.
 *
 * <p>Paths on case-sensitive filesystems (ext4) are always correctly cased, because the filesystem
 * requires exact case matching when accessing files.
 */
public abstract class PathCasingLookupValue implements SkyValue {

  @AutoCodec public static final BadPathCasing BAD = new BadPathCasing();

  @AutoCodec public static final CorrectPathCasing GOOD = new CorrectPathCasing();

  /** Singleton {@link PathCasingLookupValue} instance for incorrectly cased paths. */
  public static class BadPathCasing extends PathCasingLookupValue {
    @Override
    public boolean isCorrect() {
      return false;
    }
  }

  /** Singleton {@link PathCasingLookupValue} instance for correctly cased paths. */
  public static class CorrectPathCasing extends PathCasingLookupValue {
    @Override
    public boolean isCorrect() {
      return true;
    }
  }

  /**
   * Creates a {@code SkyKey} to request this {@code PathCasingLookupValue} from Skyframe.
   *
   * <p>The argument is a {@link RootedPath} and not a {@link Path} for two reasons:
   *
   * <ul>
   *   <li>as of 2019-12-04 the {@code PathCasingLookupFunction} depends on {@code
   *       DirectoryListingValue} whose {@code SkyKey} requires a {@code RootedPath}
   *   <li>as of 2019-12-04 the {@code PathCasingLookupValue} is only used to validate that a
   *       package label is correctly cased, and package labels are always relative to a package
   *       root, so using a {@code RootedPath} is adequate and the Root part of it doesn't even have
   *       to be correctly cased.
   * </ul>
   */
  public static SkyKey key(RootedPath path) {
    return Key.create(RootedPathAndCasing.create(path));
  }

  private PathCasingLookupValue() {}

  public abstract boolean isCorrect();

  /** {@link SkyKey} for {@link PathCasingLookupValue} computation. */
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  public static class Key extends AbstractSkyKey<RootedPathAndCasing> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(RootedPathAndCasing arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(RootedPathAndCasing arg) {
      Preconditions.checkNotNull(arg);
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PATH_CASING_LOOKUP;
    }
  }
}
