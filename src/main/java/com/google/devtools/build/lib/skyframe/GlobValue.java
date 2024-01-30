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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value corresponding to a glob. It has two subclasses, {@link GlobValueWithNestedSet} and {@link
 * GlobValueWithImmutableSet}.
 */
public abstract class GlobValue implements SkyValue {

  /** Returns all glob matching {@link PathFragment}s in {@link ImmutableSet}. */
  public abstract ImmutableSet<PathFragment> getMatches();

  /**
   * Constructs a {@link GlobDescriptor} for a glob lookup. {@code packageName} is assumed to be an
   * existing package. Trying to glob into a non-package is undefined behavior.
   *
   * @throws InvalidGlobPatternException if the pattern is not valid.
   */
  @ThreadSafe
  public static GlobDescriptor key(
      PackageIdentifier packageId,
      Root packageRoot,
      String pattern,
      Globber.Operation globOperation,
      PathFragment subdir)
      throws InvalidGlobPatternException {
    if (pattern.indexOf('?') != -1) {
      throw new InvalidGlobPatternException(pattern, "wildcard ? forbidden");
    }

    String error = UnixGlob.checkPatternForError(pattern);
    if (error != null) {
      throw new InvalidGlobPatternException(pattern, error);
    }

    return internalKey(packageId, packageRoot, subdir, pattern, globOperation);
  }

  /**
   * Constructs a {@link GlobDescriptor} for a glob lookup.
   *
   * <p>Do not use outside {@code GlobFunction}.
   */
  @ThreadSafe
  static GlobDescriptor internalKey(
      PackageIdentifier packageId,
      Root packageRoot,
      PathFragment subdir,
      String pattern,
      Globber.Operation globOperation) {
    return GlobDescriptor.create(packageId, packageRoot, subdir, pattern, globOperation);
  }
}
