// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;

/**
 * Provider containing information regarding multi-architecture Apple static libraries, as is
 * propagated that by the {@code apple_static_library} rule.
 *
 * <p>This provider contains:
 *
 * <ul>
 *   <li>'archive': The multi-arch archive (.a) output by apple_static_library
 *   <li>'objc': An {@link ObjcProvider} which contains information about the transitive
 *       dependencies linked into the library, (intended so that targets may avoid linking symbols
 *       included in this archive multiple times).
 * </ul>
 */
public final class AppleStaticLibraryProvider extends NativeInfo {

  /** Skylark name for the AppleStaticLibraryProvider. */
  public static final String SKYLARK_NAME = "AppleStaticLibrary";

  /** Skylark constructor and identifier for AppleStaticLibraryProvider. */
  public static final NativeProvider<AppleStaticLibraryProvider> SKYLARK_CONSTRUCTOR =
      new NativeProvider<AppleStaticLibraryProvider>(
          AppleStaticLibraryProvider.class, SKYLARK_NAME) {};

  private final Artifact multiArchArchive;
  private final ObjcProvider depsObjcProvider;

  /**
   * Creates a new AppleStaticLibraryProvider provider that propagates the given
   * {@code apple_static_library} information.
   */
  public AppleStaticLibraryProvider(Artifact multiArchArchive,
      ObjcProvider depsObjcProvider) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of(
        "archive", multiArchArchive,
        "objc", depsObjcProvider));
    this.multiArchArchive = multiArchArchive;
    this.depsObjcProvider = depsObjcProvider;
  }

  /**
   * Returns the multi-architecture archive that {@code apple_static_library} created.
   */
  public Artifact getMultiArchArchive() {
    return multiArchArchive;
  }

  /**
   * Returns the {@link ObjcProvider} which contains information about the transitive dependencies
   * linked into the archive.
   */
  public ObjcProvider getDepsObjcProvider() {
    return depsObjcProvider;
  }
}
