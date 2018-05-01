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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * Provider containing information regarding multi-architecture Apple static libraries, as is
 * propagated by the {@code apple_static_library} rule.
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
@SkylarkModule(
    name = "AppleStaticLibrary",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "A provider containing information regarding multi-architecture Apple static libraries, "
        + "as is propagated by the apple_static_library rule."
)
public final class AppleStaticLibraryInfo extends NativeInfo {

  /** Skylark name for the AppleStaticLibraryInfo. */
  public static final String SKYLARK_NAME = "AppleStaticLibrary";

  /** Skylark constructor and identifier for AppleStaticLibraryInfo. */
  public static final Provider SKYLARK_CONSTRUCTOR = new Provider();

  private final Artifact multiArchArchive;
  private final ObjcProvider depsObjcProvider;

  /**
   * Creates a new AppleStaticLibraryInfo provider that propagates the given
   * {@code apple_static_library} information.
   */
  public AppleStaticLibraryInfo(Artifact multiArchArchive,
      ObjcProvider depsObjcProvider) {
    super(SKYLARK_CONSTRUCTOR);
    this.multiArchArchive = multiArchArchive;
    this.depsObjcProvider = depsObjcProvider;
  }

  /**
   * Returns the multi-architecture archive that {@code apple_static_library} created.
   */
  @SkylarkCallable(name = "archive",
      structField = true,
      doc = "The multi-arch archive (.a) output by apple_static_library."
  )
  public Artifact getMultiArchArchive() {
    return multiArchArchive;
  }

  /**
   * Returns the {@link ObjcProvider} which contains information about the transitive dependencies
   * linked into the archive.
   */
  @SkylarkCallable(name = "objc",
      structField = true,
      doc = "A provider which contains information about the transitive dependencies linked into "
          + "the archive."
  )
  public ObjcProvider getDepsObjcProvider() {
    return depsObjcProvider;
  }

   /**
    * Provider class for {@link AppleStaticLibraryInfo} objects.
    */
  public static class Provider extends BuiltinProvider<AppleStaticLibraryInfo> {
    private Provider() {
      super(SKYLARK_NAME, AppleStaticLibraryInfo.class);
    }

    @SkylarkCallable(
        name = SKYLARK_NAME,
        doc = "The <code>AppleStaticLibrary</code> constructor.",
        parameters = {
          @Param(
              name = "archive",
              type = Artifact.class,
              named = true,
              positional = false,
              doc = "Multi-architecture archive (.a) representing a static library"),
          @Param(
              name = "objc",
              type = ObjcProvider.class,
              named = true,
              positional = false,
              doc = "A provider which contains information about the transitive dependencies "
                  + "linked into the archive."),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = AppleStaticLibraryInfo.class)
    public AppleStaticLibraryInfo appleStaticLibrary(
        Artifact archive, ObjcProvider objcProvider) throws EvalException {
      return new AppleStaticLibraryInfo(archive, objcProvider);
    }
  }
}
