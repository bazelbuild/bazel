// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleExecutableBinaryApi;

/**
 * Provider containing the executable binary output that was built using an apple_binary target with
 * the 'executable' type. This provider contains:
 *
 * <ul>
 *   <li>'binary': The executable binary artifact output by apple_binary
 *   <li>'objc': An {@link ObjcProvider} which contains information about the transitive
 *       dependencies linked into the binary, (intended so that bundle loaders depending on this
 *       executable may avoid relinking symbols included in the loadable binary
 * </ul>
 */
@Immutable
public final class AppleExecutableBinaryInfo extends NativeInfo
    implements AppleExecutableBinaryApi {

  /** Skylark name for the AppleExecutableBinaryInfo. */
  public static final String SKYLARK_NAME = "AppleExecutableBinary";

  /** Skylark constructor and identifier for AppleExecutableBinaryInfo. */
  public static final NativeProvider<AppleExecutableBinaryInfo> SKYLARK_CONSTRUCTOR =
      new NativeProvider<AppleExecutableBinaryInfo>(
          AppleExecutableBinaryInfo.class, SKYLARK_NAME) {};

  private final Artifact appleExecutableBinary;
  private final ObjcProvider depsObjcProvider;

  /**
   * Creates a new AppleExecutableBinaryInfo provider that propagates the given apple_binary
   * configured as an executable.
   */
  public AppleExecutableBinaryInfo(Artifact appleExecutableBinary,
      ObjcProvider depsObjcProvider) {
    super(SKYLARK_CONSTRUCTOR);
    this.appleExecutableBinary = appleExecutableBinary;
    this.depsObjcProvider = depsObjcProvider;
  }

  /**
   * Returns the multi-architecture executable binary that apple_binary created.
   */
  @Override
  public Artifact getAppleExecutableBinary() {
    return appleExecutableBinary;
  }

  /**
   * Returns the {@link ObjcProvider} which contains information about the transitive dependencies
   * linked into the dylib.
   */
  @Override
  public ObjcProvider getDepsObjcProvider() {
    return depsObjcProvider;
  }
}
