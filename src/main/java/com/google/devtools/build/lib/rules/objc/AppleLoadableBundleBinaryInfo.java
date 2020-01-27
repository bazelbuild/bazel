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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleLoadableBundleBinaryApi;

/**
 * Provider containing the executable binary output that was built using an apple_binary target with
 * the 'loadable_bundle' type. This provider contains:
 *
 * <ul>
 *   <li>'binary': The binary artifact output by apple_binary
 *   <li>'objc': An {@link ObjcProvider} which contains information about the transitive
 *       dependencies linked into the binary.
 * </ul>
 */
@Immutable
public final class AppleLoadableBundleBinaryInfo extends NativeInfo
    implements AppleLoadableBundleBinaryApi {

  /** Skylark name for the AppleLoadableBundleBinary. */
  public static final String SKYLARK_NAME = "AppleLoadableBundleBinary";

  /** Skylark constructor and identifier for AppleLoadableBundleBinary. */
  public static final NativeProvider<AppleLoadableBundleBinaryInfo> SKYLARK_CONSTRUCTOR =
      new NativeProvider<AppleLoadableBundleBinaryInfo>(
          AppleLoadableBundleBinaryInfo.class, SKYLARK_NAME) {};

  private final Artifact appleLoadableBundleBinary;
  private final ObjcProvider depsObjcProvider;

  /**
   * Creates a new AppleLoadableBundleBinaryInfo provider that propagates the given apple_binary
   * configured as a loadable bundle binary.
   */
  public AppleLoadableBundleBinaryInfo(Artifact appleLoadableBundleBinary,
      ObjcProvider depsObjcProvider) {
    super(SKYLARK_CONSTRUCTOR);
    this.appleLoadableBundleBinary = appleLoadableBundleBinary;
    this.depsObjcProvider = depsObjcProvider;
  }

  /**
   * Returns the multi-architecture binary that apple_binary created.
   */
  @Override
  public Artifact getAppleLoadableBundleBinary() {
    return appleLoadableBundleBinary;
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
