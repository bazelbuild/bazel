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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;

/**
 * Provider containing the executable binary output that was built using an apple_binary target with
 * the 'loadable_bundle' type. This provider contains:
 * <ul>
 *   <li>'binary': The binary artifact output by apple_binary</li>
 * </ul> 
 */
public final class AppleLoadableBundleBinaryProvider extends SkylarkClassObject
    implements TransitiveInfoProvider {

  /** Skylark name for the AppleLoadableBundleBinary. */
  public static final String SKYLARK_NAME = "AppleLoadableBundleBinary";

 /** Skylark constructor and identifier for AppleLoadableBundleBinary. */
  public static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
     new NativeClassObjectConstructor(SKYLARK_NAME) { };

  private final Artifact appleLoadableBundleBinary;

  /**
   * Creates a new AppleLoadableBundleBinaryProvider provider that propagates the given apple_binary
   * configured as a loadable bundle binary.
   */
  public AppleLoadableBundleBinaryProvider(Artifact appleLoadableBundleBinary) {
    super(SKYLARK_CONSTRUCTOR,
        ImmutableMap.<String, Object>of("binary", appleLoadableBundleBinary));
    this.appleLoadableBundleBinary = appleLoadableBundleBinary;
  }

  /**
   * Returns the multi-architecture binary that apple_binary created.
   */
  public Artifact getAppleLoadableBundleBinary() {
    return appleLoadableBundleBinary;
  }
}
