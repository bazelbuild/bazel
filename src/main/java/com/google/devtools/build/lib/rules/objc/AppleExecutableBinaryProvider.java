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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkClassObjectConstructor;

/**
 * Provider containing the executable binary output that was built using an apple_binary target with
 * the 'executable' type.
 */
public final class AppleExecutableBinaryProvider extends SkylarkClassObject
    implements TransitiveInfoProvider {

 /** Skylark constructor and identifier for AppleExecutableBinaryProvider. */
  public static final SkylarkClassObjectConstructor SKYLARK_CONSTRUCTOR =
      SkylarkClassObjectConstructor.createNative("AppleExecutableBinary");

  private final Artifact appleExecutableBinary;

  /**
   * Creates a new AppleExecutableBinaryProvider provider that propagates the given apple_binary
   * configured as an executable.
   */
  public AppleExecutableBinaryProvider(Artifact appleExecutableBinary) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of("binary", appleExecutableBinary));
    this.appleExecutableBinary = appleExecutableBinary;
  }

  /**
   * Returns the multi-architecture executable binary that apple_binary created.
   */
  public Artifact getAppleExecutableBinary() {
    return appleExecutableBinary;
  }
}
