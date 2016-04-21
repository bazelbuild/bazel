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

import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * A class that exposes apple rule implementation internals to skylark.
 */
@SkylarkModule(
  name = "apple_common",
  doc = "Functions for skylark to access internals of the apple rule implementations."
)
public class AppleSkylarkCommon {
  
  @SkylarkCallable(
      name = "apple_toolchain",
      doc = "Utilities for resolving items from the apple toolchain."
  )
  public static AppleToolchain getAppleToolchain() {
    return new AppleToolchain();
  }
  
  @SkylarkCallable(
      name = "keys",
      doc = "Retrieves ObjcProvider keys",
      structField = true
  )
  public static SkylarkKeyStore getKeys() {
    return new SkylarkKeyStore();
  } 
}

