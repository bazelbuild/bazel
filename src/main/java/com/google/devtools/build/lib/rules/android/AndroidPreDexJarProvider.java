// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/** A provider of the final Jar to be dexed for targets that build APKs. */
@AutoValue
@Immutable
public abstract class AndroidPreDexJarProvider implements TransitiveInfoProvider {

  public static AndroidPreDexJarProvider create(Artifact preDexJar) {
    return new AutoValue_AndroidPreDexJarProvider(preDexJar);
  }

  /** Returns the jar to be dexed. */
  public abstract Artifact getPreDexJar();

  AndroidPreDexJarProvider() {}
}
