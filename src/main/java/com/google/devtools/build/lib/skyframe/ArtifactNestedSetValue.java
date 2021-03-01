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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Represent a "promise" that the Artifacts under a NestedSet is evaluated by Skyframe and the
 * ValueOrException is available in {@link ArtifactNestedSetFunction#artifactToSkyValueMap}.
 */
@Immutable
@ThreadSafe
public final class ArtifactNestedSetValue implements SkyValue {

  @Override
  public boolean dataIsShareable() {
    // This is just a promise that data is available in memory. Not meant for cross-server sharing.
    return false;
  }
}
