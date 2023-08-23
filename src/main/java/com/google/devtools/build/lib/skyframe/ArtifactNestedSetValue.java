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
import com.google.devtools.build.skyframe.NotComparableSkyValue;

/**
 * Represent a "promise" that the Artifacts under a NestedSet are evaluated by Skyframe and are
 * available from {@link ArtifactNestedSetFunction#getValueForKey}.
 *
 * <p>Implements {@link NotComparableSkyValue} to prohibit value-based change pruning.
 */
@Immutable
@ThreadSafe
final class ArtifactNestedSetValue implements NotComparableSkyValue {

  static final ArtifactNestedSetValue INSTANCE = new ArtifactNestedSetValue();

  private ArtifactNestedSetValue() {}
}
