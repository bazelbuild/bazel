// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;

/**
 * An ephemeral check that tells whether an artifact is consumed in a build.
 *
 * <p>"Ephemeral" because it guarantees the best estimated status of an artifact before the
 * execution of its generating action, and its behavior is undefined afterwards.
 */
public interface EphemeralCheckIfOutputConsumed {
  boolean test(Artifact artifact);
}
