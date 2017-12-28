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

package com.google.devtools.build.lib.analysis.config.transitions;

/**
 * Declaration how the configuration should change when following a label or label list attribute.
 *
 * <p>Do not add to this. This is a legacy interface from when Blaze had limited support for
 * transitions. Use {@link PatchTransition} or {@link SplitTransition} instead.
 */
@Deprecated
public enum ConfigurationTransitionProxy implements Transition {
  /** No transition, i.e., the same configuration as the current. */
  NONE,

  /** Transition to a null configuration (applies to, e.g., input files). */
  NULL,

  /** Transition from the target configuration to the data configuration. */
  // TODO(bazel-team): Move this elsewhere.
  DATA,
}
