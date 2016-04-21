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

package com.google.devtools.build.lib.analysis.constraints;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;

/**
 * A provider that advertises which environments the associated target is compatible with
 * (from the point of view of the constraint enforcement system).
 */
public interface SupportedEnvironmentsProvider extends TransitiveInfoProvider {

  /**
   * Returns the static environments this target is compatible with. Static environments
   * are those that are independent of build configuration (e.g. declared in {@code restricted_to} /
   * {@code compatible_with}). See {@link ConstraintSemantics} for details.
   */
  EnvironmentCollection getStaticEnvironments();

  /**
   * Returns the refined environments this rule is compatible with. Refined environments are
   * static environments with unsupported environments from {@code select}able deps removed (on the
   * principle that others paths in the select would have provided those environments, so this rule
   * is "refined" to match whichever deps got chosen).
   *
   * <p>>Refined environments require knowledge of the build configuration. See
   * {@link ConstraintSemantics} for details.
   */
  EnvironmentCollection getRefinedEnvironments();
}
