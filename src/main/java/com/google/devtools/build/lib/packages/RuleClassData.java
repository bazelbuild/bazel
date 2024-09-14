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
package com.google.devtools.build.lib.packages;

/**
 * Subset of {@link RuleClass} fields needed by dependents of {@link Rule} instances.
 *
 * <p>This is used when serializing analysis phase values that reference this data. In the native
 * case, this can be backed by the real, local {@link RuleClass} instance by {@link
 * RuleClassProvider} lookup.
 *
 * <p>In the case of Starlark, a simple object containing these values is used instead.
 */
// TODO(b/297857068): remove this when it is possible to deserialize this by referencing a .bzl
// loaded rule instead.
interface RuleClassData {
  /** Returns the class of rule that this RuleClass represents (e.g. "cc_library"). */
  String getName();

  /** Returns the target kind of this class of rule (e.g. "cc_library rule"). */
  String getTargetKind();

  /** Returns whether rules of this class can be made available during dependency resolution. */
  boolean isDependencyResolutionRule();

  /** Returns the set of advertised transitive info providers. */
  AdvertisedProviderSet getAdvertisedProviders();

  /** Returns true if corresponding {@link RuleClass} is Starlark-defined. */
  boolean isStarlark();
}
