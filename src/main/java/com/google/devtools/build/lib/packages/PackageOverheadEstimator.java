// Copyright 2021 The Bazel Authors. All rights reserved.
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

import java.util.OptionalLong;

/**
 * Estimates "package overhead", which is a rough approximation of the memory and general accounting
 * costs associated with a loaded package.
 *
 * <p>Estimates are not intended to be perfect but should be reproducible. Some things may be
 * over-accounted, some things under, with the expectation that it all comes out roughly even in the
 * end.
 */
public interface PackageOverheadEstimator {

  PackageOverheadEstimator NOOP_ESTIMATOR = pkg -> OptionalLong.empty();

  /** Returns the estimated package overhead, or empty if not calculated. */
  OptionalLong estimatePackageOverhead(Package pkg);
}
