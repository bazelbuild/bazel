// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Throwables;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment.MissingDepException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.skyframe.SkyFunction;

/**
 * Supplies the repository mapping of the main repository to the Starlark threads that evaluate
 * BUILD files, .bzl files and symbolic macros, for use by {@link
 * com.google.devtools.build.lib.cmdline.Label#debugPrint}.
 *
 * <p>The mapping is looked up lazily so that a Skyframe dependency on it is only recorded for
 * packages and .bzl files that actually print a label. Since the main repository mapping is
 * computed at the start of every command, the lookup normally succeeds immediately. If it doesn't,
 * the Starlark evaluation is aborted with a {@link MissingDepException} and the calling {@link
 * SkyFunction} has to return null to be restarted.
 */
final class LazyMainRepoMapping {
  private LazyMainRepoMapping() {}

  /** Returns a supplier that looks up the main repository mapping via the given environment. */
  static InterruptibleSupplier<RepositoryMapping> supplier(SkyFunction.Environment env) {
    return () -> {
      var value =
          (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
      if (value == null) {
        throw new MissingDepException("Restart due to missing main repository mapping");
      }
      return value.repositoryMapping();
    };
  }

  /**
   * Returns whether the given exception thrown out of a Starlark evaluation signals that the main
   * repository mapping is missing, in which case the calling {@link SkyFunction} has to return
   * null.
   */
  static boolean isMissingDep(RuntimeException e) {
    return Throwables.getCausalChain(e).stream().anyMatch(MissingDepException.class::isInstance);
  }
}
