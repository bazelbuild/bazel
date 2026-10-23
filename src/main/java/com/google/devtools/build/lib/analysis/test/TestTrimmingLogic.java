// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.test;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;

/**
 * Contains the pure logic for trimming test configuration from non-test targets that backs {@link
 * com.google.devtools.build.lib.analysis.test.TestTrimmingTransitionFactory}.
 */
public final class TestTrimmingLogic {

  // This cache is to prevent major slowdowns when using --trim_test_configuration. This
  // transition is always invoked on every target in the top-level invocation. Thus, a wide
  // invocation, like //..., will cause the transition to be invoked on a large number of targets
  // leading to significant performance degradation. (Notably, the transition itself is somewhat
  // fast; however, the post-processing of the BuildOptions into the actual BuildConfigurationValue
  // takes a significant amount of time).
  //
  // Test any caching changes for performance impact in a longwide scenario with
  // --trim_test_configuration on versus off.
  private static final Cache<String, BuildOptions> cache =
      Caffeine.newBuilder().weakValues().build();

  /** Returns a new {@link BuildOptions} instance with test configuration removed. */
  public static BuildOptions trim(BuildOptions options) {
    return cache.get(
        options.checksum(),
        k -> {
          // LINT.IfChange
          BuildOptions.Builder builder = options.toBuilder();
          builder.removeFragmentOptions(TestOptions.class);
          // Only the label of the --run_under target (if any) needs to be part of the configuration
          // for non-test targets, all other information is directly obtained from the options in
          // RunCommand.
          CoreOptions coreOptions = builder.getFragmentOptions(CoreOptions.class);
          coreOptions.setRunUnder(RunUnder.trimForNonTestConfiguration(coreOptions.getRunUnder()));
          return builder.build();
          // LINT.ThenChange(TestConfiguration.java)
        });
  }

  /** Returns a new {@link BuildOptions} instance with test configuration removed. */
  static BuildOptions trim(BuildOptionsView options) {
    return trim(options.underlying());
  }

  private TestTrimmingLogic() {}
}
