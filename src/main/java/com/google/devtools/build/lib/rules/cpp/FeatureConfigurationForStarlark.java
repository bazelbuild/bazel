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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.FeatureConfigurationApi;

/**
 * Wrapper class for {@link FeatureConfiguration} and {@link CppConfiguration}.
 *
 * <p>Instances are created in Starlark by cc_common.configure_features(ctx, cc_toolchain), and
 * passed around pretending to be {@link FeatureConfiguration}. Then when the need arises, we get
 * the {@link CppConfiguration} from it and use it in times when configuration of cc_toolchain is
 * different than configuration of the rule depending on it.
 */
// TODO(b/129045294): Remove once cc_toolchain has target configuration.
public class FeatureConfigurationForStarlark implements FeatureConfigurationApi {

  private final FeatureConfiguration featureConfiguration;
  private final CppConfiguration cppConfiguration;

  public static FeatureConfigurationForStarlark from(
      FeatureConfiguration featureConfiguration, CppConfiguration cppConfiguration) {
    return new FeatureConfigurationForStarlark(featureConfiguration, cppConfiguration);
  }

  private FeatureConfigurationForStarlark(
      FeatureConfiguration featureConfiguration, CppConfiguration cppConfiguration) {
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
    this.cppConfiguration = Preconditions.checkNotNull(cppConfiguration);
  }

  public FeatureConfiguration getFeatureConfiguration() {
    return featureConfiguration;
  }

  /**
   * Get {@link CppConfiguration} that is threaded along with {@link FeatureConfiguration}. Do this
   * only when you're completely aware of why this method was added and hlopko@ allowed you to.
   */
  CppConfiguration
      getCppConfigurationFromFeatureConfigurationCreatedForStarlark_andIKnowWhatImDoing() {
    return cppConfiguration;
  }
}
