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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.FeatureConfigurationApi;
import com.google.devtools.build.lib.syntax.Printer;

/**
 * Wrapper for {@link FeatureConfiguration}, {@link CppConfiguration}, and {@link BuildOptions}.
 *
 * <p>Instances are created in Starlark by cc_common.configure_features(ctx, cc_toolchain), and
 * passed around pretending to be {@link FeatureConfiguration}. Then when the need arises, we get
 * the {@link CppConfiguration} and {@link BuildOptions} from it and use it in times when
 * configuration of cc_toolchain is different than configuration of the rule depending on it.
 */
// TODO(b/129045294): Remove once cc_toolchain has target configuration.
public class FeatureConfigurationForStarlark implements FeatureConfigurationApi {

  private final FeatureConfiguration featureConfiguration;
  private final CppConfiguration cppConfiguration;
  private final BuildOptions buildOptions;

  public static FeatureConfigurationForStarlark from(
      FeatureConfiguration featureConfiguration,
      CppConfiguration cppConfiguration,
      BuildOptions buildOptions) {
    return new FeatureConfigurationForStarlark(
        featureConfiguration, cppConfiguration, buildOptions);
  }

  private FeatureConfigurationForStarlark(
      FeatureConfiguration featureConfiguration,
      CppConfiguration cppConfiguration,
      BuildOptions buildOptions) {
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
    this.cppConfiguration = Preconditions.checkNotNull(cppConfiguration);
    this.buildOptions = buildOptions;
  }

  public FeatureConfiguration getFeatureConfiguration() {
    return featureConfiguration;
  }

  @Override
  public void debugPrint(Printer printer) {
    printer.append("<FeatureConfiguration(");
    printer.append(Joiner.on(", ").join(featureConfiguration.getEnabledFeatureNames()));
    printer.append(")>");
  }

  /**
   * Get {@link CppConfiguration} that is threaded along with {@link FeatureConfiguration}. Do this
   * only when you're completely aware of why this method was added and hlopko@ allowed you to.
   *
   * @deprecated will be removed soon by b/129045294.
   */
  @Deprecated
  CppConfiguration
      getCppConfigurationFromFeatureConfigurationCreatedForStarlark_andIKnowWhatImDoing() {
    return cppConfiguration;
  }

  /**
   * Get {@link BuildOptions} that is threaded along with {@link FeatureConfiguration}. Do this only
   * when you're completely aware of why this method was added and hlopko@ allowed you to.
   *
   * @deprecated will be removed soon by b/129045294.
   */
  @Deprecated
  BuildOptions getBuildOptionsFromFeatureConfigurationCreatedForStarlark_andIKnowWhatImDoing() {
    return buildOptions;
  }
}
