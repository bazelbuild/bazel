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
import com.google.devtools.build.lib.starlarkbuildapi.cpp.FeatureConfigurationApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

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

  public static FeatureConfigurationForStarlark from(FeatureConfiguration featureConfiguration) {
    return new FeatureConfigurationForStarlark(featureConfiguration);
  }

  private FeatureConfigurationForStarlark(FeatureConfiguration featureConfiguration) {
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
  }

  public FeatureConfiguration getFeatureConfiguration() {
    return featureConfiguration;
  }

  @Override
  public void str(Printer printer, StarlarkSemantics semantics) {
    printer.append("<FeatureConfiguration(");
    printer.append("ENABLED:");
    printer.append(Joiner.on(", ").join(featureConfiguration.getEnabledFeatureNames()));
    printer.append(";REQUESTED:");
    printer.append(Joiner.on(", ").join(featureConfiguration.getRequestedFeatures()));
    printer.append(")>");
  }

  @Override
  public void debugPrint(Printer printer, StarlarkSemantics semantics) {
    printer.append("<FeatureConfiguration(");
    printer.append(Joiner.on(", ").join(featureConfiguration.getEnabledFeatureNames()));
    printer.append(")>");
  }

  @StarlarkMethod(
      name = "is_enabled",
      parameters = {@Param(name = "feature")},
      documented = false,
      useStarlarkThread = true)
  // TODO(b/339328480): collect all feature names in a single location
  public boolean isEnabled(String feature, StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return featureConfiguration.isEnabled(feature);
  }
}
