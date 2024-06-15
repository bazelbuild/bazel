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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigStarlarkCommonApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkToolchainTypeRequirement;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Starlark namespace used to interact with Blaze's configurability APIs. */
public class ConfigStarlarkCommon implements ConfigStarlarkCommonApi {

  @Override
  public Provider getConfigFeatureFlagProviderConstructor() {
    return ConfigFeatureFlagProvider.STARLARK_CONSTRUCTOR;
  }

  @Override
  public ConfigurationTransitionApi createConfigFeatureFlagTransitionFactory(String attribute) {
    return new ConfigFeatureFlagTransitionFactory(attribute);
  }

  @Override
  public StarlarkToolchainTypeRequirement toolchainType(
      Object name, boolean mandatory, StarlarkThread thread) throws EvalException {

    Label label;
    if (name instanceof Label nameLabel) {
      label = nameLabel;
    } else if (name instanceof String) {
      LabelConverter converter = LabelConverter.forBzlEvaluatingThread(thread);
      try {
        label = converter.convert((String) name);
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf(
            "Unable to parse toolchain_type label '%s': %s", name, e.getMessage());
      }
    } else {
      throw Starlark.errorf(
          "config_common.toolchain_type() takes a Label or String, and instead got a %s",
          name.getClass().getSimpleName());
    }

    return ToolchainTypeRequirement.builder(label).mandatory(mandatory).build();
  }
}
