// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigGlobalLibraryApi;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigurationTransitionApi;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Implementation of {@link ConfigGlobalLibraryApi}.
 *
 * <p>A collection of top-level Starlark functions pertaining to configuration.
 */
public class ConfigGlobalLibrary implements ConfigGlobalLibraryApi {

  private static final String COMMAND_LINE_OPTION_PREFIX = "//command_line_option:";

  @Override
  public ConfigurationTransitionApi transition(
      BaseFunction implementation,
      List<String> inputs,
      List<String> outputs,
      Location location,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    SkylarkSemantics semantics = env.getSemantics();
    if (!semantics.experimentalStarlarkConfigTransitions()) {
      throw new EvalException(
          location,
          "transition() is experimental and disabled by default. "
              + "This API is in development and subject to change at any time. Use "
              + "--experimental_starlark_config_transitions to use this experimental API.");
    }
    validateBuildSettingKeys(inputs, "input", location);
    validateBuildSettingKeys(outputs, "output", location);
    return StarlarkDefinedConfigTransition.newRegularTransition(
        implementation, inputs, outputs, semantics, context);
  }

  @Override
  public ConfigurationTransitionApi analysisTestTransition(
      SkylarkDict<String, String> changedSettings, Location location, SkylarkSemantics semantics)
      throws EvalException {
    Map<String, Object> changedSettingsMap =
        changedSettings.getContents(String.class, Object.class, "changed_settings dict");
    validateBuildSettingKeys(changedSettingsMap.keySet(), "output", location);
    return StarlarkDefinedConfigTransition.newAnalysisTestTransition(changedSettingsMap, location);
  }

  private void validateBuildSettingKeys(
      Iterable<String> optionKeys, String keyErrorDescriptor, Location location)
      throws EvalException {

    HashSet<String> processedOptions = Sets.newHashSet();

    for (String optionKey : optionKeys) {
      if (!optionKey.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
        try {
          Label.parseAbsoluteUnchecked(optionKey);
        } catch (IllegalArgumentException e) {
          throw new EvalException(
              location,
              String.format(
                  "invalid transition %s '%s'. If this is intended as a native option, "
                      + "it must begin with //command_line_option:",
                  keyErrorDescriptor, optionKey),
              e);
        }
      }
      if (!processedOptions.add(optionKey)) {
        throw new EvalException(location,
            String.format("duplicate transition %s '%s'", keyErrorDescriptor, optionKey));
      }
    }
  }
}
