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

import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigGlobalLibraryApi;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigurationTransitionApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import java.util.List;

/**
 * Implementation of {@link ConfigGlobalLibraryApi}.
 *
 * <p>A collection of top-level Starlark functions pertaining to configuration.
 */
public class ConfigGlobalLibrary implements ConfigGlobalLibraryApi {

  @Override
  public ConfigurationTransitionApi transition(BaseFunction implementation, List<String> inputs,
      List<String> outputs, Boolean forAnalysisTesting,
      Location location, SkylarkSemantics semantics) throws EvalException {
    if (forAnalysisTesting) {
      if (!semantics.experimentalAnalysisTestingImprovements()) {
        throw new EvalException(
            location,
            "transition(for_analysis_testing=True) is experimental "
                + "and disabled by default. This API is in development and subject to change at "
                + "any time. Use --experimental_analysis_testing_improvements to use this "
                + "experimental API.");
      }
    } else {
      if (!semantics.experimentalStarlarkConfigTransitions()) {
        throw new EvalException(
            location,
            "transition() is experimental and disabled by default. "
                + "This API is in development and subject to change at any time. Use "
                + "--experimental_starlark_config_transitions to use this experimental API.");
      }
    }
    return new StarlarkDefinedConfigTransition(implementation, forAnalysisTesting);
  }
}
