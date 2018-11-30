// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.TriState;

/** Python-related command-line options. */
public class PythonOptions extends FragmentOptions {
  /**
   * Converter for the --force_python option.
   */
  public static class PythonVersionConverter extends EnumConverter<PythonVersion> {
    public PythonVersionConverter() {
      super(PythonVersion.class, "Python version");
    }
  }

  @Option(
    name = "build_python_zip",
    defaultValue = "auto",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Build python executable zip; on on Windows, off on other platforms"
  )
  public TriState buildPythonZip;

  @Option(
    name = "force_python",
    defaultValue = "null",
    converter = PythonVersionConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Overrides default_python_version attribute. Can be \"PY2\" or \"PY3\"."
  )
  public PythonVersion forcePython;

  @Option(
    name = "host_force_python",
    defaultValue = "null",
    converter = PythonVersionConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Overrides default_python_version attribute for the host configuration."
            + " Can be \"PY2\" or \"PY3\"."
  )
  public PythonVersion hostForcePython;

  public PythonVersion getPythonVersion() {
    return getPythonVersion(PythonVersion.getDefaultTargetValue());
  }

  public PythonVersion getPythonVersion(PythonVersion defaultVersion) {
    return (forcePython == null) ? defaultVersion : forcePython;
  }

  @Override
  public FragmentOptions getHost() {
    PythonOptions hostPythonOpts = (PythonOptions) getDefault();
    if (hostForcePython != null) {
      hostPythonOpts.forcePython = hostForcePython;
    } else {
      hostPythonOpts.forcePython = PythonVersion.PY2;
    }
    hostPythonOpts.buildPythonZip = buildPythonZip;
    return hostPythonOpts;
  }

  @Option(
    name = "experimental_build_transitive_python_runfiles",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Build the runfiles trees of py_binary targets that appear in the transitive "
            + "data runfiles of another binary."
  )
  public boolean buildTransitiveRunfilesTrees;
}

