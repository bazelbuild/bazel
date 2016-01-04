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

/**
 * Python-related command-line options.
 */
public class PythonOptions extends FragmentOptions {
  static final PythonVersion DEFAULT_PYTHON_VERSION = PythonVersion.PY2;

  /**
   * Converter for the --force_python option.
   */
  public static class PythonVersionConverter extends EnumConverter<PythonVersion> {
    public PythonVersionConverter() {
      super(PythonVersion.class, "Python version");
    }
  }

  @Option(name = "force_python",
      defaultValue = "null",
      category = "version",
      converter = PythonVersionConverter.class,
      help = "Overrides default_python_version attribute. Can be \"PY2\" or \"PY3\".")
  public PythonVersion forcePython;

  @Option(
    name = "host_force_python",
    defaultValue = "null",
    category = "version",
    converter = PythonVersionConverter.class,
    help =
        "Overrides default_python_version attribute for the host configuration."
            + " Can be \"PY2\" or \"PY3\"."
  )
  public PythonVersion hostForcePython;

  public PythonVersion getPythonVersion() {
    return (forcePython == null) ? DEFAULT_PYTHON_VERSION : forcePython;
  }

  @Override
  public FragmentOptions getHost(boolean fallback) {
    PythonOptions hostPythonOpts = (PythonOptions) getDefault();
    if (hostForcePython != null) {
      hostPythonOpts.forcePython = hostForcePython;
    } else {
      hostPythonOpts.forcePython = PythonVersion.PY2;
    }
    return hostPythonOpts;
  }
}

