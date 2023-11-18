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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.SymlinkDefinition;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Set;
import java.util.function.Function;

/** Rule definitions for Python rules. */
public class PyRuleClasses {

  public static final FileType PYTHON_SOURCE = FileType.of(".py", ".py3");

  /**
   * A value set of the target and sentinel values that doesn't mention the sentinel in error
   * messages.
   */
  public static final AllowedValueSet TARGET_PYTHON_ATTR_VALUE_SET =
      new AllowedValueSet(PythonVersion.TARGET_AND_SENTINEL_STRINGS) {
        @Override
        public String getErrorReason(Object value) {
          return String.format("has to be one of 'PY2' or 'PY3' instead of '%s'", value);
        }
      };

  /** The py3 symlinks. */
  public static final class Py3Symlink implements SymlinkDefinition {
    public static final Py3Symlink INSTANCE = new Py3Symlink();

    @Override
    public String getLinkName(String symlinkPrefix, String productName, String workspaceBaseName) {
      return symlinkPrefix + Ascii.toLowerCase(PythonVersion.PY3.toString());
    }

    @Override
    public ImmutableSet<Path> getLinkPaths(
        BuildRequestOptions buildRequestOptions,
        Set<BuildConfigurationValue> targetConfigs,
        Function<BuildOptions, BuildConfigurationValue> configGetter,
        RepositoryName repositoryName,
        Path outputPath,
        Path execRoot) {
      if (!buildRequestOptions.experimentalCreatePySymlinks) {
        return ImmutableSet.of();
      }

      return targetConfigs.stream()
          .map(
              config -> {
                BuildOptions options = config.getOptions();
                PythonOptions opts =
                    options.hasNoConfig() ? null : options.get(PythonOptions.class);
                if (opts == null || !opts.canTransitionPythonVersion(PythonVersion.PY3)) {
                  return config;
                } else {
                  BuildOptions newOptions = options.clone();
                  newOptions.get(PythonOptions.class).setPythonVersion(PythonVersion.PY3);
                  return configGetter.apply(newOptions);
                }
              })
          .map(config -> config.getOutputDirectory(repositoryName).getRoot().asPath())
          .distinct()
          .collect(toImmutableSet());
    }
  }
}
