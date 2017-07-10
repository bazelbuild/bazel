// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.python;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import com.google.devtools.common.options.proto.OptionFilters.OptionEffectTag;

/**
 * Bazel-specific Python configuration.
 */
@Immutable
public class BazelPythonConfiguration extends BuildConfiguration.Fragment {

  /**
  * A path converter for python3 path
  */
  public static class Python3PathConverter implements Converter<String> {
    @Override
    public String convert(String input) {
      if (input.equals("auto")) {
        return OS.getCurrent() == OS.WINDOWS ? "python" : "python3";
      }
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "An option for python3 path";
    }
  }

  /**
   * Bazel-specific Python configuration options.
   */
  public static final class Options extends FragmentOptions {
    @Option(
      name = "python2_path",
      defaultValue = "python",
      category = "version",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Local path to the Python2 executable. "
                + "Deprecated, please use python_path or python_top instead."
    )
    public String python2Path;

    @Option(
      name = "python3_path",
      converter = Python3PathConverter.class,
      defaultValue = "auto",
      category = "version",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Local path to the Python3 executable. "
                + "Deprecated, please use python_path or python_top instead."
    )
    public String python3Path;

    @Option(
        name = "python_top",
        converter = LabelConverter.class,
        defaultValue = "null",
        category = "version",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "The label of py_runtime rule used for the Python interpreter invoked by Bazel."
    )
    public Label pythonTop;

    @Option(
        name = "python_path",
        defaultValue = "python",
        category = "version",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "The absolute path of the Python interpreter invoked by Bazel."
    )
    public String pythonPath;

    @Option(
      name = "experimental_python_import_all_repositories",
      defaultValue = "true",
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Do not use."
    )
    public boolean experimentalPythonImportAllRepositories;

     /**
     * Make Python configuration options available for host configurations as well
     */
    @Override
    public FragmentOptions getHost(boolean fallback) {
      return clone(); // host options are the same as target options
    }
  }

  /**
   * Loader for the Bazel-specific Python configuration.
   */
  public static final class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      BazelPythonConfiguration pythonConfiguration
          = new BazelPythonConfiguration(buildOptions.get(Options.class));

      String pythonPath = pythonConfiguration.getPythonPath();
      if (!pythonPath.equals("python") && !PathFragment.create(pythonPath).isAbsolute()) {
        throw new InvalidConfigurationException(
            "python_path must be an absolute path when it is set.");
      }

      return pythonConfiguration;
    }

    @Override
    public Class<? extends Fragment> creates() {
      return BazelPythonConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(Options.class);
    }
  }

  private final Options options;

  private BazelPythonConfiguration(Options options) {
    this.options = options;
  }

  public String getPython2Path() {
    return options.python2Path;
  }

  public String getPython3Path() {
    return options.python3Path;
  }

  public Label getPythonTop() {
    return options.pythonTop;
  }

  public String getPythonPath() {
    return options.pythonPath;
  }

  public boolean getImportAllRepositories() {
    return options.experimentalPythonImportAllRepositories;
  }

}
