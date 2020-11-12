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

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.rules.python.PythonOptions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;

/** Bazel-specific Python configuration. */
@Immutable
@RequiresOptions(options = {BazelPythonConfiguration.Options.class, PythonOptions.class})
public class BazelPythonConfiguration extends Fragment {

  /** Bazel-specific Python configuration options. */
  public static final class Options extends FragmentOptions {
    @Option(
        name = "python2_path",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated, no-op. Disabled by `--incompatible_use_python_toolchains`.")
    public String python2Path;

    @Option(
        name = "python3_path",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated, no-op. Disabled by `--incompatible_use_python_toolchains`.")
    public String python3Path;

    @Option(
        name = "python_top",
        converter = LabelConverter.class,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
        help =
            "The label of a py_runtime representing the Python interpreter invoked to run Python "
                + "targets on the target platform. Deprecated; disabled by "
                + "--incompatible_use_python_toolchains.")
    public Label pythonTop;

    @Option(
        name = "python_path",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
        help =
            "The absolute path of the Python interpreter invoked to run Python targets on the "
                + "target platform. Deprecated; disabled by --incompatible_use_python_toolchains.")
    public String pythonPath;

    @Option(
        name = "experimental_python_import_all_repositories",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        help =
            "If true, the roots of repositories in the runfiles tree are added to PYTHONPATH, so "
                + "that imports like `import mytoplevelpackage.package.module` are valid."
                + " Regardless of whether this flag is true, the runfiles root itself is also"
                + " added to the PYTHONPATH, so "
                + "`import myreponame.mytoplevelpackage.package.module` is valid. The latter form "
                + "is less likely to experience import name collisions.")
    public boolean experimentalPythonImportAllRepositories;

     /**
     * Make Python configuration options available for host configurations as well
     */
    @Override
    public FragmentOptions getHost() {
      return clone(); // host options are the same as target options
    }
  }

  /**
   * Loader for the Bazel-specific Python configuration.
   */
  public static final class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new BazelPythonConfiguration(buildOptions);
    }

    @Override
    public Class<? extends Fragment> creates() {
      return BazelPythonConfiguration.class;
    }
  }

  private final Options options;

  private BazelPythonConfiguration(BuildOptions buildOptions) throws InvalidConfigurationException {
    this.options = buildOptions.get(Options.class);
    String pythonPath = getPythonPath();
    if (!pythonPath.startsWith("python") && !PathFragment.create(pythonPath).isAbsolute()) {
      throw new InvalidConfigurationException(
          "python_path must be an absolute path when it is set.");
    }
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    PythonOptions pythonOpts = buildOptions.get(PythonOptions.class);
    Options opts = buildOptions.get(Options.class);
    if (pythonOpts.incompatibleUsePythonToolchains) {
      // Forbid deprecated flags.
      if (opts.python2Path != null) {
        reporter.handle(
            Event.error(
                "`--python2_path` is disabled by `--incompatible_use_python_toolchains`. Since "
                    + "`--python2_path` is a deprecated no-op, there is no need to pass it."));
      }
      if (opts.python3Path != null) {
        reporter.handle(
            Event.error(
                "`--python3_path` is disabled by `--incompatible_use_python_toolchains`. Since "
                    + "`--python3_path` is a deprecated no-op, there is no need to pass it."));
      }
      if (opts.pythonTop != null) {
        reporter.handle(
            Event.error(
                "`--python_top` is disabled by `--incompatible_use_python_toolchains`. Instead of "
                    + "configuring the Python runtime directly, register a Python toolchain. See "
                    + "https://github.com/bazelbuild/bazel/issues/7899. You can temporarily revert "
                    + "to the legacy flag-based way of specifying toolchains by setting "
                    + "`--incompatible_use_python_toolchains=false`."));
      }
      // TODO(#7901): Also prohibit --python_path here.
    }
  }

  public Label getPythonTop() {
    return options.pythonTop;
  }

  public String getPythonPath() {
    return options.pythonPath == null ? "python" : options.pythonPath;
  }

  public boolean getImportAllRepositories() {
    return options.experimentalPythonImportAllRepositories;
  }

}
