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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.starlark.annotations.StarlarkConfigurationField;
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
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** Bazel-specific Python configuration. */
@Immutable
@StarlarkBuiltin(name = "bazel_py", category = DocCategory.CONFIGURATION_FRAGMENT)
@RequiresOptions(options = {BazelPythonConfiguration.Options.class, PythonOptions.class})
public class BazelPythonConfiguration extends Fragment {

  /** Bazel-specific Python configuration options. */
  public static final class Options extends FragmentOptions {
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
        metadataTags = {OptionMetadataTag.EXPERIMENTAL},
        help =
            "If true, the roots of repositories in the runfiles tree are added to PYTHONPATH, so "
                + "that imports like `import mytoplevelpackage.package.module` are valid."
                + " Regardless of whether this flag is true, the runfiles root itself is also"
                + " added to the PYTHONPATH, so "
                + "`import myreponame.mytoplevelpackage.package.module` is valid. The latter form "
                + "is less likely to experience import name collisions.")
    public boolean experimentalPythonImportAllRepositories;

    @Option(
        name = "incompatible_remove_ctx_bazel_py_fragment",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
        help =
            "When true, Python build flags are defined with Python rules (in BUIILD files) and"
                + " ctx.fragments.bazel_py is undefined. This is a migration flag to move all Python"
                + " flags from core Bazel to Python rules.")
    public boolean disablePyFragment;
  }

  private final Options options;

  public BazelPythonConfiguration(BuildOptions buildOptions) throws InvalidConfigurationException {
    this.options = buildOptions.get(Options.class);
    String pythonPath = getPythonPath();
    if (!pythonPath.startsWith("python") && !PathFragment.create(pythonPath).isAbsolute()) {
      throw new InvalidConfigurationException(
          "python_path must be an absolute path when it is set.");
    }
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    Options opts = buildOptions.get(Options.class);
      // Forbid deprecated flags.
      if (opts.pythonTop != null) {
      reporter.handle(
          Event.error(
              "`--python_top` is disabled by `--incompatible_use_python_toolchains`. Instead of "
                  + "configuring the Python runtime directly, register a Python toolchain. See "
                  + "https://github.com/bazelbuild/bazel/issues/7899. You can temporarily revert "
                  + "to the legacy flag-based way of specifying toolchains by setting "
                  + "`--incompatible_use_python_toolchains=false`."));
      // TODO(#7901): Also prohibit --python_path here.
    }
  }

  public boolean shouldInclude() {
    return !options.disablePyFragment;
  }

  @StarlarkConfigurationField(
      name = "python_top",
      doc = "The value of the --python_top flag; may be None if not specified")
  public Label getPythonTop() {
    return options.pythonTop;
  }

  @StarlarkMethod(
      name = "python_path",
      structField = true,
      doc = "The value of the --python_path flag.")
  public String getPythonPath() {
    return options.pythonPath == null ? "python" : options.pythonPath;
  }

  @StarlarkMethod(
      name = "python_import_all_repositories",
      structField = true,
      doc = "The value of the --experimental_python_import_all_repositories flag.")
  public boolean getImportAllRepositories() {
    return options.experimentalPythonImportAllRepositories;
  }
}
