// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.python;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Implementation for the {@code py_runtime} rule. */
public final class PyRuntime implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws ActionConflictException, InterruptedException {
    PythonConfiguration pyConfig = ruleContext.getFragment(PythonConfiguration.class);

    NestedSet<Artifact> files = PrerequisiteArtifacts.nestedSet(ruleContext, "files");
    Artifact interpreter = ruleContext.getPrerequisiteArtifact("interpreter");
    PathFragment interpreterPath =
        PathFragment.create(ruleContext.attributes().get("interpreter_path", Type.STRING));
    PythonVersion pythonVersion =
        PythonVersion.parseTargetOrSentinelValue(
            ruleContext.attributes().get("python_version", Type.STRING));

    // Determine whether we're pointing to an in-build target (hermetic) or absolute system path
    // (non-hermetic).
    if ((interpreter == null) == interpreterPath.isEmpty()) {
      ruleContext.ruleError(
          "exactly one of the 'interpreter' or 'interpreter_path' attributes must be specified");
    }
    boolean hermetic = interpreter != null;
    // Validate attributes.
    if (!hermetic && !files.isEmpty()) {
      ruleContext.ruleError("if 'interpreter_path' is given then 'files' must be empty");
    }
    if (!hermetic && !interpreterPath.isAbsolute()) {
      ruleContext.attributeError("interpreter_path", "must be an absolute path.");
    }

    if (pythonVersion == PythonVersion._INTERNAL_SENTINEL) {
      if (pyConfig.useToolchains()) {
        ruleContext.attributeError(
            "python_version",
            "When using Python toolchains, this attribute must be set explicitly to either 'PY2' "
                + "or 'PY3'. See https://github.com/bazelbuild/bazel/issues/7899 for more "
                + "information. You can temporarily avoid this error by reverting to the legacy "
                + "Python runtime mechanism (`--incompatible_use_python_toolchains=false`).");
      } else {
        // Use the same default as py_binary/py_test would use for their python_version attribute.
        // (Of course, in our case there's no configuration transition involved.)
        pythonVersion = pyConfig.getDefaultPythonVersion();
      }
    }

    if (ruleContext.hasErrors()) {
      return null;
    }
    Preconditions.checkState(pythonVersion.isTargetValue());

    PyRuntimeInfo provider =
        hermetic
            ? PyRuntimeInfo.createForInBuildRuntime(interpreter, files, pythonVersion)
            : PyRuntimeInfo.createForPlatformRuntime(interpreterPath, pythonVersion);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(files)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addNativeDeclaredProvider(provider)
        .build();
  }

}
