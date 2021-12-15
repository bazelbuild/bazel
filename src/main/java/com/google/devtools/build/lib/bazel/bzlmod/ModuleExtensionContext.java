// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkBaseExternalContext;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;

/** The Starlark object passed to the implementation function of module extensions. */
@StarlarkBuiltin(
    name = "module_ctx",
    category = DocCategory.BUILTIN,
    doc =
        "The context of the module extension containing helper functions and information about"
            + " pertinent tags across the dependency graph. You get a module_ctx object as an"
            + " argument to the <code>implementation</code> function when you create a module"
            + " extension.")
public class ModuleExtensionContext extends StarlarkBaseExternalContext {
  private final StarlarkList<StarlarkBazelModule> modules;

  protected ModuleExtensionContext(
      Path workingDirectory,
      Environment env,
      Map<String, String> envVariables,
      DownloadManager downloadManager,
      double timeoutScaling,
      @Nullable ProcessWrapper processWrapper,
      StarlarkSemantics starlarkSemantics,
      @Nullable RepositoryRemoteExecutor remoteExecutor,
      StarlarkList<StarlarkBazelModule> modules) {
    super(
        workingDirectory,
        env,
        envVariables,
        downloadManager,
        timeoutScaling,
        processWrapper,
        starlarkSemantics,
        remoteExecutor);
    this.modules = modules;
  }

  public Path getWorkingDirectory() {
    return workingDirectory;
  }

  @Override
  protected String getIdentifyingStringForLogging() {
    return "TODO";
  }

  @Override
  protected boolean isRemotable() {
    // Maybe we can some day support remote execution, but not today.
    return false;
  }

  @Override
  protected ImmutableMap<String, String> getRemoteExecProperties() throws EvalException {
    return ImmutableMap.of();
  }

  @StarlarkMethod(
      name = "modules",
      structField = true,
      doc =
          "A list of all the Bazel modules in the external dependency graph, each of which is a <a"
              + " href=\"bazel_module.html\">bazel_module</a> object that exposes all the tags it"
              + " specified for this module extension. The iteration order of this dictionary is"
              + " guaranteed to be the same as breadth-first search starting from the root module.")
  public StarlarkList<StarlarkBazelModule> getModules() {
    return modules;
  }
}
