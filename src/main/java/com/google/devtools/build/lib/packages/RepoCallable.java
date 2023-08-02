// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Definition of the {@code repo()} function used in REPO.bazel files. */
public final class RepoCallable {
  private RepoCallable() {}

  public static final RepoCallable INSTANCE = new RepoCallable();

  @StarlarkMethod(
      name = "repo",
      documented = false, // documented separately
      extraKeywords = @Param(name = "kwargs"),
      useStarlarkThread = true)
  public Object repoCallable(Map<String, Object> kwargs, StarlarkThread thread)
      throws EvalException {
    RepoThreadContext context = RepoThreadContext.fromOrFail(thread, "repo()");
    if (context.isRepoFunctionCalled()) {
      throw Starlark.errorf("'repo' can only be called once in the REPO.bazel file");
    }
    context.setRepoFunctionCalled();

    if (kwargs.isEmpty()) {
      throw Starlark.errorf("at least one argument must be given to the 'repo' function");
    }

    PackageArgs.Builder pkgArgsBuilder = PackageArgs.builder();
    for (Map.Entry<String, Object> kwarg : kwargs.entrySet()) {
      PackageArgs.processParam(
          kwarg.getKey(),
          kwarg.getValue(),
          "repo() argument '" + kwarg.getKey() + "'",
          context.getLabelConverter(),
          pkgArgsBuilder);
    }
    context.setPackageArgs(pkgArgsBuilder.build());
    return Starlark.NONE;
  }
}
