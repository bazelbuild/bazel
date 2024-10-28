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
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Definition of the {@code repo()} function used in REPO.bazel files. */
public final class RepoFileGlobals {
  private RepoFileGlobals() {}

  public static final RepoFileGlobals INSTANCE = new RepoFileGlobals();

  @StarlarkMethod(
      name = "ignore_directories",
      useStarlarkThread = true,
      documented = false, // TODO
      parameters = {
        @Param(
            name = "dirs",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
            })
      })
  public void ignoreDirectories(Iterable<?> dirsUnchecked, StarlarkThread thread)
      throws EvalException {
    Sequence<String> dirs = Sequence.cast(dirsUnchecked, String.class, "dirs");
    RepoThreadContext context = RepoThreadContext.fromOrFail(thread, "repo()");

    if (context.isIgnoredDirectoriesSet()) {
      throw new EvalException("'ignored_directories()' can only be called once");
    }

    context.setIgnoredDirectories(dirs);
  }

  @StarlarkMethod(
      name = "repo",
      documented = false, // documented separately
      extraKeywords = @Param(name = "kwargs"),
      useStarlarkThread = true)
  public void repoCallable(Map<String, Object> kwargs, StarlarkThread thread)
      throws EvalException {
    RepoThreadContext context = RepoThreadContext.fromOrFail(thread, "repo()");
    if (context.isRepoFunctionCalled()) {
      throw Starlark.errorf("'repo' can only be called once in the REPO.bazel file");
    }

    if (context.isIgnoredDirectoriesSet()) {
      throw Starlark.errorf("if repo() is called, it must be called before any other functions");
    }

    if (kwargs.isEmpty()) {
      throw Starlark.errorf("at least one argument must be given to the 'repo' function");
    }

    context.setPackageArgsMap(kwargs);
  }
}
