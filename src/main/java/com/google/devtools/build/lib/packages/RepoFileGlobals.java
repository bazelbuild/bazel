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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.GlobalMethodDocs;
import com.google.devtools.build.docgen.annot.GlobalMethodDocs.Environment;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkLibrary;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Definition of the {@code repo()} function used in REPO.bazel files. */
@GlobalMethodDocs(environment = Environment.REPO)
@StarlarkLibrary
public final class RepoFileGlobals {
  private RepoFileGlobals() {}

  public static final RepoFileGlobals INSTANCE = new RepoFileGlobals();

  @StarlarkMethod(
      name = "ignore_directories",
      doc =
          "The list of directories to ignore in this repository. <p>This function takes a list"
              + " of strings and a directory is ignored if any of the given strings matches its"
              + " repository-relative path according to the semantics of the <code>glob()</code>"
              + " function. This function can be used to ignore directories that are implementation"
              + " details of source control systems, output files of other build systems, etc.",
      useStarlarkThread = true,
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
      name = "traversal_ignore_directories",
      doc =
          "The list of directories in this repository to skip during recursive target pattern"
              + " expansion. <p>This function takes a list of repository-relative directory paths"
              + " such as <code>\"experimental\"</code>. Recursive target patterns rooted above"
              + " such a directory, such as <code>//...</code>, skip it and everything beneath it."
              + " Unlike <code>ignore_directories()</code>, however, the packages there remain"
              + " ordinary, fully buildable packages; naming one explicitly, as in"
              + " <code>//experimental/...</code> or <code>//experimental/foo:bar</code>, works as"
              + " usual.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "dirs",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
            })
      })
  public void traversalIgnoreDirectories(Iterable<?> dirsUnchecked, StarlarkThread thread)
      throws EvalException {
    Sequence<String> dirs = Sequence.cast(dirsUnchecked, String.class, "dirs");
    RepoThreadContext context = RepoThreadContext.fromOrFail(thread, "repo()");

    if (context.isTraversalIgnoreDirectoriesSet()) {
      throw new EvalException("'traversal_ignore_directories()' can only be called once");
    }

    ImmutableList.Builder<PathFragment> fragments =
        ImmutableList.builderWithExpectedSize(dirs.size());
    for (String dir : dirs) {
      if (dir.isEmpty() || dir.equals(".")) {
        // Ignoring the root directory would ignore the entire repository, which is surely a
        // mistake.
        throw Starlark.errorf(
            "invalid traversal_ignore_directories entry '%s': ignores the entire workspace", dir);
      }
      if (!PathFragment.isNormalizedRelativePath(dir)) {
        throw Starlark.errorf(
            "invalid traversal_ignore_directories entry '%s': path must be normalized and"
                + " relative to the workspace",
            dir);
      }
      String error = LabelValidator.validatePackageName(dir);
      if (error != null) {
        throw Starlark.errorf("invalid traversal_ignore_directories entry '%s': %s", dir, error);
      }
      fragments.add(PathFragment.create(dir));
    }
    context.setTraversalIgnoreDirectories(fragments.build());
  }

  @StarlarkMethod(
      name = "repo",
      doc =
          "Declares metadata that applies to every rule in the repository. It must be called at "
              + "most once per REPO.bazel file. If called, it must be the first call in the "
              + "REPO.bazel file.",
      extraKeywords =
          @Param(
              name = "kwargs",
              doc =
                  "The <code>repo()</code> function accepts exactly the same arguments as the "
                      + "<a href=\"${link functions}#package\"><code>package()</code></a> function "
                      + "in BUILD files."),
      useStarlarkThread = true)
  public void repoCallable(Map<String, Object> kwargs, StarlarkThread thread) throws EvalException {
    RepoThreadContext context = RepoThreadContext.fromOrFail(thread, "repo()");
    if (context.isRepoFunctionCalled()) {
      throw Starlark.errorf("'repo' can only be called once in the REPO.bazel file");
    }

    if (context.isIgnoredDirectoriesSet() || context.isTraversalIgnoreDirectoriesSet()) {
      throw Starlark.errorf("if repo() is called, it must be called before any other functions");
    }

    if (kwargs.isEmpty()) {
      throw Starlark.errorf("at least one argument must be given to the 'repo' function");
    }

    context.setPackageArgsMap(kwargs);
  }
}
