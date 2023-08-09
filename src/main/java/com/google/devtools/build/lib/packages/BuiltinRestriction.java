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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Static utility methods pertaining to restricting Starlark method invocations */
// TODO(bazel-team): Maybe we can merge this utility class with some other existing allowlist
// helper? But it seems like a lot of existing allowlist machinery is geared toward allowlists on
// rule attributes rather than what .bzl you're in.
public final class BuiltinRestriction {

  private BuiltinRestriction() {}

  /**
   * Throws {@code EvalException} if the innermost Starlark function in the given thread's call
   * stack is not defined within the builtins repository.
   *
   * @throws NullPointerException if there is no currently executing Starlark function, or the
   *     innermost Starlark function's module is not a .bzl file
   */
  public static void failIfCalledOutsideBuiltins(StarlarkThread thread) throws EvalException {
    Label currentFile = BazelModuleContext.ofInnermostBzlOrThrow(thread).label();
    if (!currentFile.getRepository().getNameWithAt().equals("@_builtins")) {
      throw Starlark.errorf(
          "file '%s' cannot use private @_builtins API", currentFile.getCanonicalForm());
    }
  }

  /**
   * Throws {@code EvalException} if the innermost Starlark function in the given thread's call
   * stack is not defined within either 1) the builtins repository, or 2) a package or subpackage of
   * an entry in the given allowlist.
   *
   * @throws NullPointerException if there is no currently executing Starlark function, or the
   *     innermost Starlark function's module is not a .bzl file
   */
  public static void failIfCalledOutsideAllowlist(
      StarlarkThread thread, Collection<PackageIdentifier> allowlist) throws EvalException {
    Label currentFile = BazelModuleContext.ofInnermostBzlOrThrow(thread).label();
    failIfLabelOutsideAllowlist(currentFile, allowlist);
  }

  /**
   * Throws {@code EvalException} if the given label is not within either 1) the builtins
   * repository, or 2) a package or subpackage of an entry in the given allowlist.
   *
   * <p>The error message identifies label as a file.
   */
  public static void failIfLabelOutsideAllowlist(
      Label label, Collection<PackageIdentifier> allowlist) throws EvalException {
    if (label.getRepository().getNameWithAt().equals("@_builtins")) {
      return;
    }
    if (allowlist.stream().noneMatch(allowedPkg -> isInPackageOrSubpackage(label, allowedPkg))) {
      throw Starlark.errorf("file '%s' cannot use private API", label.getCanonicalForm());
    }
  }

  private static boolean isInPackageOrSubpackage(Label label, PackageIdentifier packageId) {
    RepositoryName repo = label.getRepository();
    PathFragment pkg = label.getPackageFragment();
    return repo.equals(packageId.getRepository()) && pkg.startsWith(packageId.getPackageFragment());
  }
}
