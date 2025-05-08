// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** Definition of the functions used in VENDOR.bazel file. */
@GlobalMethods(environment = Environment.VENDOR)
public final class VendorFileGlobals {
  private VendorFileGlobals() {}

  public static final VendorFileGlobals INSTANCE = new VendorFileGlobals();

  @StarlarkMethod(
      name = "ignore",
      doc =
          "Ignore this repo from vendoring. Bazel will never vendor it or use the corresponding"
              + " directory (if exists) while building in vendor mode.",
      extraPositionals =
          @Param(name = "args", doc = "The canonical repo names of the repos to ignore."),
      useStarlarkThread = true)
  public void ignore(Tuple args, StarlarkThread thread) throws EvalException {
    VendorThreadContext context = VendorThreadContext.fromOrFail(thread, "ignore()");
    for (String repoName : Sequence.cast(args, String.class, "args")) {
      context.addIgnoredRepo(getRepositoryName(repoName));
    }
  }

  @StarlarkMethod(
      name = "pin",
      doc =
          "Pin the contents of this repo under the vendor directory. Bazel will not update this"
              + " repo while vendoring, and will use the vendored source as if there is a"
              + " --override_repository flag when building in vendor mode",
      extraPositionals =
          @Param(name = "args", doc = "The canonical repo names of the repos to pin."),
      useStarlarkThread = true)
  public void pin(Tuple args, StarlarkThread thread) throws EvalException {
    VendorThreadContext context = VendorThreadContext.fromOrFail(thread, "pin()");
    for (String repoName : Sequence.cast(args, String.class, "args")) {
      context.addPinnedRepo(getRepositoryName(repoName));
    }
  }

  private RepositoryName getRepositoryName(String repoName) throws EvalException {
    if (!repoName.startsWith("@@")) {
      throw Starlark.errorf("the canonical repository name must start with `@@`");
    }
    try {
      repoName = repoName.substring(2);
      return RepositoryName.create(repoName);
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("Invalid canonical repo name: %s", e.getMessage());
    }
  }
}
