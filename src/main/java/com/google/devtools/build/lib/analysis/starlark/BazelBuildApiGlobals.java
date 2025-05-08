// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.BzlInitThreadContext;
import com.google.devtools.build.lib.packages.BzlVisibility;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkBuildApiGlobals;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/**
 * Bazel implementation of {@link StarlarkBuildApiGlobals}: a collection of global Starlark build
 * API functions that belong in the global namespace.
 */
// TODO(bazel-team): Consider renaming this file BzlGlobals for consistency with BuildGlobals.
// Maybe wait until after eliminating the StarlarkBuildApiGlobals interface along with the rest of
// the starlarkbuildapi/ dir.
public class BazelBuildApiGlobals implements StarlarkBuildApiGlobals {

  @Override
  public void visibility(Object value, StarlarkThread thread) throws EvalException {
    // Confirm load visibility is enabled. We manually check the experimental flag here because
    // StarlarkMethod.enableOnlyWithFlag doesn't work for top-level builtins.
    if (!thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_BZL_VISIBILITY)) {
      throw Starlark.errorf("Use of `visibility()` requires --experimental_bzl_visibility");
    }

    // Fail if we're not initializing a .bzl module
    BzlInitThreadContext context = BzlInitThreadContext.fromOrFail(thread, "visibility()");
    // Fail if we're not called from the top level. (We prohibit calling visibility() from within
    // helper functions because it's more magical / less readable, and it makes it more difficult
    // for static tooling to mechanically find and modify visibility() declarations.)
    ImmutableList<StarlarkThread.CallStackEntry> callStack = thread.getCallStack();
    if (!(callStack.size() == 2
        && callStack.get(0).name.equals(StarlarkThread.TOP_LEVEL)
        && callStack.get(1).name.equals("visibility"))) {
      throw Starlark.errorf(
          "load visibility may only be set at the top level, not inside a function");
    }

    // Fail if the module's visibility is already set.
    if (context.getBzlVisibility() != null) {
      throw Starlark.errorf("load visibility may not be set more than once");
    }

    RepositoryName repo = context.getBzlFile().getRepository();
    ImmutableList<PackageSpecification> specs;
    if (value instanceof String) {
      // `visibility("public")`, `visibility("private")`, visibility("//pkg")
      specs =
          ImmutableList.of(PackageSpecification.fromStringForBzlVisibility(repo, (String) value));
    } else if (value instanceof StarlarkList) {
      // `visibility(["//pkg1", "//pkg2", ...])`
      List<String> specStrings = Sequence.cast(value, String.class, "visibility list");
      ImmutableList.Builder<PackageSpecification> specsBuilder =
          ImmutableList.builderWithExpectedSize(specStrings.size());
      for (String specString : specStrings) {
        PackageSpecification spec =
            PackageSpecification.fromStringForBzlVisibility(repo, specString);
        specsBuilder.add(spec);
      }
      specs = specsBuilder.build();
    } else {
      throw Starlark.errorf(
          "Invalid visibility: got '%s', want string or list of strings", Starlark.type(value));
    }
    context.setBzlVisibility(BzlVisibility.of(specs));
  }

  @Override
  public StarlarkLateBoundDefault<?> configurationField(
      String fragment, String name, StarlarkThread thread) throws EvalException {
    BzlInitThreadContext context = BzlInitThreadContext.fromOrFail(thread, "configuration_field()");
    Class<?> fragmentClass = context.getFragmentNameToClass().get(fragment);
    if (fragmentClass == null) {
      throw Starlark.errorf("invalid configuration fragment name '%s'", fragment);
    }
    try {
      return StarlarkLateBoundDefault.forConfigurationField(
          fragmentClass, name, context.getToolsRepository());
    } catch (StarlarkLateBoundDefault.InvalidConfigurationFieldException exception) {
      throw new EvalException(exception);
    }
  }
}
