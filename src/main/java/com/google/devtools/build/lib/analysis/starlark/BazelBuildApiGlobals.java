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

import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.BzlInitThreadContext;
import com.google.devtools.build.lib.packages.BzlVisibility;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkBuildApiGlobals;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/**
 * Bazel implementation of {@link StarlarkBuildApiGlobals}: a collection of global Starlark build
 * API functions that belong in the global namespace.
 */
// TODO(brandjon): This should probably be refactored into a StarlarkLibrary#BZL field, analogous to
// StarlarkLibrary#COMMON and StarlarkLibrary#BUILD.
public class BazelBuildApiGlobals implements StarlarkBuildApiGlobals {

  @Override
  public void visibility(String value, StarlarkThread thread) throws EvalException {
    // Manually check the experimental flag because enableOnlyWithFlag doesn't work for top-level
    // builtins.
    if (!thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_BZL_VISIBILITY)) {
      throw Starlark.errorf("Use of `visibility()` requires --experimental_bzl_visibility");
    }

    BzlInitThreadContext context = BzlInitThreadContext.fromOrFailFunction(thread, "visibility");
    if (context.getBzlVisibility() != null) {
      throw Starlark.errorf(".bzl visibility may not be set more than once");
    }

    // Check the currently initializing .bzl file's package against this experimental feature's
    // allowlist. BuildLanguageOptions isn't allowed to depend on Label, etc., so this is
    // represented as a list of strings. For simplicity we convert the strings to PackageIdentifiers
    // here, at linear cost and redundantly for each call to `visibility()`. This is ok because the
    // allowlist is temporary, expected to remain small, and calls to visibility() are relatively
    // infrequent.
    List<String> allowlist =
        thread.getSemantics().get(BuildLanguageOptions.EXPERIMENTAL_BZL_VISIBILITY_ALLOWLIST);
    PackageIdentifier pkgId = context.getBzlFile().getPackageIdentifier();
    boolean foundMatch = false;
    for (String allowedPkgString : allowlist) {
      // TODO(b/22193153): This seems incorrect since parse doesn't take into account any repository
      // map. (This shouldn't matter within Google's monorepo, which doesn't use a repo map.)
      try {
        PackageIdentifier allowedPkgId = PackageIdentifier.parse(allowedPkgString);
        if (pkgId.equals(allowedPkgId)) {
          foundMatch = true;
          break;
        }
      } catch (LabelSyntaxException ex) {
        throw new EvalException("Invalid bzl visibility allowlist", ex);
      }
    }
    if (!foundMatch) {
      throw Starlark.errorf(
          "`visibility() is not enabled for package %s; consider adding it to "
              + "--experimental_bzl_visibility_allowlist",
          pkgId.getCanonicalForm());
    }

    BzlVisibility bzlVisibility;
    if (value.equals("public")) {
      bzlVisibility = BzlVisibility.PUBLIC;
    } else if (value.equals("private")) {
      bzlVisibility = BzlVisibility.PRIVATE;
    } else {
      throw Starlark.errorf("Invalid .bzl visibility: '%s'", value);
    }
    context.setBzlVisibility(bzlVisibility);
  }

  @Override
  public StarlarkLateBoundDefault<?> configurationField(
      String fragment, String name, StarlarkThread thread) throws EvalException {
    BazelStarlarkContext context = BazelStarlarkContext.from(thread);
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
