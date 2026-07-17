// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.core;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.BazelCompileContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.GuardedValue;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Wrapper on a value in the predeclared lexical block that controls its accessibility to Starlark
 * based on the context, in particular the package path the requesting .bzl file falls under.
 */
public final class ContextGuardedValue {

  private ContextGuardedValue() {}

  /**
   * Creates a guard which only permits access of the given object when the requesting .bzl file's
   * repo name matches one of the (repoNameSubstr, pathPrefix) entries (note: under bzlmod, repos
   * are prefixed with the bzlmod module name). An error is thrown if accessing it is done outside
   * the allowed package paths.
   */
  public static GuardedValue onlyInAllowedRepos(
      Object obj, ImmutableSet<PackageIdentifier> allowedEntries) {
    return new GuardedValue() {
      @Override
      public boolean isObjectAccessibleUsingSemantics(
          StarlarkSemantics semantics, @Nullable Object clientData) {
        // Filtering of predeclareds is only done at compile time, when the client data is
        // BazelCompileContext and not BazelModuleContext.
        if (clientData != null && clientData instanceof BazelCompileContext context) {
          Label label = context.label();

          for (PackageIdentifier entry : allowedEntries) {
            String pattern;
            if (entry.getRepository().getName().isEmpty()) {
              // String.matches has ^$ implicilty, so an empty pattern matches exactly the empty
              // string.
              pattern = "";
            } else {
              // Surrounding .* because String.matches has implicit "^$" anchor.
              // Surrounding \b so it doesn't match a substring of the intended repo name.
              // Quote the name so dots in the repo name don't get treated as part of the pattern
              pattern = ".*\\b" + Pattern.quote(entry.getRepository().getName()) + "\\b.*";
            }
            if (label.getRepository().getName().matches(pattern)
                && label.getPackageFragment().startsWith(entry.getPackageFragment())) {
              return true;
            }
          }
        }
        return false;
      }

      @Override
      public String getErrorFromAttemptingAccess(String name) {
        return name
            + " may only be used from one of the following repositories or prefixes: "
            + allowedEntries.stream()
                .map(PackageIdentifier::toString)
                .collect(Collectors.joining(", "));
      }

      @Override
      public Object getObject() {
        return obj;
      }
    };
  }
}
