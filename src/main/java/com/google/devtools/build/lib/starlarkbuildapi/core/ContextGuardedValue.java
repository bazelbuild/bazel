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
   * Creates a guard which only permits access of the given object when the requesting .bzl file is
   * in a specific patckage path. An error is thrown if accessing it is done outside the allowed
   * package paths.
   */
  public static GuardedValue onlyInAllowedRepos(
      Object obj, ImmutableSet<PackageIdentifier> allowedPrefixes) {
    return new GuardedValue() {
      @Override
      public boolean isObjectAccessibleUsingSemantics(
          StarlarkSemantics semantics, @Nullable Object clientData) {
        // Filtering of predeclareds is only done at compile time, when the client data is
        // BazelCompileContext and not BazelModuleContext.
        if (clientData != null && clientData instanceof BazelCompileContext) {
          BazelCompileContext context = (BazelCompileContext) clientData;
          Label label = context.label();

          for (PackageIdentifier prefix : allowedPrefixes) {
            if (label.getRepository().equals(prefix.getRepository())
                && label.getPackageFragment().startsWith(prefix.getPackageFragment())) {
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
            + allowedPrefixes.stream()
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
