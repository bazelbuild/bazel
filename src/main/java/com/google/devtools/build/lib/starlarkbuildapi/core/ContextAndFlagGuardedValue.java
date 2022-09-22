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
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.FlagGuardedValue;
import net.starlark.java.eval.GuardedValue;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Wrapper on a value in the predeclared lexical block that controls its accessibility to Starlark
 * based on the value of a semantic flag and context, in particular the package path the requesting
 * .bzl file falls under.
 */
public final class ContextAndFlagGuardedValue {
  /**
   * Creates a flag guard which only permits access of the given object when the given boolean flag
   * is false or the requesting .bzl file is in a specific patckage path. If the given flag is true
   * and the object would be accessed, an error is thrown describing the feature as deprecated, and
   * describing that the flag may be set to false to re-enable it.
   *
   * <p>The flag identifier must have a + or - prefix; see StarlarkSemantics.
   */
  public static GuardedValue onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
      String flag, Object obj, ImmutableSet<PackageIdentifier> allowedPrefixes) {
    GuardedValue flagGuard = FlagGuardedValue.onlyWhenIncompatibleFlagIsFalse(flag, obj);
    return new GuardedValue() {
      @Override
      public boolean isObjectAccessibleUsingSemantics(
          StarlarkSemantics semantics, @Nullable Object clientData) {
        boolean accessible = flagGuard.isObjectAccessibleUsingSemantics(semantics, clientData);
        if (!accessible && clientData != null) {
          BazelModuleContext context = (BazelModuleContext) clientData;
          Label label = context.label();

          for (PackageIdentifier prefix : allowedPrefixes) {
            if (label.getRepository().equals(prefix.getRepository())
                && label.getPackageFragment().startsWith(prefix.getPackageFragment())) {
              return true;
            }
          }
        }
        return accessible;
      }

      @Override
      public String getErrorFromAttemptingAccess(String name) {
        return name
            + " may only be used from one of the following repositories or prefixes: "
            + allowedPrefixes.stream()
                .map(PackageIdentifier::toString)
                .collect(Collectors.joining(", "))
            + ". It may be temporarily re-enabled for general use by setting --"
            + flag.substring(1)
            + "=false";
      }

      @Override
      public Object getObject() {
        return obj;
      }
    };
  }

  private ContextAndFlagGuardedValue() {}
}
