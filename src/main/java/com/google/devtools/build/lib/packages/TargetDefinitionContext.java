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

import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;

/**
 * A context object, usually stored in a {@link net.starlark.java.eval.StarlarkThread}, upon which
 * rules and symbolic macros can be instantiated.
 */
// TODO(#19922): This class isn't really needed until we implement lazy macro evaluation. At that
// point, we'll need to split the concept of a Package.Builder into a separate PackagePiece.Builder
// that represents the object produced by evaluating a macro implementation. Then we can factor the
// accessors and mutations that are common to BUILD files / lazy macros and to symbolic macros into
// this common parent class, while Package.Builder retains the stuff that's prohibited inside
// symbolic macros.
public abstract class TargetDefinitionContext extends StarlarkThreadContext {

  /**
   * An exception used when the name of a target or symbolic macro clashes with another entity
   * defined in the package.
   *
   * <p>Common examples of conflicts include two targets or symbolic macros sharing the same name,
   * and one output file being a prefix of another. See {@link Package.Builder#checkForExistingName}
   * and {@link Package.Builder#checkRuleAndOutputs} for more details.
   */
  public static sealed class NameConflictException extends Exception
      permits MacroNamespaceViolationException {
    public NameConflictException(String message) {
      super(message);
    }
  }

  /**
   * An exception used when the name of a target or submacro declared within a symbolic macro
   * violates symbolic macro naming rules.
   *
   * <p>An example might be a target named "libfoo" declared within a macro named "foo".
   */
  public static final class MacroNamespaceViolationException extends NameConflictException {
    public MacroNamespaceViolationException(String message) {
      super(message);
    }
  }

  protected TargetDefinitionContext(RepositoryMapping mainRepoMapping) {
    super(() -> mainRepoMapping);
  }
}
