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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;

/**
 * Represents a symbolic macro, defined in a .bzl file, that may be instantiated during Package
 * evaluation.
 *
 * <p>This is analogous to {@link RuleClass}. In essence, a {@code MacroClass} consists of the
 * macro's schema and its implementation function.
 */
public final class MacroClass {

  private final String name;
  private final StarlarkFunction implementation;

  public MacroClass(String name, StarlarkFunction implementation) {
    this.name = name;
    this.implementation = implementation;
  }

  /** Returns the macro's exported name. */
  public String getName() {
    return name;
  }

  public StarlarkFunction getImplementation() {
    return implementation;
  }

  /** Builder for {@link MacroClass}. */
  public static final class Builder {
    private final StarlarkFunction implementation;
    @Nullable private String name = null;

    public Builder(StarlarkFunction implementation) {
      this.implementation = implementation;
    }

    @CanIgnoreReturnValue
    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    public MacroClass build() {
      Preconditions.checkNotNull(name);
      return new MacroClass(name, implementation);
    }
  }

  /**
   * Executes a symbolic macro's implementation function, in a new Starlark thread, mutating the
   * given package under construction.
   */
  // TODO: #19922 - Take a new type, PackagePiece.Builder, in place of Package.Builder. PackagePiece
  // would represent the collection of targets/macros instantiated by expanding a single symbolic
  // macro.
  public static void executeMacroImplementation(
      MacroInstance macro, Package.Builder builder, StarlarkSemantics semantics)
      throws InterruptedException {
    try (Mutability mu =
        Mutability.create("macro", builder.getPackageIdentifier(), macro.getName())) {
      StarlarkThread thread =
          StarlarkThread.create(
              mu,
              semantics,
              /* contextDescription= */ "",
              SymbolGenerator.create(
                  MacroId.create(builder.getPackageIdentifier(), macro.getName())));
      thread.setPrintHandler(Event.makeDebugPrintHandler(builder.getLocalEventHandler()));

      // TODO: #19922 - Technically the embedded SymbolGenerator field should use a different key
      // than the one in the main BUILD thread, but that'll be fixed when we change the type to
      // PackagePiece.Builder.
      builder.storeInThread(thread);

      // TODO: #19922 - If we want to support creating analysis_test rules inside symbolic macros,
      // we'd need to call `thread.setThreadLocal(RuleDefinitionEnvironment.class,
      // ruleClassProvider)`. In that case we'll need to consider how to get access to the
      // ConfiguredRuleClassProvider. For instance, we could put it in the builder.

      try {
        Starlark.fastcall(
            thread,
            macro.getMacroClass().getImplementation(),
            /* positional= */ new Object[] {},
            /* named= */ new Object[] {"name", macro.getName()});
      } catch (EvalException ex) {
        builder
            .getLocalEventHandler()
            .handle(
                Package.error(
                    /* location= */ null, ex.getMessageWithStack(), Code.STARLARK_EVAL_ERROR));
        builder.setContainsErrors();
      }
    }
  }

  @AutoValue
  abstract static class MacroId {
    static MacroId create(PackageIdentifier id, String name) {
      return new AutoValue_MacroClass_MacroId(id, name);
    }

    abstract PackageIdentifier packageId();

    abstract String name();
  }
}
