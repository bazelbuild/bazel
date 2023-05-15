// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import net.starlark.java.eval.Module;

/**
 * BazelModuleContext records Bazel-specific information associated with a .bzl {@link
 * net.starlark.java.eval.Module}.
 *
 * <p>Maintainer's note: This object is determined prior to the module's evaluation in
 * BzlLoadFunction. It is saved in the {@code Module} as {@link Module#getClientData client data}.
 * The {@code Module} used during .bzl compilation is separate and uses {@link BazelCompileContext}
 * as client data. For data that is computed after the module's evaluation and which need not be
 * exposed outside the module-loading machinery, consider {@link BzlLoadValue}.
 */
// Immutability is useful because this object is retrievable from a Module and therefore from a
// BzlLoadValue.
@AutoValue
public abstract class BazelModuleContext {
  /** Label associated with the Starlark {@link net.starlark.java.eval.Module}. */
  public abstract Label label();

  /** The repository mapping applicable to the repo where the .bzl file is located in. */
  public abstract RepositoryMapping repoMapping();

  /** Returns the name of the module's .bzl file, as provided to the parser. */
  public abstract String filename();

  /**
   * Returns a list of modules loaded by this .bzl file, in source order.
   *
   * <p>By traversing these modules' loads, it is possible to reconstruct the complete load DAG (not
   * including {@code @_builtins} .bzl files).
   */
  public abstract ImmutableList<Module> loads();

  /**
   * Transitive digest of the .bzl file of the {@link net.starlark.java.eval.Module} itself and all
   * files it transitively loads.
   */
  @SuppressWarnings({"AutoValueImmutableFields", "mutable"})
  @AutoValue.CopyAnnotations
  public abstract byte[] bzlTransitiveDigest();

  /**
   * Returns a label for a {@link net.starlark.java.eval.Module}.
   *
   * <p>This is a user-facing value and we rely on this string to be a valid label for the {@link
   * net.starlark.java.eval.Module} (and that only).
   */
  @Override
  public final String toString() {
    return label().toString();
  }

  /** Returns the BazelModuleContext associated with the specified Starlark module. */
  public static BazelModuleContext of(Module m) {
    return (BazelModuleContext) m.getClientData();
  }

  public static BazelModuleContext create(
      Label label,
      RepositoryMapping repoMapping,
      String filename,
      ImmutableList<Module> loads,
      byte[] bzlTransitiveDigest) {
    return new AutoValue_BazelModuleContext(
        label, repoMapping, filename, loads, bzlTransitiveDigest);
  }

  public final Label.PackageContext packageContext() {
    return Label.PackageContext.of(label().getPackageIdentifier(), repoMapping());
  }
}
