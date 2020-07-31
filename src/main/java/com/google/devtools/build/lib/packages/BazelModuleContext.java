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

package com.google.devtools.build.lib.packages;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.Module;

/**
 * BazelModuleContext records Bazel-specific information associated with a .bzl {@link
 * com.google.devtools.build.lib.syntax.Module}.
 */
@AutoValue
public abstract class BazelModuleContext {
  /** Label associated with the Starlark {@link com.google.devtools.build.lib.syntax.Module}. */
  public abstract Label label();

  /** Returns the name of the module's .bzl file, as provided to the parser. */
  public abstract String filename();

  /**
   * Maps the load string for each load statement in this .bzl file (in source order) to the module
   * it loads. It thus records the complete load DAG.
   */
  public abstract ImmutableMap<String, Module> loads();

  /**
   * Transitive digest of the .bzl file of the {@link com.google.devtools.build.lib.syntax.Module}
   * itself and all files it transitively loads.
   */
  @SuppressWarnings({"AutoValueImmutableFields", "mutable"})
  @AutoValue.CopyAnnotations
  public abstract byte[] bzlTransitiveDigest();

  /**
   * Returns a label for a {@link com.google.devtools.build.lib.syntax.Module}.
   *
   * <p>This is a user-facing value and we rely on this string to be a valid label for the {@link
   * com.google.devtools.build.lib.syntax.Module} (and that only). Please see the documentation of
   * {@link com.google.devtools.build.lib.syntax.Module#setClientData(Object)} for more details.
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
      String filename,
      ImmutableMap<String, Module> loads,
      byte[] bzlTransitiveDigest) {
    return new AutoValue_BazelModuleContext(label, filename, loads, bzlTransitiveDigest);
  }
}
