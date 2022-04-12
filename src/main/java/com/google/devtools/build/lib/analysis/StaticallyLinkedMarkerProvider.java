// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/**
 * A marker provider for rules that are to be linked statically.
 *
 * <p>Used in license checking.
 */
@Immutable
@StarlarkBuiltin(
    name = "StaticallyLinkedMarkerProvider",
    category = DocCategory.BUILTIN,
    documented = false)
public class StaticallyLinkedMarkerProvider extends NativeInfo implements StarlarkValue {
  private final boolean isLinkedStatically;

  /** Provider class for {@link StaticallyLinkedMarkerProvider} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  public static class Provider extends BuiltinProvider<StaticallyLinkedMarkerProvider> {
    private Provider() {
      super("StaticallyLinkedMarkerProvider", StaticallyLinkedMarkerProvider.class);
    }

    @StarlarkMethod(
        name = "StaticallyLinkedMarkerProvider",
        documented = false,
        selfCall = true,
        parameters = {@Param(name = "is_linked_statically", positional = false, named = true)})
    @StarlarkConstructor
    public StaticallyLinkedMarkerProvider createStaticallyLinkedMarkerProvider(
        boolean isLinkedStatically) {
      return new StaticallyLinkedMarkerProvider(isLinkedStatically);
    }
  }

  public static final Provider PROVIDER = new Provider();

  public StaticallyLinkedMarkerProvider(boolean isLinkedStatically) {
    this.isLinkedStatically = isLinkedStatically;
  }

  public boolean isLinkedStatically() {
    return isLinkedStatically;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }
}
