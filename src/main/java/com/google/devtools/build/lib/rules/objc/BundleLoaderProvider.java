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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;

/**
 * Provider containing a subset of ObjcProvider's fields to be propagated to apple_binary targets of
 * the BUNDLE type. This enables compilation and linking against symbols contained in the bundle
 * loader binary without having to redeclare dependencies already part of the bundle loader's
 * dependency graph. This provider explicitly filters out anything that may affect the symbols being
 * linked into the dependents (SDK_DYLIB, SDK_FRAMEWORK, LINKOPT, LIBRARY and FORCE_LOAD_LIBRARY).
 */
public final class BundleLoaderProvider implements TransitiveInfoProvider {

  /** List of keys that BundleLoaderProvider should propagate to bundle binaries. */
  static final ImmutableSet<ObjcProvider.Key<?>> KEPT_KEYS =
      ImmutableSet.<ObjcProvider.Key<?>>of(
          ObjcProvider.HEADER,
          ObjcProvider.INCLUDE,
          ObjcProvider.DEFINE,
          ObjcProvider.DYNAMIC_FRAMEWORK_FILE,
          ObjcProvider.STATIC_FRAMEWORK_FILE,
          ObjcProvider.FRAMEWORK_SEARCH_PATH_ONLY);

  private final ObjcProvider objcProvider;

  /** Creates a new BundleLoader provider that propagates a subset of ObjcProvider's fields, */
  public BundleLoaderProvider(ObjcProvider objcProvider) {
    this.objcProvider = objcProvider;
  }

  /** Returns an ObjcProvider representation of this provider to be used as a dependency. */
  public ObjcProvider toObjcProvider() {
    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();
    for (ObjcProvider.Key<?> key : KEPT_KEYS) {
      addTransitiveAndPropagate(objcProviderBuilder, key);
    }

    return objcProviderBuilder
        // Notice that DYNAMIC_FRAMEWORK_DIR and STATIC_FRAMEWORK_DIR are being rerouted into
        // FRAMEWORK_SEARCH_PATH_ONLY, to avoid linking them into the BUNDLE binary, but making
        // their headers search paths available for compilation.
        .addTransitiveAndPropagate(
            ObjcProvider.FRAMEWORK_SEARCH_PATH_ONLY,
            objcProvider.get(ObjcProvider.DYNAMIC_FRAMEWORK_DIR))
        .addTransitiveAndPropagate(
            ObjcProvider.FRAMEWORK_SEARCH_PATH_ONLY,
            objcProvider.get(ObjcProvider.STATIC_FRAMEWORK_DIR))
        .build();
  }

  private <T> void addTransitiveAndPropagate(
      ObjcProvider.Builder objcProviderBuilder, ObjcProvider.Key<T> key) {
    objcProviderBuilder.addTransitiveAndPropagate(key, objcProvider.get(key));
  }
}
