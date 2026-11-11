// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.jni;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

/** Provides access to functions exported from Bazel's native library through FFM. */
public final class NativeFunctions {
  private static final Linker LINKER = Linker.nativeLinker();

  private NativeFunctions() {}

  /**
   * Returns a downcall handle for a function exported from the native library loaded by {@link
   * JniLoader}.
   *
   * @throws UnsatisfiedLinkError if the native library doesn't export the function, just like
   *     calling a missing JNI method
   */
  public static MethodHandle downcallHandle(
      String name, FunctionDescriptor descriptor, Linker.Option... options) {
    JniLoader.loadJni();
    // The lookup only finds symbols in libraries loaded by this class's class loader, which is the
    // one used by JniLoader.
    MemorySegment symbol =
        SymbolLookup.loaderLookup()
            .find(name)
            .orElseThrow(() -> new UnsatisfiedLinkError("undefined symbol: " + name));
    return LINKER.downcallHandle(symbol, descriptor, options);
  }
}
