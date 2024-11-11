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

import java.util.List;
import net.starlark.java.eval.StarlarkSemantics;

/** Listener for package-loading events. */
public interface PackageLoadingListener {

  PackageLoadingListener NOOP_LISTENER = (pkg, semantics, loadTimeNanos) -> {};

  /** Returns a {@link PackageLoadingListener} from a composed of the input listeners. */
  static PackageLoadingListener create(List<PackageLoadingListener> listeners) {
    return switch (listeners.size()) {
      case 0 -> NOOP_LISTENER;
      case 1 -> listeners.get(0);
      default ->
          (pkg, semantics, loadTimeNanos) -> {
            for (PackageLoadingListener listener : listeners) {
              listener.onLoadingCompleteAndSuccessful(pkg, semantics, loadTimeNanos);
            }
          };
    };
  }

  /**
   * Called after {@link com.google.devtools.build.lib.skyframe.PackageFunction} has successfully
   * loaded the given {@link Package}.
   *
   * @param pkg the loaded {@link Package}
   * @param starlarkSemantics are the semantics used to load the package
   * @param loadTimeNanos the wall time, in ns, that it took to load the package. More precisely,
   *     this is the wall time of the call to {@link PackageFactory#createPackageFromAst}. Notably,
   *     this does not include the time to read and parse the package's BUILD file, nor the time to
   *     read, parse, or evaluate any of the transitively loaded .bzl files, and it includes time
   *     the OS thread is runnable but not running.
   */
  void onLoadingCompleteAndSuccessful(
      Package pkg, StarlarkSemantics starlarkSemantics, long loadTimeNanos);
}
