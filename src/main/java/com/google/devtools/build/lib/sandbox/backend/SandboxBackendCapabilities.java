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
package com.google.devtools.build.lib.sandbox.backend;

import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.ExecutionCapabilities;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.devtools.build.lib.remote.ApiVersion;
import com.google.devtools.build.lib.remote.ClientApiVersion;
import java.util.List;
import javax.annotation.Nullable;

/** Checks that what a sandbox backend says it can do is what Bazel needs from it. */
final class SandboxBackendCapabilities {
  private SandboxBackendCapabilities() {}

  /**
   * Returns why the backend can't be used, or null if it can.
   *
   * <p>A backend has to declare that it executes, which digest functions it handles and which
   * versions of the API it speaks. A backend that doesn't implement {@code GetCapabilities} at all
   * is rejected by the caller as well: nothing is assumed about it, so that a backend that can't
   * tell what it supports fails at startup with a clear message instead of in the middle of a
   * build.
   */
  @Nullable
  static String check(
      String name, ServerCapabilities capabilities, DigestFunction.Value digestFunction) {
    ClientApiVersion client = ClientApiVersion.current;
    ApiVersion low = new ApiVersion(capabilities.getLowApiVersion());
    ApiVersion high = new ApiVersion(capabilities.getHighApiVersion());
    // The ranges have to overlap.
    if (low.compareTo(client.getHigh()) > 0 || high.compareTo(client.getLow()) < 0) {
      return String.format(
          "sandbox backend '%s' supports Remote Execution API versions %s to %s, but Bazel"
              + " needs a version between %s and %s",
          name, low, high, client.getLow(), client.getHigh());
    }
    ExecutionCapabilities execution = capabilities.getExecutionCapabilities();
    if (!execution.getExecEnabled()) {
      return String.format(
          "sandbox backend '%s' does not enable execution (execution_capabilities.exec_enabled is"
              + " false)",
          name);
    }
    // The singular field is what older servers set, the list was added later.
    if (!supports(
        execution.getDigestFunctionsList(), execution.getDigestFunction(), digestFunction)) {
      return unsupportedDigestFunction(name, digestFunction);
    }
    // The backend also serves the CAS. It may not say anything about it, but if it does it must
    // include the digest function.
    CacheCapabilities cache = capabilities.getCacheCapabilities();
    if (!cache.getDigestFunctionsList().isEmpty()
        && !cache.getDigestFunctionsList().contains(digestFunction)) {
      return unsupportedDigestFunction(name, digestFunction);
    }
    return null;
  }

  private static boolean supports(
      List<DigestFunction.Value> functions,
      DigestFunction.Value single,
      DigestFunction.Value wanted) {
    if (!functions.isEmpty()) {
      return functions.contains(wanted);
    }
    // Nothing declared means SHA-256, as in the API.
    return (single == DigestFunction.Value.UNKNOWN ? DigestFunction.Value.SHA256 : single)
        == wanted;
  }

  private static String unsupportedDigestFunction(String name, DigestFunction.Value function) {
    return String.format("sandbox backend '%s' does not support digest function %s", name, function);
  }
}
