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

package com.google.devtools.build.lib.skymetal;

import static java.lang.foreign.ValueLayout.ADDRESS;

import com.google.devtools.build.lib.util.CustomExitCodePublisher;
import com.google.devtools.build.lib.util.CustomFailureDetailPublisher;
import com.google.devtools.build.runfiles.Runfiles;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.nio.file.Path;

/** Subprocess helper for {@link SkymetalBridgeTest} that throws inside an FFM upcall stub. */
public final class UpcallTestHelper {
  public static void main(String[] args) throws Throwable {
    CustomExitCodePublisher.setAbruptExitStatusFileDir(args[0]);
    CustomFailureDetailPublisher.setFailureDetailFilePath(args[1]);
    String libPath = Runfiles.preload().unmapped().rlocation(System.getProperty("native.test.lib"));
    System.load(Path.of(libPath).toAbsolutePath().toString());

    MethodHandle invokeCallback =
        SkymetalBridge.downcallHandle(
            "bazel_test_invoke_callback", FunctionDescriptor.ofVoid(ADDRESS));
    MethodHandle throwFailure =
        MethodHandles.insertArguments(
            MethodHandles.throwException(void.class, IllegalStateException.class),
            0,
            new IllegalStateException("upcall failure"));

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment stub =
          SkymetalBridge.upcallStub(throwFailure, FunctionDescriptor.ofVoid(), arena);
      invokeCallback.invokeExact(stub);
    }
  }

  private UpcallTestHelper() {}
}
