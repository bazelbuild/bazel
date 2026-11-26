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

import static com.google.common.truth.Truth.assertThat;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SkymetalBridgeTest {
  @BeforeClass
  public static void loadNativeLibrary() throws IOException {
    String libPath = Runfiles.preload().unmapped().rlocation(System.getProperty("native.test.lib"));
    System.load(Path.of(libPath).toAbsolutePath().toString());
  }

  @Test
  public void downcallHandle_appearsInStackTrace() throws Throwable {
    MethodHandle invokeCallback =
        SkymetalBridge.downcallHandle(
            "bazel_test_invoke_callback", FunctionDescriptor.ofVoid(ADDRESS));
    AtomicReference<StackTraceElement[]> captured = new AtomicReference<>();

    try (Arena arena = Arena.ofConfined()) {
      MethodHandle callback =
          MethodHandles.lookup()
              .bind(
                  (Runnable) () -> captured.set(new Exception().getStackTrace()),
                  "run",
                  MethodType.methodType(void.class));
      MemorySegment stub =
          Linker.nativeLinker().upcallStub(callback, FunctionDescriptor.ofVoid(), arena);
      invokeCallback.invokeExact(stub);
    }

    StackTraceElement[] trace = captured.get();
    assertThat(trace.length).isAtLeast(3);

    assertThat(trace[0].getClassName()).isEqualTo(SkymetalBridgeTest.class.getName());

    assertThat(trace[1].getClassName())
        .startsWith("com.google.devtools.build.lib.skymetal.Downcall$");
    assertThat(trace[1].getMethodName()).isEqualTo("bazel_test_invoke_callback");
    assertThat(trace[1].getFileName()).isEqualTo("FFM downcall");
    assertThat(trace[1].getLineNumber()).isEqualTo(-1);
    assertThat(trace[1].isNativeMethod()).isFalse();
    assertThat(trace[1].toString())
        .matches(
            "com\\.google\\.devtools\\.build\\.lib\\.skymetal\\.Downcall\\$\\d+"
                + "\\.bazel_test_invoke_callback\\(FFM downcall\\)");

    assertThat(trace[2].getClassName()).isEqualTo(SkymetalBridgeTest.class.getName());
    assertThat(trace[2].getMethodName()).isEqualTo("downcallHandle_appearsInStackTrace");
  }

  @Test
  public void downcallHandle_mixedTypesAndOptions() throws Throwable {
    // critical(true) is required to pass a heap-backed MemorySegment, verifying option forwarding.
    MethodHandle mixedTypes =
        SkymetalBridge.downcallHandle(
            "bazel_test_mixed_types",
            FunctionDescriptor.of(JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_DOUBLE, ADDRESS),
            Linker.Option.critical(/* allowHeapAccess= */ true));

    long sum = (long) mixedTypes.invokeExact(1, 2L, 3.0d, MemorySegment.ofArray(new int[] {40}));
    assertThat(sum).isEqualTo(46L);
  }

  @Test
  public void downcallHandle_duplicateSymbolNames_doNotCollide() throws Throwable {
    FunctionDescriptor desc =
        FunctionDescriptor.of(JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_DOUBLE, ADDRESS);
    Linker.Option critical = Linker.Option.critical(/* allowHeapAccess= */ true);
    MethodHandle first = SkymetalBridge.downcallHandle("bazel_test_mixed_types", desc, critical);
    MethodHandle second = SkymetalBridge.downcallHandle("bazel_test_mixed_types", desc, critical);

    MemorySegment cell = MemorySegment.ofArray(new int[] {40});
    assertThat((long) first.invokeExact(1, 2L, 3.0d, cell)).isEqualTo(46L);
    assertThat((long) second.invokeExact(10, 20L, 30.0d, cell)).isEqualTo(100L);
  }

  @Test
  public void downcallHandle_invalidInputs_throwExpectedExceptions() {
    FunctionDescriptor desc = FunctionDescriptor.ofVoid();

    assertThrows(NullPointerException.class, () -> SkymetalBridge.downcallHandle(null, desc));
    assertThrows(IllegalArgumentException.class, () -> SkymetalBridge.downcallHandle("", desc));
    assertThrows(
        IllegalArgumentException.class, () -> SkymetalBridge.downcallHandle("123bad", desc));
    assertThrows(IllegalArgumentException.class, () -> SkymetalBridge.downcallHandle("a.b", desc));
    assertThrows(
        UnsatisfiedLinkError.class,
        () -> SkymetalBridge.downcallHandle("bazel_test_nonexistent_symbol", desc));
  }
}
