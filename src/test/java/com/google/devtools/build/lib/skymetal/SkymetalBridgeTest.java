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

import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.runfiles.Runfiles;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.ExtensionRegistry;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SkymetalBridgeTest {
  private static final FunctionDescriptor MIXED_CALLBACK_DESC =
      FunctionDescriptor.of(
          JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_DOUBLE, ADDRESS.withTargetLayout(JAVA_INT));

  private static final MethodHandle NO_OP;
  private static final MethodHandle MIXED_TYPES_CALLBACK;
  private static final MethodHandle THROWS_CHECKED_EXCEPTION;
  private static final MethodHandle THROWS_DECLARED_RUNTIME_EXCEPTION;

  static {
    try {
      MethodHandles.Lookup lookup = MethodHandles.lookup();
      NO_OP =
          lookup.findStatic(SkymetalBridgeTest.class, "noOp", MethodType.methodType(void.class));
      MIXED_TYPES_CALLBACK =
          lookup.findStatic(
              SkymetalBridgeTest.class, "mixedTypesCallback", MIXED_CALLBACK_DESC.toMethodType());
      THROWS_CHECKED_EXCEPTION =
          lookup.findStatic(
              SkymetalBridgeTest.class,
              "throwsCheckedException",
              MethodType.methodType(void.class));
      THROWS_DECLARED_RUNTIME_EXCEPTION =
          lookup.findStatic(
              SkymetalBridgeTest.class,
              "throwsDeclaredRuntimeException",
              MethodType.methodType(void.class));
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

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

  @Test
  public void upcallStub_invokesTarget() throws Throwable {
    MethodHandle invokeMixed =
        SkymetalBridge.downcallHandle(
            "bazel_test_invoke_mixed_types_callback",
            FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_INT, JAVA_LONG, JAVA_DOUBLE, ADDRESS));

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment stub =
          SkymetalBridge.upcallStub(MIXED_TYPES_CALLBACK, MIXED_CALLBACK_DESC, arena);
      MemorySegment cell = arena.allocateFrom(JAVA_INT, 40);
      assertThat((long) invokeMixed.invokeExact(stub, 1, 2L, 3.0d, cell)).isEqualTo(46L);
    }
  }

  @Test
  public void upcallStub_uncaughtException_forwardsToBugReport() throws Exception {
    int expectedExitCode = ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode();
    assertThat(runCrashSubprocess())
        .isEqualTo(
            new CrashResult(
                expectedExitCode,
                String.valueOf(expectedExitCode),
                Code.CRASH_UNKNOWN,
                IllegalStateException.class.getName(),
                "upcall failure"));
  }

  @Test
  public void upcallStub_declaredException_throwsIllegalArgumentException() {
    FunctionDescriptor voidDesc = FunctionDescriptor.ofVoid();
    try (Arena arena = Arena.ofConfined()) {
      assertThrows(
          IllegalArgumentException.class,
          () -> SkymetalBridge.upcallStub(THROWS_CHECKED_EXCEPTION, voidDesc, arena));
      assertThrows(
          IllegalArgumentException.class,
          () -> SkymetalBridge.upcallStub(THROWS_DECLARED_RUNTIME_EXCEPTION, voidDesc, arena));
    }
  }

  @Test
  public void upcallStub_nullArguments_throwsNullPointerException() {
    FunctionDescriptor voidDesc = FunctionDescriptor.ofVoid();
    try (Arena arena = Arena.ofConfined()) {
      assertThrows(
          NullPointerException.class, () -> SkymetalBridge.upcallStub(null, voidDesc, arena));
      assertThrows(NullPointerException.class, () -> SkymetalBridge.upcallStub(NO_OP, null, arena));
      assertThrows(
          NullPointerException.class, () -> SkymetalBridge.upcallStub(NO_OP, voidDesc, null));
      assertThrows(
          NullPointerException.class,
          () -> SkymetalBridge.upcallStub(NO_OP, voidDesc, arena, (Linker.Option[]) null));
    }
  }

  @Test
  public void upcallStub_signatureMismatch_throwsIllegalArgumentException() {
    FunctionDescriptor intDesc = FunctionDescriptor.of(JAVA_INT);
    try (Arena arena = Arena.ofConfined()) {
      assertThrows(
          IllegalArgumentException.class, () -> SkymetalBridge.upcallStub(NO_OP, intDesc, arena));
    }
  }

  @Test
  public void upcallStub_invalidOption_throwsIllegalArgumentException() {
    FunctionDescriptor voidDesc = FunctionDescriptor.ofVoid();
    Linker.Option criticalOption = Linker.Option.critical(false);
    try (Arena arena = Arena.ofConfined()) {
      assertThrows(
          IllegalArgumentException.class,
          () -> SkymetalBridge.upcallStub(NO_OP, voidDesc, arena, criticalOption));
    }
  }

  @Test
  public void upcallStub_closedArena_throwsIllegalStateException() {
    FunctionDescriptor voidDesc = FunctionDescriptor.ofVoid();
    Arena closed;
    try (Arena arena = Arena.ofConfined()) {
      closed = arena;
    }
    assertThrows(
        IllegalStateException.class, () -> SkymetalBridge.upcallStub(NO_OP, voidDesc, closed));
  }

  @Keep
  private static void noOp() {}

  @Keep
  private static long mixedTypesCallback(int i, long j, double d, MemorySegment p) {
    return i + j + (long) d + p.get(JAVA_INT, 0);
  }

  @Keep
  private static void throwsCheckedException() throws IOException {}

  @Keep
  @SuppressWarnings("ThrowsUncheckedException")
  private static void throwsDeclaredRuntimeException() throws IllegalStateException {}

  private record CrashResult(
      int exitCode,
      String exitCodeFileText,
      Code crashCode,
      String causeClass,
      String causeMessage) {}

  private CrashResult runCrashSubprocess() throws Exception {
    Path dir = tempFolder.newFolder("upcall_crash").toPath();
    Path failureDetailFile = dir.resolve("failure_detail");
    String helperPath =
        Runfiles.preload().unmapped().rlocation(System.getProperty("upcall.test.helper"));

    ProcessBuilder pb =
        new ProcessBuilder(
            Path.of(helperPath).toAbsolutePath().toString(),
            dir.toString(),
            failureDetailFile.toString());
    // Set BAZEL_SHELL_TEST while keeping TEST_TMPDIR set so TestType is SHELL_INTEGRATION:
    // BugReport actually calls Runtime.halt with the crash exit code while still skipping remote
    // bug-report RPCs.
    pb.environment().put("BAZEL_SHELL_TEST", "1");
    pb.inheritIO();

    int exitCode = pb.start().waitFor();
    String exitCodeFileText = Files.readString(dir.resolve("exit_code_to_use_on_abrupt_exit"));
    FailureDetail failureDetail =
        FailureDetail.parseFrom(
            Files.readAllBytes(failureDetailFile), ExtensionRegistry.getEmptyRegistry());
    return new CrashResult(
        exitCode,
        exitCodeFileText,
        failureDetail.getCrash().getCode(),
        failureDetail.getCrash().getCauses(0).getThrowableClass(),
        failureDetail.getCrash().getCauses(0).getMessage());
  }
}
