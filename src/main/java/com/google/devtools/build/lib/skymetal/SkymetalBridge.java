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

import static com.google.common.base.Preconditions.checkNotNull;
import static java.lang.classfile.ClassFile.ACC_FINAL;
import static java.lang.classfile.ClassFile.ACC_PUBLIC;
import static java.lang.classfile.ClassFile.ACC_STATIC;
import static java.lang.classfile.ClassFile.ACC_SUPER;

import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.jni.JniLoader;
import com.google.errorprone.annotations.CompileTimeConstant;
import com.google.errorprone.annotations.Keep;
import java.lang.classfile.ClassFile;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.SourceFileAttribute;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Foreign Function and Memory (FFM) facilities for Skymetal.
 *
 * <p>Facilities in this class should be used in preference to the bare {@link Linker} and {@link
 * SymbolLookup} APIs to interact with Skymetal.
 */
public final class SkymetalBridge {
  private static final String DOWNCALL_CLASS_PREFIX =
      "com.google.devtools.build.lib.skymetal.Downcall$";
  private static final String DOWNCALL_SOURCE_FILE = "FFM downcall";

  private static final Linker NATIVE_LINKER = Linker.nativeLinker();
  private static final SymbolLookup SYMBOL_LOOKUP = createLookup();
  private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();
  private static final ClassFile CLASS_FILE = ClassFile.of();
  private static final AtomicInteger NEXT_CLASS_ID = new AtomicInteger();
  private static final MethodHandle HANDLE_UPCALL_EXCEPTION;

  static {
    try {
      HANDLE_UPCALL_EXCEPTION =
          LOOKUP.findStatic(
              SkymetalBridge.class,
              "handleUpcallException",
              MethodType.methodType(Object.class, Throwable.class));
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  /** Returns a {@link SymbolLookup} for symbols in the server's native library. */
  public static SymbolLookup symbolLookup() {
    return SYMBOL_LOOKUP;
  }

  /**
   * Resolves {@code name} in the server's native library and returns a downcall method handle for
   * it.
   *
   * <p>Unlike a regular downcall handle created by {@link Linker#downcallHandle}, which is
   * invisible in a Java stack trace, the returned handle will, upon invocation, add a stack frame
   * with the name of the native method being called.
   *
   * <p>Each call defines a new class that remains loaded for the lifetime of this classloader.
   * Callers should store the returned handle in a {@code static final} field rather than calling
   * this method repeatedly, which both avoids unbounded metaspace growth and enables JIT inlining.
   *
   * @throws UnsatisfiedLinkError if {@code name} cannot be found
   * @throws IllegalArgumentException if {@code name} is not a valid Java method name or if {@link
   *     Linker#downcallHandle} rejects the arguments
   */
  public static MethodHandle downcallHandle(
      @CompileTimeConstant String name, FunctionDescriptor function, Linker.Option... options) {
    validateMethodName(name);
    MemorySegment address =
        SYMBOL_LOOKUP
            .find(name)
            .orElseThrow(() -> new UnsatisfiedLinkError("undefined symbol: " + name));
    MethodHandle raw = NATIVE_LINKER.downcallHandle(address, function, options);
    return wrapInStackFrame(name, raw);
  }

  /**
   * Creates an upcall stub for {@code target} associated with {@code arena}.
   *
   * <p>Unlike a regular upcall stub created by {@link Linker#upcallStub}, where an uncaught
   * exception causes the JVM to abort immediately, the returned stub catches any {@link Throwable}
   * thrown by {@code target} and forwards it to {@link BugReport} as a crash.
   *
   * <p>As with {@link Linker#upcallStub}, {@code target} must not declare any thrown exceptions.
   *
   * @throws IllegalArgumentException if {@link Linker#upcallStub} rejects the arguments (including
   *     if {@code target} declares thrown exceptions)
   * @throws IllegalStateException if {@code arena.scope().isAlive()} is {@code false}
   * @throws WrongThreadException if {@code arena} is a confined arena and this method is called
   *     from a thread other than the arena's owner thread
   */
  public static MemorySegment upcallStub(
      MethodHandle target, FunctionDescriptor function, Arena arena, Linker.Option... options) {
    checkNotNull(target, "target");
    checkNotNull(function, "function");
    checkNotNull(arena, "arena");
    checkNotNull(options, "options");

    // Validate the unwrapped target with the native linker before wrapping it with
    // MethodHandles.catchException, which returns a BoundMethodHandle that hides declared
    // exceptions from Linker#upcallStub's check.
    try (Arena probe = Arena.ofConfined()) {
      var _ = NATIVE_LINKER.upcallStub(target, function, probe, options);
    }

    MethodHandle wrapped = wrapInExceptionHandler(target);
    return NATIVE_LINKER.upcallStub(wrapped, function, arena, options);
  }

  private static SymbolLookup createLookup() {
    JniLoader.loadJni();
    SymbolLookup lookup = SymbolLookup.loaderLookup();
    return lookup;
  }

  private static void validateMethodName(String name) {
    checkNotNull(name, "name");
    if (name.isEmpty()
        || !Character.isJavaIdentifierStart(name.codePointAt(0))
        || !name.codePoints().skip(1).allMatch(Character::isJavaIdentifierPart)) {
      throw new IllegalArgumentException("Invalid Java method name: " + name);
    }
  }

  /**
   * Wraps {@code target} in a dynamically generated trampoline method named {@code methodName} so
   * invocations appear on Java stack traces as {@code
   * com.google.devtools.build.lib.skymetal.Downcall$<id>.<methodName>(FFM downcall)}.
   *
   * <p>Generates and loads a regular (non-hidden) class in this package with a single static method
   * equivalent to:
   *
   * <pre>{@code
   * final class Downcall$<id> {
   *   public static R <methodName>(MethodHandle target, A... args) throws Throwable {
   *     return (R) target.invokeExact(args...);
   *   }
   * }
   * }</pre>
   *
   * <p>The class is defined via {@link MethodHandles.Lookup#defineClass} (rather than a hidden
   * class) so its frame is retained in stack traces and thread dumps, with a unique numeric suffix
   * so multiple handles for the same symbol name do not collide. Passing {@code target} as a
   * leading parameter bound via {@link MethodHandles#insertArguments} avoids static fields while
   * still allowing the JIT to constant-fold {@code target} and inline through the trampoline.
   */
  private static MethodHandle wrapInStackFrame(
      @CompileTimeConstant String methodName, MethodHandle target) {
    MethodType targetType = target.type();
    MethodType trampolineType = targetType.insertParameterTypes(0, MethodHandle.class);
    String className = DOWNCALL_CLASS_PREFIX + NEXT_CLASS_ID.incrementAndGet();
    ClassDesc classDesc = ClassDesc.of(className);
    MethodTypeDesc trampolineDesc = MethodTypeDesc.ofDescriptor(trampolineType.descriptorString());
    MethodTypeDesc invokeExactDesc = MethodTypeDesc.ofDescriptor(targetType.descriptorString());

    byte[] classBytes =
        CLASS_FILE.build(
            classDesc,
            clb ->
                clb.withFlags(ACC_FINAL | ACC_SUPER)
                    .with(SourceFileAttribute.of(DOWNCALL_SOURCE_FILE))
                    .withMethodBody(
                        methodName,
                        trampolineDesc,
                        ACC_PUBLIC | ACC_STATIC,
                        cob -> {
                          for (int i = 0; i < trampolineType.parameterCount(); i++) {
                            cob.loadLocal(
                                TypeKind.from(trampolineType.parameterType(i)),
                                cob.parameterSlot(i));
                          }
                          cob.invokevirtual(
                                  ConstantDescs.CD_MethodHandle, "invokeExact", invokeExactDesc)
                              .return_(TypeKind.from(targetType.returnType()));
                        }));

    try {
      Class<?> defined = LOOKUP.defineClass(classBytes);
      MethodHandle trampoline = LOOKUP.findStatic(defined, methodName, trampolineType);
      return MethodHandles.insertArguments(trampoline, 0, target);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Failed to link downcall trampoline for " + methodName, e);
    }
  }

  /**
   * Wraps {@code target} in a {@link MethodHandles#catchException} combinator that forwards any
   * thrown {@link Throwable} to {@link BugReport}.
   */
  private static MethodHandle wrapInExceptionHandler(MethodHandle target) {
    return MethodHandles.catchException(
        target,
        Throwable.class,
        HANDLE_UPCALL_EXCEPTION.asType(
            MethodType.methodType(target.type().returnType(), Throwable.class)));
  }

  /** Reports an uncaught upcall exception to {@link BugReport} as a halting crash. */
  @Keep
  private static Object handleUpcallException(Throwable t) {
    BugReport.handleCrash(Crash.from(t), CrashContext.halt());
    throw new AssertionError("Should have halted", t);
  }

  private SkymetalBridge() {}
}
