// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import static com.dylibso.chicory.runtime.Memory.PAGE_SIZE;
import static com.dylibso.chicory.wasm.types.MemoryLimits.MAX_PAGES;
import static com.google.devtools.build.lib.profiler.ProfilerTask.WASM_EXEC;
import static com.google.devtools.build.lib.profiler.ProfilerTask.WASM_LOAD;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.dylibso.chicory.runtime.ByteArrayMemory;
import com.dylibso.chicory.runtime.ExportFunction;
import com.dylibso.chicory.runtime.Instance;
import com.dylibso.chicory.wasm.ChicoryException;
import com.dylibso.chicory.wasm.WasmModule;
import com.dylibso.chicory.wasm.types.MemoryLimits;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.time.Duration;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkValue;

@Immutable
@StarlarkBuiltin(
    name = "wasm_module",
    category = DocCategory.BUILTIN,
    doc = "A WebAssembly module loaded by <code>repository_ctx.load_wasm()</code>.")
final class StarlarkWasmModule implements StarlarkValue {
  private final StarlarkPath path;
  private final Object origPath;
  private final WasmModule wasmModule;
  private final String allocFnName;
  private final boolean hasInitializeFn;

  public StarlarkWasmModule(
      StarlarkPath path, Object origPath, byte[] moduleContent, String allocFnName)
      throws EvalException {
    WasmModule wasmModule;
    Profiler prof = Profiler.instance();
    try (SilentCloseable c1 = prof.profile(WASM_LOAD, () -> "load " + path.toString())) {
      try (SilentCloseable c2 = prof.profile(WASM_LOAD, "parse")) {
        try {
          wasmModule = com.dylibso.chicory.wasm.Parser.parse(moduleContent);
        } catch (ChicoryException e) {
          throw new EvalException(e);
        }
      }
      validateModule(wasmModule, allocFnName);
    }

    this.path = path;
    this.origPath = origPath;
    this.wasmModule = wasmModule;
    this.allocFnName = allocFnName;
    this.hasInitializeFn = hasInitializeFn(wasmModule);
  }

  private static boolean hasInitializeFn(WasmModule wasmModule) {
    var exports = wasmModule.exportSection();
    int exportCount = exports.exportCount();
    for (int ii = 0; ii < exportCount; ii++) {
      if (exports.getExport(ii).name().equals("_initialize")) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<wasm_module path=");
    printer.repr(origPath);
    printer.append(" allocate_fn=");
    printer.repr(allocFnName);
    printer.append(">");
  }

  public StarlarkPath getPath() {
    return path;
  }

  @StarlarkMethod(
      name = "path",
      structField = true,
      doc = "The path this WebAssembly module was loaded from.")
  public Object getOrigPath() {
    return origPath;
  }

  public StarlarkWasmExecutionResult execute(
      String execFnName, byte[] input, Duration timeout, long memLimitBytes)
      throws EvalException, InterruptedException {
    Profiler prof = Profiler.instance();
    try (SilentCloseable c = prof.profile(WASM_EXEC, () -> "execute " + execFnName)) {
      var memLimits = getMemLimits(memLimitBytes);
      // Perform initialization and execution in a separate thread so it can be interrupted
      // in case of timeout.
      var wasmThreadFactory =
          Thread.ofPlatform().name(Thread.currentThread().getName() + "_wasm").factory();
      StarlarkWasmExecutionResult result;
      String errMessage;
      try (var executor = Executors.newSingleThreadExecutor(wasmThreadFactory)) {
        return executor.invokeAny(
            ImmutableList.of(() -> run(execFnName, input, memLimits)),
            timeout.toMillis(),
            TimeUnit.MILLISECONDS);
      } catch (TimeoutException e) {
        errMessage = String.format("Error executing %s: timed out", execFnName);
      } catch (ExecutionException e) {
        errMessage = String.format("Error executing %s: %s", execFnName, e.getCause().getMessage());
      }
      return StarlarkWasmExecutionResult.newErr(errMessage);
    }
  }

  private StarlarkWasmExecutionResult run(String execFnName, byte[] input, MemoryLimits memLimits)
      throws EvalException, InterruptedException {
    Instance instance;
    Profiler prof = Profiler.instance();
    try {
      instance =
          Instance.builder(wasmModule)
              .withMemoryLimits(memLimits)
              // Disable calling `_start()`, which is the entry point for WASI-style
              // command modules.
              .withStart(false)
              // Chicory documentation recommends ByteArrayMemory for OpenJDK
              // https://chicory.dev/docs/advanced/memory
              .withMemoryFactory(ByteArrayMemory::new)
              .build();
      // If `_initialize()` is present then call it to perform early setup.
      //
      // Note: The WebAssembly spec describes a "start function", named in a
      // "start section", that is to be called as part of module initialization.
      // Actual implementations such as LLVM have instead used the start function
      // as the equivalent of a native binary's entry point, and expect (or emit)
      // a function named `_initialize` to be used for early initialization.
      //
      // For additional context, see:
      // - https://bugs.llvm.org/show_bug.cgi?id=37198
      // - https://reviews.llvm.org/D40559
      // - https://github.com/WebAssembly/design/issues/1160
      if (hasInitializeFn) {
        try (SilentCloseable c = prof.profile(WASM_EXEC, "initialize")) {
          instance.export("_initialize").apply();
        }
      }
    } catch (ChicoryException e) {
      throw new EvalException(e);
    }

    var memory = instance.memory();
    ExportFunction allocFn = instance.export(allocFnName);
    // TODO: #26092 - Is this check needed? Might be redundant with validateModule().
    if (allocFn == null) {
      throw Starlark.errorf("WebAssembly module doesn't export \"%s\"", allocFnName);
    }
    ExportFunction execFn = instance.export(execFnName);
    // TODO: #26092 - Validate execFn has the expected signature?
    if (execFn == null) {
      throw Starlark.errorf("WebAssembly module doesn't export \"%s\"", execFnName);
    }

    int inputLen = Math.toIntExact(input.length);
    int inputPtr = alloc(allocFnName, allocFn, inputLen, 1);
    try (SilentCloseable c = prof.profile(WASM_EXEC, "copy input")) {
      memory.write(inputPtr, input);
    }

    // struct { output_ptr_ptr: **u8, output_len_ptr: *u32 }
    int paramsPtr = alloc(allocFnName, allocFn, 8, 4);
    int outputPtrPtr = paramsPtr;
    int outputLenPtr = paramsPtr + 4;
    memory.writeI32(outputPtrPtr, 0);
    memory.writeI32(outputLenPtr, 0);

    long[] execResult;
    try (SilentCloseable c = prof.profile(WASM_EXEC, "execute")) {
      execResult = execFn.apply(inputPtr, inputLen, outputPtrPtr, outputLenPtr);
    }

    // TODO: #26092 - Not 100% sure this check is necessary, but the ambiguity between
    // signed/unsigned in Java vs WebAssembly makes me nervous.
    //
    // Might be unnecessary if the function signature is verified before execution?
    long returnCode = execResult[0];
    if (returnCode < 0 || returnCode > 0xFFFFFFFFL) {
      returnCode = 0xFFFFFFFFL;
    }
    int outputPtr = memory.readInt(outputPtrPtr);
    int outputLen = memory.readInt(outputLenPtr);

    String output = "";
    if (outputLen > 0) {
      try (SilentCloseable c = prof.profile(WASM_EXEC, "copy output")) {
        byte[] outputBytes = memory.readBytes(outputPtr, outputLen);
        output = new String(outputBytes, ISO_8859_1);
      }
    }
    return StarlarkWasmExecutionResult.newOk(returnCode, output);
  }

  private static void validateModule(WasmModule wasmModule, String allocFnName)
      throws EvalException {
    var exports = wasmModule.exportSection();
    int exportCount = exports.exportCount();
    for (int ii = 0; ii < exportCount; ii++) {
      var export = exports.getExport(ii);
      if (export.name().equals(allocFnName)) {
        // TODO: #26092 - Validate exported type is a function and has the expected signature?
        return;
      }
    }
    throw Starlark.errorf("WebAssembly module doesn't contain an export named \"%s\"", allocFnName);
  }

  MemoryLimits getMemLimits(long memLimitBytes) throws EvalException {
    int initialPages = 1;
    int memLimitPages = getMemLimitPages(memLimitBytes);

    if (wasmModule.memorySection().isPresent()) {
      var memories = wasmModule.memorySection().get();
      int memoryCount = memories.memoryCount();
      if (memoryCount > 1) {
        // TODO: #26092 - Figure out what memory limits mean when applied to
        // a WebAssembly module with multiple memories.
        throw Starlark.errorf("WebAssembly modules with multiple memories not yet supported");
      }
      if (memoryCount != 0) {
        MemoryLimits limits = memories.getMemory(0).limits();
        if (limits.initialPages() > initialPages) {
          initialPages = limits.initialPages();
        }
      }
    }
    if (initialPages > memLimitPages) {
      // TODO: #26092 - Should probably throw an exception. The execution will likely fail anyway,
      // and
      // throwing an exception from this point would provide more relevant details.
      initialPages = memLimitPages;
    }
    return new MemoryLimits(initialPages, memLimitPages);
  }

  static int getMemLimitPages(long memLimitBytes) {
    if (memLimitBytes == 0) {
      return 1;
    }
    return (int) Math.min((long) MAX_PAGES, Math.ceilDiv(memLimitBytes, PAGE_SIZE));
  }

  static int alloc(String allocFnName, ExportFunction allocFn, int size, int align)
      throws ChicoryException, EvalException {
    long[] allocResult = allocFn.apply(size, align);
    long ptr = allocResult[0];
    if (ptr == 0) {
      throw Starlark.errorf(
          "allocation failed: %s(%d, %d) returned NULL", allocFnName, size, align);
    }
    try {
      return Math.toIntExact(ptr);
    } catch (ArithmeticException e) {
      throw Starlark.errorf(
          "allocation failed: %s(%d, %d) returned invalid pointer 0x%08X (out of range)",
          allocFnName, size, align, ptr);
    }
  }
}
