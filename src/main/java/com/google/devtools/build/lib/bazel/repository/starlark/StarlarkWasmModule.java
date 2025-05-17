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
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.dylibso.chicory.runtime.ByteArrayMemory;
import com.dylibso.chicory.runtime.ExportFunction;
import com.dylibso.chicory.runtime.Instance;
import com.dylibso.chicory.wasm.ChicoryException;
import com.dylibso.chicory.wasm.types.MemoryLimits;
import com.dylibso.chicory.wasm.WasmModule;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.time.Duration;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
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
  private final String allocateFn;

  public StarlarkWasmModule(
      StarlarkPath path,
      Object origPath,
      byte[] moduleContent,
      String allocateFn)
      throws EvalException {
    WasmModule wasmModule;
    try {
      wasmModule = com.dylibso.chicory.wasm.Parser.parse(moduleContent);
    } catch (ChicoryException e) {
      throw new EvalException(e);
    }
    validateModule(wasmModule, allocateFn);

    this.path = path;
    this.origPath = origPath;
    this.wasmModule = wasmModule;
    this.allocateFn = allocateFn;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<wasm_module path=");
    printer.repr(origPath);
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
      String functionName,
      byte[] input,
      Duration timeout,
      long memLimitBytes)
      throws EvalException, InterruptedException {
    var memLimits = getMemLimits(memLimitBytes);
    // Perform initialization and execution in a separate thread so it can be interrupted
    // in case of timeout.
    ExecutorService execService = Executors.newSingleThreadExecutor();
    var execResultFuture = execService.submit(() -> {
      return run(functionName, input, memLimits);
    });
    execService.shutdown();

    StarlarkWasmExecutionResult result;
    try {
      result = execResultFuture.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
    } catch (TimeoutException _e) {
      execService.shutdownNow();
      while (!execService.isTerminated()) {
        execService.awaitTermination(1, TimeUnit.SECONDS);
      }
      // Timeouts result in a `wasm_exec_result` with negative `return_code`.
      result = new StarlarkWasmExecutionResult(-1, "");
    } catch (ExecutionException e) {
      // FIXME: This branch will be entered if out-of-memory occurs. Should such
      // a condition be an exception, or act like a timeout and return a result
      // with a negative `return_code`?
      throw new EvalException(e.getCause());
    }
    return result;
  }

  private StarlarkWasmExecutionResult run(String execFunc, byte[] input, MemoryLimits memLimits)
      throws EvalException, InterruptedException {
    Instance instance;
    try {
      instance = Instance.builder(wasmModule)
          .withMemoryLimits(memLimits)
          // Disable calling `_start()`, which is the entry point for WASI-style
          // command modules.
          .withStart(false)
          // Chicory documentation recommends ByteArrayMemory for OpenJDK
          // https://chicory.dev/docs/advanced/memory
          .withMemoryFactory(limits -> { return new ByteArrayMemory(limits); })
          .build();
      // If `_initialize()` is present then call it to perform early setup.
      ExportFunction initFn = instance.export("_initialize");
      if (initFn != null) {
        // FIXME: Include this in the timeout-guarded section.
        initFn.apply();
      }
    } catch (ChicoryException e) {
      throw new EvalException(e);
    }

    var memory = instance.memory();
    ExportFunction allocFn = instance.export(allocateFn);
    // FIXME: Is this check needed? Might be redundant with validateModule().
    if (allocFn == null) {
      throw Starlark.errorf("WebAssembly module doesn't export \"%s\"", allocateFn);
    }
    ExportFunction execFn = instance.export(execFunc);
    // FIXME: Validate execFn has the expected signature?
    if (execFn == null) {
      throw Starlark.errorf("WebAssembly module doesn't export \"%s\"", execFunc);
    }

    int inputLen = Math.toIntExact(input.length);
    int inputPtr = alloc(allocateFn, allocFn, inputLen, 1);
    memory.write(inputPtr, input);

    // struct { output_ptr_ptr: **u8, output_len_ptr: *u32 }
    int paramsPtr = alloc(allocateFn, allocFn, 8, 4);
    int outputPtrPtr = paramsPtr;
    int outputLenPtr = paramsPtr + 4;
    memory.writeI32(outputPtrPtr, 0);
    memory.writeI32(outputLenPtr, 0);

    long[] execResult = execFn.apply(inputPtr, inputLen, outputPtrPtr, outputLenPtr);

    // FIXME: Not 100% sure this check is necessary, but the ambiguity between
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
      byte[] outputBytes = memory.readBytes(outputPtr, outputLen);
      output = new String(outputBytes, ISO_8859_1);
    }
    return new StarlarkWasmExecutionResult(returnCode, output);
  }

  private static void validateModule(WasmModule wasmModule, String allocateFn) throws EvalException {
    var exports = wasmModule.exportSection();
    int exportCount = exports.exportCount();
    for (int ii = 0; ii < exportCount; ii++) {
      var export = exports.getExport(ii);
      if (export.name().equals(allocateFn)) {
        // FIXME: Validate exported type is a function and has the expected signature?
        return;
      }
    }
    throw Starlark.errorf("WebAssembly module doesn't contain an export named \"%s\"", allocateFn);
  }

  MemoryLimits getMemLimits(long memLimitBytes) {
    int initialPages = 1;
    int memLimitPages = getMemLimitPages(memLimitBytes);

    if (wasmModule.memorySection().isPresent()) {
      var memories = wasmModule.memorySection().get();
      int memoryCount = memories.memoryCount();
      for (int ii = 0; ii < memoryCount; ii++) {
        MemoryLimits limits = memories.getMemory(ii).limits();
        if (limits.initialPages() > initialPages) {
          initialPages = limits.initialPages();
        }
      }
    }
    if (initialPages > memLimitPages) {
      // FIXME: Should probably throw an exception. The execution will likely fail anyway, and
      // throwing an exception from this point would provide more relevant details.
      initialPages = memLimitPages;
    }
    return new MemoryLimits(initialPages, memLimitPages);
  }

  static int getMemLimitPages(long memLimitBytes) {
    if (memLimitBytes <= (long)PAGE_SIZE) {
      return 1;
    }
    long memLimitPagesL = memLimitBytes / (long)PAGE_SIZE;
    if (memLimitPagesL >= (long)MAX_PAGES) {
      return MAX_PAGES;
    }
    int memLimitPages = (int)memLimitPagesL;
    if (memLimitBytes % PAGE_SIZE != 0) {
      memLimitPages += 1;
    }
    return memLimitPages;
  }

  static int alloc(String allocFnName, ExportFunction allocFn, int size, int align)
      throws ChicoryException, EvalException {
    long[] allocResult = allocFn.apply(size, align);
    long ptr = allocResult[0];
    if (ptr == 0) {
      throw Starlark.errorf(
        "allocation failed: %s(%d, %d) returned NULL", allocFnName, size, align);
    }
    int intPtr = (int)ptr;
    if (intPtr != ptr) {
      throw Starlark.errorf(
        "allocation failed: %s(%d, %d) returned invalid pointer 0x%08X (out of range)",
        allocFnName, size, align, ptr);
    }
    return intPtr;
  }
}
