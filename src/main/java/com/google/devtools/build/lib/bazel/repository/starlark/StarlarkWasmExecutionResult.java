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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

@Immutable
@StarlarkBuiltin(
    name = "wasm_exec_result",
    category = DocCategory.BUILTIN,
    doc =
        """
        The result of executing a WebAssembly function with
        <code>repository_ctx.execute_wasm()</code>. It contains the function's
        return value and output buffer.

        <p>If execution failed before the function returned then the return code will be negative
        and the <code>error_message</code> field will be set.
        """)
final class StarlarkWasmExecutionResult implements StarlarkValue {
  private final long returnCode;
  private final String output;
  private final String errorMessage;

  private StarlarkWasmExecutionResult(long returnCode, String output, String errorMessage) {
    this.returnCode = returnCode;
    this.output = output;
    this.errorMessage = errorMessage;
  }

  public static StarlarkWasmExecutionResult newOk(long returnCode, String output) {
    return new StarlarkWasmExecutionResult(returnCode, output, "");
  }

  public static StarlarkWasmExecutionResult newErr(String errorMessage) {
    return new StarlarkWasmExecutionResult(-1, "", errorMessage);
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<wasm_exec_result return_code=");
    printer.repr(returnCode);
    printer.append(" output=");
    printer.repr(output);
    printer.append(" error_message=");
    printer.repr(errorMessage);
    printer.append(">");
  }

  @StarlarkMethod(
      name = "return_code",
      structField = true,
      doc =
          """
          The return value of the WebAssembly function, or a negative value if execution
          was terminated before the function returned.
          """)
  public long getReturnCode() {
    return returnCode;
  }

  @StarlarkMethod(
      name = "output",
      structField = true,
      doc = "The content of the output buffer returned by the WebAssembly function.")
  public String getOutput() {
    return output;
  }

  @StarlarkMethod(
      name = "error_message",
      structField = true,
      doc = "Contains an error message if execution failed before the function returned.")
  public String getErrorMessage() {
    return errorMessage;
  }
}
