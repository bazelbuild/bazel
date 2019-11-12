// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkCallable;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * "Fake" implementation of {@link StarlarkCallable} which accepts any parameters and always returns
 * None.
 */
public final class FakeStarlarkCallable implements StarlarkCallable {

  private final String functionName;

  public FakeStarlarkCallable(String functionName) {
    this.functionName = functionName;
  }

  @Override
  public Object call(
      List<Object> args,
      @Nullable Map<String, Object> kwargs,
      @Nullable FuncallExpression call,
      StarlarkThread thread) {
    return Starlark.NONE;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<faked no-op function " + functionName + ">");
  }
}
