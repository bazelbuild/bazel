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

package net.starlark.java.annot.processor.testsources;

import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkLibrary;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Test source file verifying that inheriting one annotation while holding the other fails. */
public class StarlarkBuiltinAndLibraryInherited {
  @StarlarkBuiltin(name = "base", documented = false)
  public static class Base implements StarlarkValue {}

  @StarlarkLibrary
  public static class Sub extends Base {
    @StarlarkMethod(name = "foo", documented = false)
    public String foo() {
      return "foo";
    }
  }
}
