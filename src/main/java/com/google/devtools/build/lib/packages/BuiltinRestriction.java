// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Static utility methods pertaining to restricting Starlark method invocations */
public final class BuiltinRestriction {
  /** Throws if the calling Starlark function is not in the builtins repository */
  public static void throwIfNotBuiltinUsage(StarlarkThread thread) throws EvalException {
    RepositoryName repository =
        BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread))
            .label()
            .getRepository();
    if (!"@_builtins".equals(repository.getName())) {
      throw Starlark.errorf("private API only for use in builtins");
    }
  }

  private BuiltinRestriction() {}
}
