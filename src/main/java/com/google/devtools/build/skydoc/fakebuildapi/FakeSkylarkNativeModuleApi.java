// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkNativeModuleApi;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkCallable;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import javax.annotation.Nullable;

/** Fake implementation of {@link SkylarkNativeModuleApi}. */
public class FakeSkylarkNativeModuleApi implements SkylarkNativeModuleApi, ClassObject {

  @Override
  public Sequence<?> glob(
      Sequence<?> include,
      Sequence<?> exclude,
      Integer excludeDirectories,
      Object allowEmpty,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    return StarlarkList.of(thread.mutability());
  }

  @Override
  public Object existingRule(String name, StarlarkThread thread) {
    return null;
  }

  @Override
  public Dict<String, Dict<String, Object>> existingRules(StarlarkThread thread) {
    return Dict.of(thread.mutability());
  }

  @Override
  public NoneType packageGroup(
      String name, Sequence<?> packages, Sequence<?> includes, StarlarkThread thread) {
    return null;
  }

  @Override
  public NoneType exportsFiles(
      Sequence<?> srcs, Object visibility, Object licenses, StarlarkThread thread) {
    return null;
  }

  @Override
  public String packageName(StarlarkThread thread) {
    return "";
  }

  @Override
  public String repositoryName(StarlarkThread thread) {
    return "";
  }

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    // Bazel's notion of the global "native" isn't fully exposed via public interfaces, for example,
    // as far as native rules are concerned. Returning None on all unsupported invocations of
    // native.[func_name]() is the safest "best effort" approach to implementing a fake for
    // "native".
    return new StarlarkCallable() {
      @Override
      public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
        return Starlark.NONE;
      }

      @Override
      public String getName() {
        return name;
      }

      @Override
      public Location getLocation() {
        return Location.BUILTIN;
      }

      @Override
      public void repr(Printer printer) {
        printer.append("<faked no-op function " + name + ">");
      }
    };
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public String getErrorMessageForUnknownField(String field) {
    return "";
  }
}
