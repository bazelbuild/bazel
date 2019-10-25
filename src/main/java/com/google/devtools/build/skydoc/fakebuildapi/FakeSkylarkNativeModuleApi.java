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
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import javax.annotation.Nullable;

/** Fake implementation of {@link SkylarkNativeModuleApi}. */
public class FakeSkylarkNativeModuleApi implements SkylarkNativeModuleApi, ClassObject {

  @Override
  public SkylarkList<?> glob(
      SkylarkList<?> include,
      SkylarkList<?> exclude,
      Integer excludeDirectories,
      Object allowEmpty,
      FuncallExpression ast,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    return MutableList.of(thread);
  }

  @Override
  public Object existingRule(String name, FuncallExpression ast, StarlarkThread thread)
      throws EvalException, InterruptedException {
    return null;
  }

  @Override
  public SkylarkDict<String, SkylarkDict<String, Object>> existingRules(
      FuncallExpression ast, StarlarkThread thread) throws EvalException, InterruptedException {
    return SkylarkDict.of(thread);
  }

  @Override
  public NoneType packageGroup(
      String name,
      SkylarkList<?> packages,
      SkylarkList<?> includes,
      FuncallExpression ast,
      StarlarkThread thread)
      throws EvalException {
    return null;
  }

  @Override
  public NoneType exportsFiles(
      SkylarkList<?> srcs,
      Object visibility,
      Object licenses,
      FuncallExpression ast,
      StarlarkThread thread)
      throws EvalException {
    return null;
  }

  @Override
  public String packageName(FuncallExpression ast, StarlarkThread thread) throws EvalException {
    return "";
  }

  @Override
  public String repositoryName(Location location, StarlarkThread thread) throws EvalException {
    return "";
  }

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    // Bazel's notion of the global "native" isn't fully exposed via public interfaces, for example,
    // as far as native rules are concerned. Returning None on all unsupported invocations of
    // native.[func_name]() is the safest "best effort" approach to implementing a fake for
    // "native".
    return new FakeStarlarkCallable(name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() throws EvalException {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public String getErrorMessageForUnknownField(String field) {
    return "";
  }
}
