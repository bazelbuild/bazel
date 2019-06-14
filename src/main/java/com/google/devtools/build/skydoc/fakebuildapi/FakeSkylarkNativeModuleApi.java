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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkNativeModuleApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;

/**
 * Fake implementation of {@link SkylarkNativeModuleApi}.
 */
public class FakeSkylarkNativeModuleApi implements SkylarkNativeModuleApi {

  @Override
  public SkylarkList<?> glob(
      SkylarkList<?> include,
      SkylarkList<?> exclude,
      Integer excludeDirectories,
      Object allowEmpty,
      FuncallExpression ast,
      Environment env)
      throws EvalException, InterruptedException {
    return MutableList.of(env);
  }

  @Override
  public Object existingRule(String name, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    return null;
  }

  @Override
  public SkylarkDict<String, SkylarkDict<String, Object>> existingRules(FuncallExpression ast,
      Environment env) throws EvalException, InterruptedException {
    return SkylarkDict.of(env);
  }

  @Override
  public NoneType packageGroup(String name, SkylarkList<?> packages, SkylarkList<?> includes,
      FuncallExpression ast, Environment env) throws EvalException {
    return null;
  }

  @Override
  public NoneType exportsFiles(SkylarkList<?> srcs, Object visibility, Object licenses,
      FuncallExpression ast, Environment env) throws EvalException {
    return null;
  }

  @Override
  public String packageName(FuncallExpression ast, Environment env) throws EvalException {
    return "";
  }

  @Override
  public String repositoryName(Location location, Environment env) throws EvalException {
    return "";
  }
}
