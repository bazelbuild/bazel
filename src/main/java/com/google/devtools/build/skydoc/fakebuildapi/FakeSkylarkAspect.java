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

import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAspectApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Fake implementation of {@link SkylarkAspectApi}. */
public class FakeSkylarkAspect extends BaseFunction implements SkylarkAspectApi {

  /**
   * Each fake is constructed with a unique name, controlled by this counter being the name suffix.
   */
  private static int idCounter = 0;

  private final String name = "AspectIdentifier" + idCounter++;

  public FakeSkylarkAspect() {
    super(FunctionSignature.KWARGS);
  }

  @Override
  public Object callImpl(
      StarlarkThread thread,
      @Nullable FuncallExpression call,
      List<Object> args,
      Map<String, Object> kwargs)
      throws EvalException {
    throw new EvalException("not implemented");
  }

  @Override
  public String getName() {
    return name;
  }

  @Override
  public void repr(Printer printer) {}
}
