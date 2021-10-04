// Copyright 2021 The Bazel Authors. All rights reserved.
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
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Structure;
import net.starlark.java.eval.Tuple;

/**
 * A fake Starlark structure that returns itself for any field that it's asked for and that can be
 * called, but the call does nothing.
 */
@StarlarkBuiltin(name = "FakeDeepStructure", documented = false)
public class FakeDeepStructure extends FakeProviderApi implements Structure, StarlarkCallable {
  private final String fullName;

  private FakeDeepStructure(@Nullable String name, String fullName) {
    super(name);
    this.fullName = fullName;
  }

  /** Creates a new fake deep structure with the given name. */
  public static FakeDeepStructure create(String name) {
    return new FakeDeepStructure(name, name);
  }

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    return new FakeDeepStructure(name, fullName + "." + name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public String getErrorMessageForUnknownField(String field) {
    return null;
  }

  @Override
  public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) {
    return new FakeDeepStructure(getName() + "()", fullName + "()");
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<fake object ").append(fullName).append(">");
  }
}
