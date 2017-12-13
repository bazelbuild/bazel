// Copyright 2017 The Bazel Authors. All Rights Reserved.
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

package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.lang.reflect.Field;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Runtime}. */
@RunWith(JUnit4.class)
public final class RuntimeTest {

  private static final Object DUMMY = new Object();
  private static final BuiltinFunction DUMMY_FUNC = new BuiltinFunction("dummyFunc") {
    // This would normally be done by @SkylarkSignature annotation and configure(), but a simple
    // stub suffices.
    @Override
    public Class<?> getObjectType() {
      return DummyType.class;
    }
  };

  private static class DummyType implements SkylarkValue {
    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("DummyType");
    }
  }

  @Test
  public void checkRegistry_GetBuiltins() {
    Runtime.BuiltinRegistry reg = new Runtime.BuiltinRegistry();
    reg.registerBuiltin(DummyType.class, "dummy", DUMMY);
    assertThat(reg.getBuiltins()).contains(DUMMY);
  }

  @Test
  public void checkRegistry_GetFunction() {
    Runtime.BuiltinRegistry reg = new Runtime.BuiltinRegistry();
    reg.registerFunction(DummyType.class, DUMMY_FUNC);
    assertThat(reg.getFunction(DummyType.class, "dummyFunc")).isEqualTo(DUMMY_FUNC);
  }

  @Test
  public void checkRegistry_GetFunctionNames() {
    Runtime.BuiltinRegistry reg = new Runtime.BuiltinRegistry();
    reg.registerFunction(DummyType.class, DUMMY_FUNC);
    assertThat(reg.getFunctionNames(DummyType.class)).contains("dummyFunc");
  }

  @Test
  public void checkStaticallyRegistered_Method() throws Exception {
    Field splitField = MethodLibrary.class.getDeclaredField("split");
    splitField.setAccessible(true);
    Object splitFieldValue = splitField.get(null);
    Object splitFunc = Runtime.getBuiltinRegistry().getFunction(String.class, "split");
    assertThat(splitFunc).isSameAs(splitFieldValue);
  }

  @Test
  public void checkStaticallyRegistered_Global() throws Exception {
    Field lenField = MethodLibrary.class.getDeclaredField("len");
    lenField.setAccessible(true);
    Object lenFieldValue = lenField.get(null);
    Set<Object> builtins = Runtime.getBuiltinRegistry().getBuiltins();
    assertThat(builtins).contains(lenFieldValue);
  }
}
