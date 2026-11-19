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
package com.google.devtools.build.lib.skyframe.serialization.bytecode;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.bytecode.FieldAccessUtils.unreflectAdaptedGetter;
import static com.google.devtools.build.lib.skyframe.serialization.bytecode.FieldAccessUtils.unreflectAdaptedSetter;
import static org.junit.Assert.assertThrows;

import java.lang.invoke.MethodHandle;
import java.lang.reflect.Field;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FieldAccessUtilsTest {

  private static final class SampleHolder {
    private final String finalText;
    int packageCount;

    private SampleHolder(String finalText, int packageCount) {
      this.finalText = finalText;
      this.packageCount = packageCount;
    }
  }

  private static final class CustomException extends Exception {
    private CustomException(String message) {
      super(message);
    }
  }

  @Test
  public void unreflectAdaptedGetterAndSetter_readsAndMutatesPrivateFinalAndPrimitiveFields()
      throws Throwable {
    Field finalText = SampleHolder.class.getDeclaredField("finalText");
    Field packageCount = SampleHolder.class.getDeclaredField("packageCount");
    MethodHandle textGetter = unreflectAdaptedGetter(finalText, Object.class);
    MethodHandle textSetter = unreflectAdaptedSetter(finalText, Object.class);
    MethodHandle countGetter = unreflectAdaptedGetter(packageCount, int.class);
    MethodHandle countSetter = unreflectAdaptedSetter(packageCount, int.class);

    SampleHolder holder = new SampleHolder("before", 10);
    holder.packageCount = 15;
    assertThat(textGetter.invokeExact((Object) holder)).isEqualTo("before");
    assertThat((int) countGetter.invokeExact((Object) holder)).isEqualTo(15);

    textSetter.invokeExact((Object) holder, (Object) "after");
    countSetter.invokeExact((Object) holder, 99);
    assertThat(holder.finalText).isEqualTo("after");
    assertThat(holder.packageCount).isEqualTo(99);
  }

  @Test
  public void unreflectAdaptedGetterAndSetter_accessesCrossPackageSuperclassPrivateField()
      throws Throwable {
    Field detailMessage = Throwable.class.getDeclaredField("detailMessage");
    MethodHandle getter = unreflectAdaptedGetter(detailMessage, Object.class);
    MethodHandle setter = unreflectAdaptedSetter(detailMessage, Object.class);

    CustomException ex = new CustomException("initial");
    assertThat(getter.invokeExact((Object) ex)).isEqualTo("initial");

    setter.invokeExact((Object) ex, (Object) "updated");
    assertThat(ex).hasMessageThat().isEqualTo("updated");
  }

  @Test
  public void unreflectAdaptedSetter_wrongFieldValueType_throwsClassCastException()
      throws Exception {
    Field finalText = SampleHolder.class.getDeclaredField("finalText");
    MethodHandle textSetter = unreflectAdaptedSetter(finalText, Object.class);
    Object target = new SampleHolder("before", 1);
    Object badValue = 123;

    assertThrows(ClassCastException.class, () -> invokeObjectSetter(textSetter, target, badValue));
  }

  private static void invokeObjectSetter(MethodHandle setter, Object target, Object value)
      throws Throwable {
    setter.invokeExact(target, value);
  }
}
