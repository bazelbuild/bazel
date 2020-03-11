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
package com.google.devtools.build.android.xml;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.xml.SimpleXmlResourceValue.convertPrimitiveToString;

import com.android.aapt.Resources.Primitive;
import java.util.Locale;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SimpleXmlResourceValue}. */
@RunWith(JUnit4.class)
public final class SimpleXmlResourceValueTest {

  @Test
  public void convertPrimitiveToString_float() {
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setFloatValue(1.0f).build()))
        .isEqualTo("1.0");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setFloatValue(2.0f).build()))
        .isEqualTo("2.0");
  }

  @Test
  public void convertPrimitiveToString_int() {
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setIntDecimalValue(1).build()))
        .isEqualTo("1");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setIntDecimalValue(15).build()))
        .isEqualTo("15");

    assertThat(convertPrimitiveToString(Primitive.newBuilder().setIntHexadecimalValue(1).build()))
        .isEqualTo("0x1");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setIntHexadecimalValue(15).build()))
        .isEqualTo("0xf");
  }

  @Test
  public void convertPrimitiveToString_boolean() {
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setBooleanValue(false).build()))
        .isEqualTo("false");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setBooleanValue(true).build()))
        .isEqualTo("true");
  }

  @Test
  public void convertPrimitiveToString_dimension() {
    // varying units (bits [3:0])
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x0).build()))
        .isEqualTo("0.0px");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x1).build()))
        .isEqualTo("0.0dp");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x2).build()))
        .isEqualTo("0.0sp");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x3).build()))
        .isEqualTo("0.0pt");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x4).build()))
        .isEqualTo("0.0in");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x5).build()))
        .isEqualTo("0.0mm");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0xF).build()))
        .isEqualTo("0.0 (unknown unit)");

    // varying mantissa (bits [31:8]); negative numbers are stored using two's complement.
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x100).build()))
        .isEqualTo("1.0px");
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0xFFFFFF00).build()))
        .isEqualTo("-1.0px");

    // varying exponent (bits [5:4])
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x40000000).build()))
        .isEqualTo("4194304.0px");
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x40000010).build()))
        .isEqualTo("32768.0px");
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x40000020).build()))
        .isEqualTo("128.0px");
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x40000030).build()))
        .isEqualTo("0.5px");
  }

  @Test
  public void convertPrimitiveToString_dimensionIsNotLocalized() {
    Locale.setDefault(Locale.GERMAN); // comma is normally used as decimal point
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setDimensionValue(0x0).build()))
        .isEqualTo("0.0px");
  }

  @Test
  public void convertPrimitiveToString_fraction() {
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setFractionValue(0x0).build()))
        .isEqualTo("0.0%");
    assertThat(convertPrimitiveToString(Primitive.newBuilder().setFractionValue(0x1).build()))
        .isEqualTo("0.0%p");
  }

  @Test
  public void convertPrimitiveToString_color() {
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setColorRgb8Value(0xffadbeef).build()))
        .isEqualTo("#FFADBEEF");
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setColorArgb8Value(0xdeadbeef).build()))
        .isEqualTo("#DEADBEEF");
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setColorRgb4Value(0xffadbeef).build()))
        .isEqualTo("#FFADBEEF");
    assertThat(
            convertPrimitiveToString(Primitive.newBuilder().setColorArgb4Value(0xdeadbeef).build()))
        .isEqualTo("#DEADBEEF");
  }
}
