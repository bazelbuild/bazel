// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options.testing;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converters;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the ConverterTesterMap map builder. */
@RunWith(JUnit4.class)
public final class ConverterTesterMapTest {

  @Test
  public void add_mapsTestedConverterClassToTester() throws Exception {
    ConverterTester stringTester = new ConverterTester(Converters.StringConverter.class);
    ConverterTester intTester = new ConverterTester(Converters.IntegerConverter.class);
    ConverterTester doubleTester = new ConverterTester(Converters.DoubleConverter.class);
    ConverterTester booleanTester = new ConverterTester(Converters.BooleanConverter.class);
    ConverterTesterMap map =
        new ConverterTesterMap.Builder()
            .add(stringTester)
            .add(intTester)
            .add(doubleTester)
            .add(booleanTester)
            .build();
    assertThat(map)
        .containsExactly(
            Converters.StringConverter.class,
            stringTester,
            Converters.IntegerConverter.class,
            intTester,
            Converters.DoubleConverter.class,
            doubleTester,
            Converters.BooleanConverter.class,
            booleanTester);
  }

  @Test
  public void addAll_mapsTestedConverterClassesToTester() throws Exception {
    ConverterTester stringTester = new ConverterTester(Converters.StringConverter.class);
    ConverterTester intTester = new ConverterTester(Converters.IntegerConverter.class);
    ConverterTester doubleTester = new ConverterTester(Converters.DoubleConverter.class);
    ConverterTester booleanTester = new ConverterTester(Converters.BooleanConverter.class);
    ConverterTesterMap map =
        new ConverterTesterMap.Builder()
            .addAll(ImmutableList.of(stringTester, intTester, doubleTester, booleanTester))
            .build();
    assertThat(map)
        .containsExactly(
            Converters.StringConverter.class,
            stringTester,
            Converters.IntegerConverter.class,
            intTester,
            Converters.DoubleConverter.class,
            doubleTester,
            Converters.BooleanConverter.class,
            booleanTester);
  }

  @Test
  public void addAll_dumpsConverterTesterMapIntoNewMap() throws Exception {
    ConverterTester stringTester = new ConverterTester(Converters.StringConverter.class);
    ConverterTester intTester = new ConverterTester(Converters.IntegerConverter.class);
    ConverterTester doubleTester = new ConverterTester(Converters.DoubleConverter.class);
    ConverterTester booleanTester = new ConverterTester(Converters.BooleanConverter.class);
    ConverterTesterMap baseMap =
        new ConverterTesterMap.Builder()
            .addAll(ImmutableList.of(stringTester, intTester, doubleTester))
            .build();
    ConverterTesterMap map =
        new ConverterTesterMap.Builder().addAll(baseMap).add(booleanTester).build();
    assertThat(map)
        .containsExactly(
            Converters.StringConverter.class,
            stringTester,
            Converters.IntegerConverter.class,
            intTester,
            Converters.DoubleConverter.class,
            doubleTester,
            Converters.BooleanConverter.class,
            booleanTester);
  }

  @Test
  public void build_forbidsDuplicates() throws Exception {
    ConverterTesterMap.Builder builder =
        new ConverterTesterMap.Builder()
            .add(new ConverterTester(Converters.StringConverter.class))
            .add(new ConverterTester(Converters.IntegerConverter.class))
            .add(new ConverterTester(Converters.DoubleConverter.class))
            .add(new ConverterTester(Converters.BooleanConverter.class))
            .add(new ConverterTester(Converters.BooleanConverter.class));

    IllegalArgumentException expected =
        assertThrows(IllegalArgumentException.class, () -> builder.build());
    assertThat(expected)
        .hasMessageThat()
        .contains(Converters.BooleanConverter.class.getSimpleName());
  }
}
