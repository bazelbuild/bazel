// Copyright 2016 The Bazel Authors. All rights reserved.
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


package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for AspectDescriptor.
 */
@RunWith(JUnit4.class)
public class AspectDescriptorTest {

  @Test
  public void serializeDescriptorNoArguments() {
    assertDescription("foobar", "foobar");
  }

  @Test
  public void serializeDescriptorArgument() {
    assertDescription("foobar[x=\"1\"]",
        "foobar",
        "x", "1");
  }

  @Test
  public void serializeDescriptorArgumentEscaped() {
    assertDescription("foobar[x=\"\\\"1\\\"\"]",
        "foobar",
        "x", "\"1\"");
  }


  @Test
  public void serializeDescriptorTwoArguments() {
    assertDescription("foobar[x=\"1\",y=\"2\"]",
        "foobar",
        "x", "1",
        "y", "2");
  }

  @Test
  public void serializeDescriptorTwoArgumentsMulti() {
    assertDescription("foobar[x=\"1\",y=\"2\",y=\"3\"]",
        "foobar",
        "x", "1",
        "y", "2",
        "y", "3");
  }

  private static void assertDescription(
      String description,
      String aspectClassName,
      String... params) {
    assertThat(aspectDescriptor(aspectClass(aspectClassName), params).getDescription())
        .isEqualTo(description);
  }

  private static AspectDescriptor aspectDescriptor(
      AspectClass aspectClass,
      String... parameters) {
    assertThat(parameters.length % 2).isEqualTo(0);

    AspectParameters.Builder params = new AspectParameters.Builder();
    for (int i = 0; i < parameters.length; i += 2) {
      params.addAttribute(parameters[i], parameters[i + 1]);
    }
    return AspectDescriptor.of(aspectClass, params.build());
  }

  private static AspectClass aspectClass(final String name) {
    return new NativeAspectClass() {

      @Override
      public String getName() {
        return name;
      }

      @Override
      public AspectDefinition getDefinition(AspectParameters aspectParameters) {
        throw new UnsupportedOperationException();
      }
    };
  }
}
