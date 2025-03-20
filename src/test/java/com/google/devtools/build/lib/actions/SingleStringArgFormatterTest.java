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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.SingleStringArgFormatter.format;
import static com.google.devtools.build.lib.actions.SingleStringArgFormatter.formattedLength;
import static com.google.devtools.build.lib.actions.SingleStringArgFormatter.isValid;
import static org.junit.Assert.assertThrows;

import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link SingleStringArgFormatter} */
@RunWith(TestParameterInjector.class)
public final class SingleStringArgFormatterTest {

  @Test
  public void invalid(
      @TestParameter({
            "hello",
            "hello %%s",
            "hello %s %s",
            "%s hello %s",
            "hello %",
            "hello %f",
            "hello %s %f",
            "hello %s %"
          })
          String format) {
    assertThat(isValid(format)).isFalse();
    assertThrows(IllegalArgumentException.class, () -> format(format, "world"));
    assertThrows(IllegalArgumentException.class, () -> formattedLength(format));
  }

  @Test
  @TestParameters("{format: 'hello %s', expected: 'hello world'}")
  @TestParameters("{format: '%s hello', expected: 'world hello'}")
  @TestParameters("{format: 'hello %s, hello', expected: 'hello world, hello'}")
  @TestParameters("{format: 'hello %%s %s', expected: 'hello %s world'}")
  public void valid(String format, String expected) {
    assertThat(isValid(format)).isTrue();
    assertThat(format(format, "world")).isEqualTo(expected);
    assertThat(formattedLength(format)).isEqualTo(expected.length() - "world".length());
  }
}
