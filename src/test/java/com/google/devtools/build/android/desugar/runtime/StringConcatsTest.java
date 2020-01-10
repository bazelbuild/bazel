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

package com.google.devtools.build.android.desugar.runtime;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link StringConcats}, the utility class for string concatenation desugaring. */
@RunWith(JUnit4.class)
public final class StringConcatsTest {

  @Test
  public void concat_directConcat() {
    assertThat(StringConcats.concat(new String[] {"a", "bc"}, "\1\1", new Object[] {}))
        .isEqualTo("abc");
  }

  @Test
  public void concat_recipeTemplate() {
    assertThat(
            StringConcats.concat(
                new String[] {"a", "bc"}, "\2<tag>\1<br>\1\2", new Object[] {"<li>", "</tag>"}))
        .isEqualTo("<li><tag>a<br>bc</tag>");
  }
}
