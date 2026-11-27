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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AttributeValuesTest {

  private static AttributeValues create() {
    Label label = Label.parseCanonicalUnchecked("@@repo//pkg:target");
    return AttributeValues.create(
        Dict.immutableCopyOf(
            ImmutableMap.of(
                "strings", StarlarkList.immutableOf("a"),
                "labels", StarlarkList.immutableOf(label, label),
                "ints", StarlarkList.immutableOf(StarlarkInt.of(1)),
                "dict",
                    Dict.immutableCopyOf(ImmutableMap.of("k", StarlarkList.immutableOf("b"))))));
  }

  @Test
  public void create_internsListsOfStringsAndLabels() {
    AttributeValues first = create();
    AttributeValues second = create();

    assertThat(second).isEqualTo(first);
    Dict<String, Object> firstAttrs = first.attributes();
    Dict<String, Object> secondAttrs = second.attributes();
    assertThat(secondAttrs.get("strings")).isSameInstanceAs(firstAttrs.get("strings"));
    assertThat(secondAttrs.get("labels")).isSameInstanceAs(firstAttrs.get("labels"));
    assertThat(((Dict<?, ?>) secondAttrs.get("dict")).get("k"))
        .isSameInstanceAs(((Dict<?, ?>) firstAttrs.get("dict")).get("k"));
    assertThat(secondAttrs.get("ints")).isNotSameInstanceAs(firstAttrs.get("ints"));
  }
}
