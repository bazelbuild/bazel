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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.truth.StringSubject;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.android.AndroidDataConverter.JoinerType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidDataConverter}. */
@RunWith(JUnit4.class)
public class AndroidDataConverterTest {
  private static final String TO_MAP = "a placeholder object to ensure the correct thing is mapped";

  @Test
  public void testWithLabel() throws LabelSyntaxException {
    assertMap(
            AndroidDataConverter.<String>builder(JoinerType.COLON_COMMA)
                .withLabel(getFunction(Label.create("foo/bar", "baz")))
                .withLabel(getFunction(Label.create("foo/bar", "quux")))
                .build())
        .isEqualTo("//foo/bar\\:baz://foo/bar\\:quux");
  }

  @Test
  public void testWithLabel_escapingNotNeeded() throws LabelSyntaxException {
    assertMap(
            AndroidDataConverter.<String>builder(JoinerType.SEMICOLON_AMPERSAND)
                .withLabel(getFunction(Label.create("foo/bar", "baz")))
                .withLabel(getFunction(Label.create("foo/bar", "quux")))
                .build())
        .isEqualTo("//foo/bar:baz;//foo/bar:quux");
  }

  @Test
  public void testWithRoots() {
    assertMap(
            AndroidDataConverter.<String>builder(JoinerType.COLON_COMMA)
                .withRoots(
                    getFunction(
                        ImmutableList.of(
                            PathFragment.create("a/b/c"), PathFragment.create("d/e/f/g"))))
                .withRoots(getFunction(ImmutableList.of(PathFragment.create("h/i"))))
                .build())
        .isEqualTo("a/b/c#d/e/f/g:h/i");
  }

  @Test
  public void testMaybeWithArtifact() {
    assertMap(
            AndroidDataConverter.<String>builder(JoinerType.COLON_COMMA)
                .maybeWithArtifact(getFunction(null))
                .maybeWithArtifact(getFunction(null))
                .build())
        .isEqualTo(":");
  }

  @Test
  public void test() {
    assertMap(
            AndroidDataConverter.<String>builder(JoinerType.SEMICOLON_AMPERSAND)
                .withRoots(
                    getFunction(
                        ImmutableList.of(PathFragment.create("q/w"), PathFragment.create("r/t/y"))))
                .with(getFunction("some string"))
                .with(getFunction("another string"))
                .build())
        .isEqualTo("q/w#r/t/y;some string;another string");
  }

  @Test
  public void testGetVectorArg() {
    AndroidDataConverter<String> converter =
        AndroidDataConverter.<String>builder(JoinerType.SEMICOLON_AMPERSAND)
            .with(x -> x + "1")
            .with(x -> x + "2")
            .build();

    assertThat(
            CustomCommandLine.builder()
                .addAll(
                    "somekey",
                    converter.getVectorArg(
                        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, "a", "b")))
                .build()
                .toString())
        .isEqualTo("somekey a1;a2&b1;b2");
  }

  private static StringSubject assertMap(AndroidDataConverter<String> converter) {
    return assertThat(converter.map(TO_MAP));
  }

  private static <T> Function<String, T> getFunction(T t) {
    return x -> {
      assertThat(x).isSameInstanceAs(TO_MAP);
      return t;
    };
  }
}
