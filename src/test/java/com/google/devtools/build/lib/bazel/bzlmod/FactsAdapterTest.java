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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.gson.JsonSyntaxException;
import com.google.gson.Strictness;
import com.google.gson.stream.JsonReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FactsAdapterTest {
  private static Facts read(String json) throws IOException {
    try (var reader = new JsonReader(new StringReader(json))) {
      return new FactsAdapter().read(reader);
    }
  }

  @Test
  public void factsAreSortedAndImmutableAndReadOnlyOneValue() throws Exception {
    try (var reader =
        new JsonReader(
            new StringReader(
                """
                [{"z": [null, true, false, 4294967296, 1.25, 1e300],
                  "a": {"c": "third", "a": "first"}}, true]
                """))) {
      reader.beginArray();
      Facts facts = new FactsAdapter().read(reader);
      assertThat(reader.nextBoolean()).isTrue();
      reader.endArray();

      assertThat(facts.value().keySet()).containsExactly("a", "z").inOrder();
      assertThat(facts.value().isImmutable()).isTrue();
      var nested = (Dict<?, ?>) facts.value().get("a");
      assertThat(nested.keySet()).containsExactly("a", "c").inOrder();
      assertThat(nested.isImmutable()).isTrue();
      var list = (StarlarkList<?>) facts.value().get("z");
      assertThat((Iterable<?>) list)
          .containsExactly(
              Starlark.NONE,
              true,
              false,
              StarlarkInt.of(4294967296L),
              StarlarkFloat.of(1.25),
              StarlarkFloat.of(1e300))
          .inOrder();
      assertThat(list.isImmutable()).isTrue();
    }
  }

  @Test
  public void integersAreUnboundedAndOnlyDecimalOrExponentNumbersAreFloats() throws Exception {
    var facts =
        read(
            """
            {"big": 987654321012345678909876543210,
             "int": -0, "decimal": -0.0, "exponent": 1e3, "infinity": 1e999}
            """);
    assertThat((Map<?, ?>) facts.value())
        .containsExactly(
            "big", StarlarkInt.parse("987654321012345678909876543210", 10),
            "int", StarlarkInt.of(0),
            "decimal", StarlarkFloat.of(-0.0),
            "exponent", StarlarkFloat.of(1e3),
            "infinity", StarlarkFloat.of(Double.POSITIVE_INFINITY));
  }

  @Test
  public void nullObjectMembersAreOmittedIncludingAfterDuplicates() throws Exception {
    var facts =
        read(
            """
            {"omit": null, "remove": 3, "remove": null, "keep": null, "keep": 4,
             "nested": [{"omit": null}, null], "same": 1, "same": 5}
            """);
    assertThat((Map<?, ?>) facts.value())
        .containsExactly(
            "keep", StarlarkInt.of(4),
            "nested", StarlarkList.immutableOf(Dict.empty(), Starlark.NONE),
            "same", StarlarkInt.of(5));
  }

  @Test
  public void stringsKeepFactsEncodingForLiteralsAndEscapes() throws Exception {
    var facts =
        read(
            """
            {"literal": "café 😀", "escaped": "caf\\u00e9 \\ud83d\\ude00",
             "quoted": "<tag> \\n \\t \\u0000", "separators": "\\u2028\\u2029"}
            """);
    // JsonParser first unescaped the JSON. Gson then emitted ordinary non-ASCII chars literally,
    // but always escaped U+2028/U+2029; Json.decode encoded those escapes as raw UTF-8 bytes.
    assertThat((Map<?, ?>) facts.value())
        .containsExactly(
            "literal", "café 😀",
            "escaped", "café 😀",
            "quoted", "<tag> \n \t \u0000",
            "separators", "\u00e2\u0080\u00a8\u00e2\u0080\u00a9");
  }

  @Test
  public void decodedNameCollisionsKeepLastValueAfterRawJsonDuplicates() throws Exception {
    var facts = read("{\"\\u2028\":1,\"\u00e2\u0080\u00a8\":2,\"\\u2028\":3}");
    assertThat((Map<?, ?>) facts.value())
        .containsExactly("\u00e2\u0080\u00a8", StarlarkInt.of(2));
  }

  @Test
  public void depthValidationOnlyAppliesToValuesNotReplacedByDuplicateKeys() throws Exception {
    var facts = read("{\"a\":[[[[[[[[0]]]]]]]],\"a\":1}");
    assertThat((Map<?, ?>) facts.value()).containsExactly("a", StarlarkInt.of(1));

    IOException e = assertThrows(IOException.class, () -> read("{\"a\":[[[[[[[[0]]]]]]]]}"));
    assertThat(e).hasMessageThat().contains("Facts cannot be nested more than 7 levels deep");
  }

  @Test
  public void factsMustBeAnObject() {
    IOException e = assertThrows(IOException.class, () -> read("[null]"));
    assertThat(e).hasMessageThat().contains("Failed to decode facts JSON");
  }

  @Test
  public void exhaustedReaderIsNotAnEmptyInput() throws IOException {
    assertThrows(IOException.class, () -> read(""));
    try (var reader = new JsonReader(new StringReader("{}"))) {
      new FactsAdapter().read(reader);
      assertThrows(IllegalStateException.class, () -> new FactsAdapter().read(reader));
    }
  }

  @Test
  public void readRespectsExplicitStrictnessAndRestoresDefaultStrictness() throws Exception {
    try (var reader = new JsonReader(new StringReader("{unquoted: 'value'}"))) {
      assertThat(reader.getStrictness()).isEqualTo(Strictness.LEGACY_STRICT);
      var facts = new FactsAdapter().read(reader);
      assertThat((Map<?, ?>) facts.value()).containsExactly("unquoted", "value");
      assertThat(reader.getStrictness()).isEqualTo(Strictness.LEGACY_STRICT);
    }
    try (var reader = new JsonReader(new StringReader("{unquoted: 'value'}"))) {
      reader.setStrictness(Strictness.STRICT);
      assertThrows(JsonSyntaxException.class, () -> new FactsAdapter().read(reader));
      assertThat(reader.getStrictness()).isEqualTo(Strictness.STRICT);
    }
  }
}
