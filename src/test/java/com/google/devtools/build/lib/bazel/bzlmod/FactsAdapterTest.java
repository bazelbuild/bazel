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
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static org.junit.Assert.assertThrows;

import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
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
  private static final FactsAdapter ADAPTER = new FactsAdapter();

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
      Facts facts = ADAPTER.read(reader);
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
        ADAPTER.fromJson(
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
        ADAPTER.fromJson(
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
  public void unicodeKeysAndValuesRoundTrip() throws Exception {
    String json =
        """
        {"café 😀": "caf\\u00e9 \\ud83d\\ude00", "caf\\u00e9": "café 😀", "\\u2028": "\\u2029"}
        """;
    Facts facts = ADAPTER.fromJson(json);
    assertThat((Map<?, ?>) facts.value())
        .containsExactly(
            unicodeToInternal("café 😀"), unicodeToInternal("café 😀"),
            unicodeToInternal("café"), unicodeToInternal("café 😀"),
            unicodeToInternal("\u2028"), unicodeToInternal("\u2029"));
    assertThat(JsonParser.parseString(ADAPTER.toJson(facts)))
        .isEqualTo(JsonParser.parseString(json));
  }

  @Test
  public void depthLimitAppliesAfterDuplicateNamesAreResolved() throws Exception {
    String atLimit = "[".repeat(6) + "0" + "]".repeat(6);
    String tooDeep = "[" + atLimit + "]";
    assertThat((Map<?, ?>) ADAPTER.fromJson("{\"a\":" + atLimit + "}").value()).containsKey("a");
    assertThat((Map<?, ?>) ADAPTER.fromJson("{\"a\":" + tooDeep + ",\"a\":1}").value())
        .containsExactly("a", StarlarkInt.of(1));
    JsonParseException e =
        assertThrows(JsonParseException.class, () -> ADAPTER.fromJson("{\"a\":" + tooDeep + "}"));
    assertThat(e).hasMessageThat().contains("Facts cannot be nested more than 7 levels deep");
  }

  @Test
  public void factsMustBeAnObject() {
    assertThrows(JsonParseException.class, () -> ADAPTER.fromJson("[null]"));
    assertThrows(JsonParseException.class, () -> ADAPTER.fromJson(""));
  }
}
