// Copyright 2023 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.bazel.bzlmod.AttributeValuesAdapter.STRING_ESCAPE_SEQUENCE;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.gson.JsonSyntaxException;
import com.google.gson.Strictness;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.Tuple;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class AttributeValuesAdapterTest extends FoundationTestCase {

  @Test
  public void testAttributeValuesAdapter() throws IOException {
    Dict.Builder<String, Object> dict = new Dict.Builder<>();
    Label l1 = Label.parseCanonicalUnchecked("@//foo:bar");
    Label l2 = Label.parseCanonicalUnchecked("@//foo:tar");
    dict.put("Integer", StarlarkInt.of(56));
    dict.put("Boolean", false);
    dict.put("String", "Hello String");
    dict.put("StringWithAngleBrackets", "<Hello>");
    dict.put("Unicode", unicodeToInternal("café/🙂"));
    dict.put(
        "LabelLikeString",
        StarlarkList.of(Mutability.IMMUTABLE, "@//foo:bar", ":baz", "@@//baz:quz"));
    dict.put(
        "StringsWithEscapeSequence",
        StarlarkList.of(
            Mutability.IMMUTABLE,
            "@@//foo:bar" + STRING_ESCAPE_SEQUENCE,
            STRING_ESCAPE_SEQUENCE + "@@//foo:bar",
            STRING_ESCAPE_SEQUENCE + "@@//foo:bar" + STRING_ESCAPE_SEQUENCE,
            STRING_ESCAPE_SEQUENCE
                + STRING_ESCAPE_SEQUENCE
                + "@@//foo:bar"
                + STRING_ESCAPE_SEQUENCE
                + STRING_ESCAPE_SEQUENCE));
    dict.put("Label", l1);
    dict.put(
        "ListOfInts", StarlarkList.of(Mutability.IMMUTABLE, StarlarkInt.of(1), StarlarkInt.of(2)));
    dict.put("ListOfLabels", StarlarkList.of(Mutability.IMMUTABLE, l1, l2));
    dict.put("ListOfStrings", StarlarkList.of(Mutability.IMMUTABLE, "Hello", "There!"));
    Dict.Builder<Label, String> dictLabelString = new Dict.Builder<>();
    dictLabelString.put(l1, "Label#1");
    dictLabelString.put(l2, "Label#2");
    dict.put("DictOfLabel-String", dictLabelString.buildImmutable());

    Dict<String, Object> builtDict = dict.buildImmutable();
    AttributeValuesAdapter attrAdapter = new AttributeValuesAdapter();
    String jsonString;
    try (StringWriter stringWriter = new StringWriter()) {
      attrAdapter.write(new JsonWriter(stringWriter), AttributeValues.create(builtDict));
      jsonString = stringWriter.toString();
    }
    AttributeValues attributeValues;
    try (StringReader stringReader = new StringReader(jsonString)) {
      attributeValues = attrAdapter.read(new JsonReader(stringReader));
    }

    // Verify that the JSON string does not contain any escaped angle brackets.
    assertThat(jsonString).doesNotContain("\\u003c");
    // Verify that the String "Hello String" is preserved as is, without any additional escaping.
    assertThat(jsonString).contains(":\"Hello String\"");
    assertThat((Map<?, ?>) attributeValues.attributes()).containsExactlyEntriesIn(builtDict);
  }

  @Test
  public void testTuple() throws IOException {
    Dict.Builder<String, Object> dict = new Dict.Builder<>();
    dict.put("Tuple", Tuple.of("bzl", "mod"));

    Dict<String, Object> builtDict = dict.buildImmutable();
    AttributeValuesAdapter attrAdapter = new AttributeValuesAdapter();
    String jsonString;
    try (StringWriter stringWriter = new StringWriter()) {
      attrAdapter.write(new JsonWriter(stringWriter), AttributeValues.create(builtDict));
      jsonString = stringWriter.toString();
    }
    AttributeValues attributeValues;
    try (StringReader stringReader = new StringReader(jsonString)) {
      attributeValues = attrAdapter.read(new JsonReader(stringReader));
    }
    assertThat((Map<?, ?>) attributeValues.attributes())
        .containsExactly("Tuple", StarlarkList.of(Mutability.IMMUTABLE, "bzl", "mod"));
  }

  @Test
  public void testReadNumbersWithJsonPrimitiveIntConversions() throws IOException {
    AttributeValues value =
        new AttributeValuesAdapter()
            .fromJson(
                """
                {
                  "min": -2147483648,
                  "max": 2147483647,
                  "overflow": 2147483648,
                  "largerThanLong": 18446744073709551617,
                  "fraction": -3.75,
                  "exponent": 2e2
                }
                """);

    assertThat((Map<?, ?>) value.attributes())
        .containsExactly(
            "min", StarlarkInt.of(Integer.MIN_VALUE),
            "max", StarlarkInt.of(Integer.MAX_VALUE),
            "overflow", StarlarkInt.of(Integer.MIN_VALUE),
            "largerThanLong", StarlarkInt.of(1),
            "fraction", StarlarkInt.of(-3),
            "exponent", StarlarkInt.of(200))
        .inOrder();
  }

  @Test
  public void testReadDuplicateJsonNamesBeforeDecodingDictKeys() throws IOException {
    AttributeValues value =
        new AttributeValuesAdapter()
            .fromJson(
                """
                {
                  "first": 1,
                  "nested": {
                    "foo": 1,
                    "'foo'": 2,
                    "foo": 3,
                    "@@//foo:bar": [null, true],
                    "'@@//foo:bar'": "café/🙂"
                  },
                  "first": 4
                }
                """);

    assertThat(value.attributes().keySet()).containsExactly("first", "nested").inOrder();
    assertThat(value.attributes().get("first")).isEqualTo(StarlarkInt.of(4));
    assertThat((Map<?, ?>) value.attributes().get("nested"))
        .containsExactly(
            "foo", StarlarkInt.of(2),
            Label.parseCanonicalUnchecked("@@//foo:bar"),
                StarlarkList.immutableOf(Starlark.NONE, true),
            "@@//foo:bar", unicodeToInternal("café/🙂"))
        .inOrder();
  }

  @Test
  public void testReadDiscardsConversionErrorsInOverwrittenJsonValues() throws IOException {
    AttributeValues value =
        new AttributeValuesAdapter()
            .fromJson(
                """
                {
                  "label": "@@//foo:..", "label": 1,
                  "number": 1e9999999999, "number": 2,
                  "list": ["@@//foo:.."], "list": 3,
                  "dict": {"'": 0}, "dict": 4,
                  "nested": {"bad": "@@//foo:..", "bad": 5}
                }
                """);

    assertThat((Map<?, ?>) value.attributes())
        .containsExactly(
            "label", StarlarkInt.of(1),
            "number", StarlarkInt.of(2),
            "list", StarlarkInt.of(3),
            "dict", StarlarkInt.of(4),
            "nested", Dict.immutableCopyOf(Map.of("bad", StarlarkInt.of(5))))
        .inOrder();
  }

  @Test
  public void testReadDoesNotDiscardErrorsWhenOnlyDecodedKeysCollide() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new AttributeValuesAdapter().fromJson(
            "{\"a\":{\"foo\":\"@@//foo:..\",\"'foo'\":1}}"));
    assertThrows(
        IndexOutOfBoundsException.class,
        () -> new AttributeValuesAdapter().fromJson("{\"a\":{\"'\":0}}"));
    assertThrows(
        NumberFormatException.class,
        () -> new AttributeValuesAdapter().fromJson("{\"a\":1e9999999999}"));
  }

  @Test
  public void testReadReportsInvalidJsonBeforeConversionErrors() {
    assertThrows(
        JsonSyntaxException.class,
        () ->
            GsonTypeAdapterUtil.LOCKFILE_GSON.fromJson(
                "{\"a\":\"@@//foo:..\", ]", AttributeValues.class));
  }

  @Test
  public void testReadRestoresJsonReaderStrictness() throws IOException {
    try (JsonReader reader = new JsonReader(new StringReader("{unquoted: 1}"))) {
      AttributeValues value = new AttributeValuesAdapter().read(reader);

      assertThat((Map<?, ?>) value.attributes()).containsExactly("unquoted", StarlarkInt.of(1));
      assertThat(reader.getStrictness()).isEqualTo(Strictness.LEGACY_STRICT);
    }
  }
}
