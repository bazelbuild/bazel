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
package com.google.devtools.build.lib.query2.query.output;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.MoreCollectors.onlyElement;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.common.options.OptionsParser;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ProtoOutputFormatter}. */
@RunWith(JUnit4.class)
public final class ProtoOutputFormatterTest extends PackageLoadingTestCase {

  @Before
  public void writeDictRule() throws Exception {
    scratch.file(
        "pkg/defs.bzl",
        """
        dict_rule = rule(
            implementation = lambda ctx: [],
            attrs = {"entries": attr.string_dict()},
        )
        """);
  }

  @Test
  public void flattenedDictWithManySelectsIsTheUnionOfAllBranches() throws Exception {
    int selects = 40;
    StringBuilder build = new StringBuilder("load(':defs.bzl', 'dict_rule')\n");
    for (int i = 0; i < selects; i++) {
      build.append(
          String.format("config_setting(name = 'c%d', define_values = {'f%d': '1'})\n", i, i));
    }
    // Each select has a different condition, so the attribute has 2^40 possible values.
    build.append("dict_rule(name = 't', entries = ");
    build.append(
        IntStream.range(0, selects)
            .mapToObj(
                i ->
                    String.format(
                        "select({':c%d': {'k%d': 'v%d'}, '//conditions:default': {}})", i, i, i))
            .collect(Collectors.joining(" | ")));
    build.append(")\n");
    scratch.file("pkg/BUILD", build.toString());

    assertThat(flattenedEntries("//pkg:t"))
        .containsExactlyEntriesIn(
            IntStream.range(0, selects)
                .boxed()
                .collect(toImmutableMap(i -> "k" + i, i -> "v" + i)));
  }

  @Test
  public void flattenedDictTakesTheLastValueForARepeatedKey() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        load(":defs.bzl", "dict_rule")

        config_setting(name = "a", define_values = {"a": "1"})

        config_setting(name = "b", define_values = {"b": "1"})

        dict_rule(
            name = "t",
            entries = select({
                ":a": {"k": "a1", "only_a": "x"},
                "//conditions:default": {"k": "default1"},
            }) | select({
                ":b": {"k": "b2"},
                "//conditions:default": {},
            }),
        )
        """);

    assertThat(flattenedEntries("//pkg:t")).containsExactly("k", "b2", "only_a", "x");
  }

  @Test
  public void flattenedDictKeepsALaterSelectsOverride() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        load(":defs.bzl", "dict_rule")

        config_setting(name = "a", define_values = {"a": "1"})

        dict_rule(
            name = "t",
            entries = select({
                ":a": {"k": "A"},
                "//conditions:default": {"k": "A"},
            }) | {"k": "B"},
        )
        """);

    assertThat(flattenedEntries("//pkg:t")).containsExactly("k", "B");
  }

  private ImmutableMap<String, String> flattenedEntries(String label) throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(QueryOptions.class).build();
    parser.parse("--proto:flatten_selects");
    ProtoOutputFormatter formatter = new ProtoOutputFormatter();
    formatter.setOptions(
        parser.getOptions(QueryOptions.class),
        AspectResolver.Mode.OFF.createResolver(getPackageManager(), reporter),
        DigestHashFunction.SHA256.getHashFunction());
    Build.Target target = formatter.toTargetProtoBuffer(getTarget(label), LabelPrinter.legacy());
    Build.Attribute entries =
        target.getRule().getAttributeList().stream()
            .filter(attr -> attr.getName().equals("entries"))
            .collect(onlyElement());
    return entries.getStringDictValueList().stream()
        .collect(toImmutableMap(Build.StringDictEntry::getKey, Build.StringDictEntry::getValue));
  }
}
