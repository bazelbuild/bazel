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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelToStringEntryConverter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CoreOptionConverters}. */
@RunWith(JUnit4.class)
public class CoreOptionConvertersTest {

  private final LabelToStringEntryConverter labelToStringEntryConverter =
      new LabelToStringEntryConverter();

  @Test
  public void labelToStringEntryConverter_reverseForStarlark() throws Exception {
    Label label = Label.parseCanonical("//foo:bar");
    Map.Entry<Label, String> entry = Maps.immutableEntry(label, "baz");
    assertThat(labelToStringEntryConverter.reverseForStarlark(entry)).isEqualTo("//foo:bar=baz");
  }

  @Test
  public void labelToStringEntryConverter_roundTrip() throws Exception {
    String input = "//foo:bar=baz";
    Map.Entry<Label, String> converted =
        labelToStringEntryConverter.convert(input, /* conversionContext= */ null);
    assertThat(labelToStringEntryConverter.reverseForStarlark(converted)).isEqualTo(input);
  }

  @Test
  public void labelToStringEntryConverter_failure_noEquals() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> labelToStringEntryConverter.convert("//foo:bar", /* conversionContext= */ null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Variable definitions must be in the form of a 'name=value' assignment. 'name' and"
                + " 'value' must be non-empty and may not include '='.");
  }

  @Test
  public void labelToStringEntryConverter_failure_multipleEquals() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () ->
                labelToStringEntryConverter.convert(
                    "//foo:bar=baz=quux", /* conversionContext= */ null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Variable definitions must be in the form of a 'name=value' assignment. 'name' and"
                + " 'value' must be non-empty and may not include '='.");
  }

  @Test
  public void labelToStringEntryConverter_failure_emptyKey() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> labelToStringEntryConverter.convert("=baz", /* conversionContext= */ null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Variable definitions must be in the form of a 'name=value' assignment. 'name' and"
                + " 'value' must be non-empty and may not include '='.");
  }

  @Test
  public void labelToStringEntryConverter_failure_emptyValue() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> labelToStringEntryConverter.convert("//foo:bar=", /* conversionContext= */ null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Variable definitions must be in the form of a 'name=value' assignment. 'name' and"
                + " 'value' must be non-empty and may not include '='.");
  }
}
