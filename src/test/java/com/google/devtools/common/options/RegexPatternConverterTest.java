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
package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.common.options.Converters.RegexPatternConverter;
import com.google.devtools.common.options.testing.ConverterTester;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/** A test for {@link RegexPatternConverter} */
@RunWith(TestParameterInjector.class)
public class RegexPatternConverterTest {
  @Test
  public void consistentEqualsAndHashCodeForSamePattern() {
    new ConverterTester(RegexPatternConverter.class, /* conversionContext= */ null)
        .addEqualityGroup("")
        .addEqualityGroup(".*")
        .addEqualityGroup("[^\\s]+")
        .testConvert();
  }

  @Test
  public void comparisonBasedOnInputOnly() {
    String regex = "a";
    String semanticallyTheSame = "[a]";

    new ConverterTester(RegexPatternConverter.class, /* conversionContext= */ null)
        .addEqualityGroup(regex)
        .addEqualityGroup(semanticallyTheSame)
        .testConvert();
  }

  @Test
  public void createsProperPattern() throws OptionsParsingException {
    RegexPatternConverter converter = new RegexPatternConverter();
    for (String regex : new String[] {"", ".*", "\\s*(\\w+)", "prefix (suffix1|suffix2)"}) {
      // We are not testing {@link Pattern} itself -- the assumption is that if {@link
      // Pattern#pattern} returns the proper string, we created the right pattern.
      assertThat(converter.convert(regex).regexPattern().pattern()).isEqualTo(regex);
    }
  }

  @Test
  public void throwsForWrongPattern() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> new RegexPatternConverter().convert("{"));
    assertThat(e).hasMessageThat().startsWith("Not a valid regular expression:");
  }

  @Test
  public void unicodeLiteral() throws OptionsParsingException {
    // Options passed on the command line are passed to convertes in the internal encoding.
    var regex = new RegexPatternConverter().convert(StringEncoding.unicodeToInternal("äöüÄÖÜß🌱"));
    assertThat(regex.regexPattern().matcher("äöüÄÖÜß🌱").matches()).isTrue();
    assertThat(regex.matcher().test(StringEncoding.unicodeToInternal("äöüÄÖÜß🌱"))).isTrue();
  }

  @Test
  public void unicodeLiteral_caseInsensitive() throws OptionsParsingException {
    // Options passed on the command line are passed to convertes in the internal encoding.
    var regex =
        new RegexPatternConverter().convert(StringEncoding.unicodeToInternal("(?ui)äöüÄÖÜß🌱"));
    assertThat(regex.regexPattern().matcher("ÄÖÜäöüß🌱").matches()).isTrue();
    assertThat(regex.matcher().test(StringEncoding.unicodeToInternal("ÄÖÜäöüß🌱"))).isTrue();
  }

  @Test
  public void unicodeLiteral_suffix() throws OptionsParsingException {
    // Options passed on the command line are passed to convertes in the internal encoding.
    var regex =
        new RegexPatternConverter().convert(StringEncoding.unicodeToInternal(".*äöüÄÖÜß🌱"));
    assertThat(regex.regexPattern().matcher("äöüäöüÄÖÜß🌱").matches()).isTrue();
    assertThat(regex.matcher().test(StringEncoding.unicodeToInternal("äöüäöüÄÖÜß🌱"))).isTrue();
  }

  @Test
  public void unicodeClass() throws OptionsParsingException {
    // Options passed on the command line are passed to convertes in the internal encoding.
    var regex =
        new RegexPatternConverter()
            .convert(StringEncoding.unicodeToInternal("\\p{L}{7}\\p{IsEmoji}"));
    assertThat(regex.regexPattern().matcher("äöüÄÖÜß🌱").matches()).isTrue();
    assertThat(regex.matcher().test(StringEncoding.unicodeToInternal("äöüÄÖÜß🌱"))).isTrue();
  }
}
