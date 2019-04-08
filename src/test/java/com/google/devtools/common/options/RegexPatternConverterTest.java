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
import static org.junit.Assert.fail;

import com.google.devtools.common.options.Converters.RegexPatternConverter;
import com.google.devtools.common.options.testing.ConverterTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link RegexPatternConverter} */
@RunWith(JUnit4.class)
public class RegexPatternConverterTest {
  @Test
  public void consistentEqualsAndHashCodeForSamePattern() {
    new ConverterTester(RegexPatternConverter.class)
        .addEqualityGroup("")
        .addEqualityGroup(".*")
        .addEqualityGroup("[^\\s]+")
        .testConvert();
  }

  @Test
  public void comparisonBasedOnInputOnly() {
    String regex = "a";
    String semanticallyTheSame = "[a]";

    new ConverterTester(RegexPatternConverter.class)
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
    try {
      new RegexPatternConverter().convert("{");
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e).hasMessageThat().startsWith("Not a valid regular expression:");
    }
  }
}
