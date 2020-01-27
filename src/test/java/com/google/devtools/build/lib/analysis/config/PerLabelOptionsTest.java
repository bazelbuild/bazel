// Copyright 2009 The Bazel Authors. All rights reserved.
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

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link PerLabelOptions}. */
@RunWith(JUnit4.class)
public class PerLabelOptionsTest {
  private PerLabelOptions options = null;

  private PerLabelOptions createOptions(String string) throws OptionsParsingException {
    options = new PerLabelOptions.PerLabelOptionsConverter().convert(string);
    return options;
  }

  private void assertRegexParsing(String filter) throws OptionsParsingException {
    RegexFilter regexFilter = new RegexFilter.RegexFilterConverter().convert(filter);
    assertThat(options.getRegexFilter().toString()).isEqualTo(regexFilter.toString());
  }

  private void assertOptions(String pattern, String opts, List<String> expectedOptions)
      throws OptionsParsingException {
    createOptions(pattern + "@" + opts);
    assertRegexParsing(pattern);
    assertThat(options.getOptions()).isNotNull();
    assertThat(options.getOptions()).isEqualTo(expectedOptions);
  }

  @Test
  public void testEmpty() throws Exception {
    createOptions("");
    assertRegexParsing("");
    assertThat(options.getOptions()).isEmpty();
  }

  @Test
  public void testParsing() throws Exception {
    assertOptions("", "", Collections.<String> emptyList());
    assertOptions("", ", ,\t,", Collections.<String> emptyList());
    assertOptions("a/b,+^c,_test$", ", ,\t,", Collections.<String> emptyList());
    assertOptions("a/b,+^c,_test$", "", Collections.<String> emptyList());
    assertOptions("a/b,+^c,_test$", "-g,-O0", Arrays.asList("-g", "-O0"));
    assertOptions("a/b,+^c,_test$", "-g@,-O0", Arrays.asList("-g@", "-O0"));
    assertOptions("a/b,+^c,_test$", "-g\\,,-O0", Arrays.asList("-g,", "-O0"));
    assertOptions("a/b,+^c,_test$", "-g\\,,,,,-O0,,,@,", Arrays.asList("-g,", "-O0", "@"));
  }

  @Test
  public void testEquals() throws Exception {
    new EqualsTester()
        .addEqualityGroup(createOptions("a/b,+^c,_test$@-g,-O0"),
            createOptions("a/b,+^c,_test$@-g,-O0"))
        .addEqualityGroup(createOptions("a/b,+^c,_test$@-O0"))
        .testEquals();
  }
}
