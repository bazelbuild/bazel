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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import java.io.ByteArrayInputStream;
import javax.xml.stream.XMLStreamException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests to verify that DTD processing and external entity resolution are disabled in {@link
 * TestXmlOutputParser}.
 */
@RunWith(JUnit4.class)
public final class TestXmlOutputParserXxeTest {

  /**
   * Verifies that entity expansion is not performed because DTD declarations are ignored, causing
   * references to DTD-defined entities to throw an exception.
   */
  @Test
  public void entityReference_throwsExceptionBecauseDtdIsIgnored() throws Exception {
    String xml =
        """
        <?xml version="1.0"?>
        <!DOCTYPE testsuites [
          <!ENTITY c "expanded_value">
        ]>
        <testsuites name="s"><testcase name="&c;" time="0"/></testsuites>
        """;
    TestXmlOutputParserException e =
        assertThrows(TestXmlOutputParserException.class, () -> parse(xml));
    assertThat(e).hasCauseThat().isInstanceOf(XMLStreamException.class);
    assertThat(e)
        .hasCauseThat()
        .hasMessageThat()
        .contains("entity \"c\" was referenced, but not declared");
  }

  /**
   * Verifies that external DTD references (e.g. SYSTEM "http://...") are ignored and do not cause
   * parsing failures or network lookups.
   */
  @Test
  public void externalDtdHttp_ignored_noSsrf() throws Exception {
    String xml =
        """
        <?xml version="1.0"?>
        <!DOCTYPE testsuites SYSTEM "http://127.0.0.1:1/evil.dtd">
        <testsuites name="s"><testcase name="n" time="0"/></testsuites>
        """;
    TestCase root = parse(xml);
    assertThat(root.getChild(0).getName()).isEqualTo("n");
  }

  private static TestCase parse(String xml) throws TestXmlOutputParserException {
    return new TestXmlOutputParser()
        .parseXmlIntoTestResult(new ByteArrayInputStream(xml.getBytes(UTF_8)));
  }
}
