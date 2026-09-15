// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.model;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Joiner;
import java.io.ByteArrayOutputStream;
import java.io.UnsupportedEncodingException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link XmlWriter}
 */
@RunWith(JUnit4.class)
public class XmlWriterTest {
  private static final Joiner LINE_JOINER = Joiner.on(XmlWriter.EOL);
  private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
  
  @Rule
  public final ExpectedException expectedException = ExpectedException.none();
  
  @Test
  public void encodingShouldBeUtf8() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("String");
    String utf8String = "z\u0080\u0800\u010000"; // 1+2+3+4 bytes
    xmlWriter.writeCharacters(utf8String);
    xmlWriter.close();

    // Note: assertHasContents() reads the bytes of the outputStream as a UTF-8 string
    assertHasContents(
        "<?xml version='1.0' encoding='UTF-8'?>", "<String>" + utf8String + "</String>");
  }

  @Test
  public void header() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.close();

    assertHasContents("<?xml version='1.0' encoding='UTF-8'?>");
  }

  @Test
  public void emptyDocument() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("DocumentName");
    xmlWriter.close();

    assertHasContents("<?xml version='1.0' encoding='UTF-8'?>",
        "<DocumentName />");
  }

  @Test
  public void emptyDocumentWithOneAttribute() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Properties");
    xmlWriter.writeAttribute("name", "value");
    xmlWriter.close();

    assertHasContents("<?xml version='1.0' encoding='UTF-8'?>",
        "<Properties name='value' />");
  }
 
  @Test
  public void emptyDocumentWithTwoAttributes() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("TestSuite");
    xmlWriter.writeAttribute("count", 7);
    xmlWriter.writeAttribute("size", "large");
    xmlWriter.close();

    assertHasContents("<?xml version='1.0' encoding='UTF-8'?>",
        "<TestSuite count='7' size='large' />");
  }

  @Test
  public void emptyDocumentWithThreeAttributes() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("TestSuite");
    xmlWriter.writeAttribute("count", 7);
    xmlWriter.writeAttribute("size", "large");
    xmlWriter.writeAttribute("time", 1.0);
    xmlWriter.close();

    assertHasContents("<?xml version='1.0' encoding='UTF-8'?>",
        "<TestSuite count='7' size='large' time='1.0' />");
  }

  @Test
  public void documentWithOneEmptyElement() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Root");
    xmlWriter.writeAttribute("childCount", 1);
    xmlWriter.startElement("Child");
    xmlWriter.close();

    assertHasContents(
        "<?xml version='1.0' encoding='UTF-8'?>", "<Root childCount='1'>", "  <Child /></Root>");
  }

  @Test
  public void documentWithOneEmptyElementWithAttribute() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Root");
    xmlWriter.startElement("Child");
    xmlWriter.writeAttribute("name", "value");
    xmlWriter.close();

    assertHasContents(
        "<?xml version='1.0' encoding='UTF-8'?>", "<Root>", "  <Child name='value' /></Root>");
  }
  
  @Test
  public void documentWithOneElementWithCharactersNoEscaping()
      throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Root");
    xmlWriter.startElement("Child");
    xmlWriter.writeCharacters("some text\nmore text");
    xmlWriter.close();

    assertHasContents(
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<Root>",
        "  <Child>some text",
        "more text</Child></Root>");
  }

  @Test
  public void documentWithOneElementWithCharactersNeedingEscaping()
      throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Root");
    xmlWriter.startElement("Child");
    xmlWriter.writeCharacters("foo]]>bar");
    xmlWriter.close();

    assertHasContents(
        "<?xml version='1.0' encoding='UTF-8'?>", "<Root>", "  <Child>foo]]&gt;bar</Child></Root>");
  }
  
  @Test
  public void documentWithOneElementChild() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Root");
    xmlWriter.startElement("Child");
    xmlWriter.writeAttribute("name", "value");
    xmlWriter.startElement("Grandchild");
    xmlWriter.close();

    assertHasContents(
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<Root>",
        "  <Child name='value'>",
        "    <Grandchild /></Child></Root>");
  }

  @Test
  public void documentWithTwoElements() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Parent");
    xmlWriter.startElement("Child");
    xmlWriter.writeAttribute("name", "Deanna");
    xmlWriter.endElement();
    xmlWriter.startElement("Child");
    xmlWriter.writeAttribute("name", "Kyle");
    xmlWriter.close();

    assertHasContents(
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<Parent>",
        "  <Child name='Deanna' />",
        "  <Child name='Kyle' /></Parent>");
  }

  @Test
  public void attributeValuesEscaped() throws Exception {
    XmlWriter xmlWriter = new XmlWriter(outputStream);
    xmlWriter.startDocument();
    xmlWriter.startElement("Expression");
    xmlWriter.writeAttribute("name", "a > b");
    xmlWriter.close();

    assertHasContents("<?xml version='1.0' encoding='UTF-8'?>",
        "<Expression name='a &gt; b' />");
  }
  
  private void assertHasContents(String... contents) throws UnsupportedEncodingException {
    Object[] expected = contents;

    assertThat(outputStream.toString("UTF-8").trim()).isEqualTo(LINE_JOINER.join(expected).trim());
  }
}
