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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.xml.XmlEscapers;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.StringWriter;
import java.io.Writer;
import java.util.List;

/**
 * Writer for XML documents. We do not use third-party code, because all
 * java_test rules have the test runner in their run-time classpath.
 */
class XmlWriter {
  @VisibleForTesting
  static final String EOL = System.getProperty("line.separator", "\n");
  
  private final Writer writer;
  private boolean started;
  private boolean inElement;
  private final List<String> elementStack = Lists.newArrayList();

  /**
   * Creates an XML writer that writes to the given {@code OutputStream}.
   *
   * @param outputStream stream to write to
   */
  public XmlWriter(OutputStream outputStream) {
    this(new OutputStreamWriter(outputStream, UTF_8));
  }

  /**
   * Creates an XML writer for testing purposes. Note that if you decide to
   * serialize the {@code StringWriter} (to disk or network) encode it in {@code
   * UTF-8}.
   *
   * @param writer
   */
  @VisibleForTesting
  static XmlWriter createForTesting(StringWriter writer) {
    return new XmlWriter(writer);
  }
  
  private XmlWriter(Writer writer) {
    this.writer = writer;
  }

  /**
   * Starts the XML document.
   *
   * @throws IOException if the underlying writer throws an exception
   */
  public void startDocument() throws IOException {
    Preconditions.checkState(!started, "already started");

    started = true;
    Writer out = writer;
    out.write("<?xml version='1.0' encoding='UTF-8'?>");
  }

  /**
   * Completes the XML document and closes the underlying writer.
   *
   * @throws IOException if the underlying writer throws an exception
   */
  public void close() throws IOException {
    while (!elementStack.isEmpty()) {
      endElement();
    }
    writer.append(EOL);
    writer.close();
  }

  private void closeElement() throws IOException {
    if (inElement) {
      writer.append('>');
      inElement = false;
    }
  }
  
  private String indentation() {
    return Strings.repeat("  ", elementStack.size());
  }

  /**
   * Starts an XML element. The element is left open until either
   * {@link #endElement()} or {@link #close()} are called. This method may be
   * called multiple times before calling {@link #endElement()}; the writer
   * keeps a stack of currently open elements.
   * 
   * @param elementName name of the element (must be XML safe or escaped)
   * @throws IOException if the underlying writer throws an exception
   */
  public void startElement(String elementName) throws IOException {
    Preconditions.checkState(started);
    closeElement();
    inElement = true;
    writer.append(EOL + indentation() + "<" + elementName);
    elementStack.add(elementName);
  }

  /**
   * Ends the current XML element.
   *
   * @throws IOException if the underlying writer throws an exception
   */
  public void endElement() throws IOException {
    String elementName = elementStack.remove(elementStack.size() - 1);
    if (inElement) {
      writer.write(" />");
      inElement = false;
    } else {
      writer.write(EOL + indentation() + "</");
      writer.write(elementName);
      writer.write('>');
    }
  }

  /**
   * Writes an attribute with the given integer value to the currently open XML
   * element.
   * 
   * @param name attribute name
   * @param value attribute value
   * @throws IOException
   */
  public void writeAttribute(String name, int value) throws IOException {
    writeAttributeWithoutEscaping(name, String.valueOf(value));
  }

  /**
   * Writes an attribute with the given double value to the currently open XML
   * element.
   * 
   * @param name attribute name
   * @param value attribute value (must be XML safe or escaped)
   * @throws IOException
   */
  public void writeAttribute(String name, double value) throws IOException {
    writeAttributeWithoutEscaping(name, String.valueOf(value));
  }

  /**
   * Writes an attribute to the currently open XML element.
   *
   * @param name attribute name (must be XML safe or escaped)
   * @param value attribute value (will be escaped by this method)
   * @throws IOException
   */
  public void writeAttribute(String name, String value) throws IOException {
    if (value != null) {
      value = XmlEscapers.xmlAttributeEscaper().escape(value);
    }
    writeAttributeWithoutEscaping(name, value);
  }

  private void writeAttributeWithoutEscaping(String name, String value) throws IOException {
    writer.write(" " + name + "='");
    if (value != null) {
      writer.write(value);
    }
    writer.write("'");
  }

  /**
   * Writes the given characters as the content of the element. Closes the
   * element if it is currently open.
   *
   * @param text String to append to the current content of the element
   * @throws IOException
   */
  public void writeCharacters(String text) throws IOException {
    if (Strings.isNullOrEmpty(text)) {
      return;
    }

    closeElement();
    writer.write(XmlEscapers.xmlContentEscaper().escape(text));
  }

  /**
   * Gets the writer that this object uses for writing.
   */
  @VisibleForTesting
  Writer getUnderlyingWriter() {
    return writer;
  }
}
