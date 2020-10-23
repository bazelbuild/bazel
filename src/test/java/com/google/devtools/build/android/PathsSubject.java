// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.truth.Fact.fact;
import static com.google.common.truth.Fact.simpleFact;
import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.asList;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.devtools.build.android.aapt2.ResourceCompiler.CompiledType;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.Document;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

/** A testing utility that allows assertions against Paths. */
public class PathsSubject extends Subject {

  private static final String PATH_NORMALIZER =
      String.format(
          "(%s)/(.*/)(javatests)",
          Arrays.stream(CompiledType.values())
              .map(CompiledType::asPrefix)
              .map(p -> String.format("(?:%s)", p))
              .collect(joining("|")));

  private final Path actual;

  PathsSubject(FailureMetadata failureMetadata, @Nullable Path subject) {
    super(failureMetadata, subject);
    this.actual = subject;
  }

  void exists() {
    if (actual == null) {
      failWithoutActual(simpleFact("expected not to be null"));
    }
    if (!Files.exists(actual)) {
      failWithActual(simpleFact("expected to exist"));
    }
  }

  void containsExactlyUncompressedFilesIn(String... paths) throws IOException {
    if (actual == null) {
      failWithoutActual(simpleFact("expected not to be null"));
    }
    exists();
    assertThat(
            new ZipFile(actual.toFile())
                .stream()
                    .filter(entry -> entry.getMethod() == ZipEntry.STORED)
                    .map(ZipEntry::getName)
                    .map(n -> n.replaceAll(PATH_NORMALIZER, "$1/$3"))
                    .collect(Collectors.toSet()))
        .containsExactlyElementsIn(paths);
  }

  void containsAllArchivedFilesIn(String... paths) throws IOException {
    if (actual == null) {
      failWithoutActual(simpleFact("expected not to be null"));
    }
    exists();

    assertThat(
            new ZipFile(actual.toFile())
                .stream()
                    .map(ZipEntry::getName)
                    .map(n -> n.replaceAll(PATH_NORMALIZER, "$1/$3"))
                    .collect(Collectors.toSet()))
        .containsAtLeastElementsIn(paths);
  }

  void containsNoArchivedFilesIn(String... paths) throws IOException {
    if (actual == null) {
      failWithoutActual(simpleFact("expected not to be null"));
    }
    exists();
    assertThat(
            new ZipFile(actual.toFile())
                .stream()
                    .map(ZipEntry::getName)
                    .map(n -> n.replaceAll(PATH_NORMALIZER, "$1/$3"))
                    .collect(Collectors.toSet()))
        .containsNoneIn(Arrays.asList(paths));
  }

  void xmlContentsIsEqualTo(String... contents) {
    if (actual == null) {
      failWithoutActual(simpleFact("expected not to be null"));
    }
    exists();
    try {
      DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
      Transformer transformer = TransformerFactory.newInstance().newTransformer();
      transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
      transformer.setOutputProperty(OutputKeys.INDENT, "no");

      assertThat(
              normalizeXml(
                  newXmlDocument(factory, Files.readAllLines(actual, StandardCharsets.UTF_8)),
                  transformer))
          .isEqualTo(normalizeXml(newXmlDocument(factory, Arrays.asList(contents)), transformer));
    } catch (IOException | SAXException | ParserConfigurationException | TransformerException e) {
      failWithoutActual(
          fact("expected to have contents", asList(contents)),
          fact("but failed to read file with", e),
          fact("path was", actual));
    }
  }

  private Document newXmlDocument(DocumentBuilderFactory factory, List<String> contents)
      throws SAXException, IOException, ParserConfigurationException {
    return factory
        .newDocumentBuilder()
        .parse(new InputSource(new StringReader(Joiner.on("").join(contents))));
  }

  private String normalizeXml(Document doc, Transformer transformer) throws TransformerException {
    StringWriter writer = new StringWriter();
    transformer.transform(new DOMSource(doc), new StreamResult(writer));
    return writer.toString().replaceAll("\n|\r", "").replaceAll(">\\s+<", "><");
  }

  public void containsExactlyTheseLines(String... lines) throws IOException {
    assertThat(Files.readAllLines(actual, StandardCharsets.UTF_8)).containsExactlyElementsIn(lines);
  }
}
