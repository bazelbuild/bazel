// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.android.resources.ResourceType.DECLARE_STYLEABLE;
import static com.android.resources.ResourceType.ID;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.android.AndroidDataSet.KeyValueConsumer;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.xml.ArrayXmlResourceValue;

import com.android.resources.ResourceType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

import javax.xml.stream.FactoryConfigurationError;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.StartElement;

/**
 * Represents an Android Resource defined in the xml and value folder.
 *
 * <p>Basically, if the resource is defined inside a &lt;resources&gt; tag, this class will
 * handle it. Layouts are treated separately as they don't declare anything besides ids.
 */
public class XmlDataResource implements DataResource {

  /**
   * Parses xml resources from a Path to the provided overwritable and nonOverwritable collections.
   *
   * This method is a bit tricky in the service of performance -- creating several collections and
   * merging them was more expensive than writing to mutable collections directly.
   *
   * @param xmlInputFactory Used to create an XMLEventReader from the supplied resource path.
   * @param path The path to the xml resource to be parsed.
   * @param fqnFactory Used to create {@link FullyQualifiedName}s from the resource names.
   * @param overwritingConsumer A consumer for overwritable {@link XmlDataResource}s.
   * @param nonOverwritingConsumer  A consumer for nonoverwritable {@link XmlDataResource}s.
   * @throws XMLStreamException Thrown with the resource format is invalid.
   * @throws FactoryConfigurationError Thrown with the {@link XMLInputFactory} is misconfigured.
   * @throws IOException Thrown when there is an error reading a file.
   */
  public static void fromPath(
      XMLInputFactory xmlInputFactory,
      Path path,
      Factory fqnFactory,
      KeyValueConsumer<DataKey, DataResource> overwritingConsumer,
      KeyValueConsumer<DataKey, DataResource> nonOverwritingConsumer)
      throws XMLStreamException, FactoryConfigurationError, IOException {
    XMLEventReader eventReader =
        xmlInputFactory.createXMLEventReader(Files.newBufferedReader(path, StandardCharsets.UTF_8));
    // TODO(corysmith): Make the xml parsing more readable.
    while (XmlResourceValues.moveToResources(eventReader)) {
      for (StartElement start = XmlResourceValues.findNextStart(eventReader);
          start != null;
          start = XmlResourceValues.findNextStart(eventReader)) {
        ResourceType resourceType = getResourceType(start);
        if (resourceType == DECLARE_STYLEABLE) {
          // Styleables are special, as they produce multiple overwrite and non-overwrite values,
          // so we let the value handle the assignments.
          XmlResourceValues.parseDeclareStyleable(
              fqnFactory, path, overwritingConsumer, nonOverwritingConsumer, eventReader, start);
        } else {
          // Of simple resources, only IDs are nonOverwriting.
          KeyValueConsumer<DataKey, DataResource> consumer =
              resourceType == ID ? nonOverwritingConsumer : overwritingConsumer;
          FullyQualifiedName key =
              fqnFactory.create(resourceType, XmlResourceValues.getElementName(start));
          consumer.consume(
              key, XmlDataResource.of(path, parseXmlElements(resourceType, eventReader, start)));
        }
      }
    }
  }

  private static XmlResourceValue parseXmlElements(
      ResourceType resourceType, XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    switch (resourceType) {
      case STYLE:
        return XmlResourceValues.parseStyle(eventReader, start);
      case ARRAY:
        return ArrayXmlResourceValue.parseArray(eventReader, start);
      case PLURALS:
        return XmlResourceValues.parsePlurals(eventReader);
      case ATTR:
        return XmlResourceValues.parseAttr(eventReader, start);
      case ID:
        return XmlResourceValues.parseId();
      case STRING:
      case BOOL:
      case COLOR:
      case DIMEN:
        return XmlResourceValues.parseSimple(eventReader, resourceType);
      default:
        throw new XMLStreamException(
            String.format("Unhandled resourceType %s", resourceType), start.getLocation());
    }
  }

  private static ResourceType getResourceType(StartElement start) {
    if (XmlResourceValues.isItem(start)) {
      return ResourceType.getEnum(XmlResourceValues.getElementType(start));
    }
    return ResourceType.getEnum(start.getName().getLocalPart());
  }

  private Path source;
  private XmlResourceValue xml;

  private XmlDataResource(Path source, XmlResourceValue xmlValue) {
    this.source = source;
    this.xml = xmlValue;
  }

  public static XmlDataResource of(Path source, XmlResourceValue xml) {
    return new XmlDataResource(source, xml);
  }

  @Override
  public Path source() {
    return source;
  }

  @Override
  public int hashCode() {
    return Objects.hash(source, xml);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof XmlDataResource)) {
      return false;
    }
    XmlDataResource other = (XmlDataResource) obj;
    return Objects.equals(source, other.source) && Objects.equals(xml, other.xml);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("source", source).add("xml", xml).toString();
  }

  @Override
  public void write(Path newResourceDirectory) throws IOException {
    // TODO(corysmith): Implement write.
    throw new UnsupportedOperationException();
  }
}
