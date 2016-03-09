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

import com.google.common.base.MoreObjects;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.xml.ArrayXmlResourceValue;

import com.android.resources.ResourceType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
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
   * @param targetOverwritableResources A collection that will have overwritable
   *        {@link XmlDataResource}s added to it.
   * @param targetNonOverwritableResources A collection that will have nonoverwritable
   *        {@link XmlDataResource}s added to it.
   * @throws XMLStreamException Thrown with the resource format is invalid.
   * @throws FactoryConfigurationError Thrown with the {@link XMLInputFactory} is misconfigured.
   * @throws IOException Thrown when there is an error reading a file.
   */
  public static void fromPath(
      XMLInputFactory xmlInputFactory,
      Path path,
      Factory fqnFactory,
      List<DataResource> targetOverwritableResources,
      List<DataResource> targetNonOverwritableResources)
      throws XMLStreamException, FactoryConfigurationError, IOException {
    XMLEventReader eventReader =
        xmlInputFactory.createXMLEventReader(Files.newBufferedReader(path, StandardCharsets.UTF_8));
    // TODO(corysmith): Make the xml parsing more readable.
    while (XmlResourceValues.moveToResources(eventReader)) {
      for (StartElement start = XmlResourceValues.findNextStart(eventReader);
          start != null;
          start = XmlResourceValues.findNextStart(eventReader)) {
        parseXmlElements(
            path,
            fqnFactory,
            targetOverwritableResources,
            targetNonOverwritableResources,
            eventReader,
            start);
      }
    }
  }

  private static void parseXmlElements(
      Path path,
      Factory fqnFactory,
      List<DataResource> overwritable,
      List<DataResource> nonOverwritable,
      XMLEventReader eventReader,
      StartElement start)
      throws XMLStreamException {
    ResourceType resourceType = getResourceType(start);
    switch (resourceType) {
      case STYLE:
        overwritable.add(
            XmlDataResource.of(
                fqnFactory.create(ResourceType.STYLE, XmlResourceValues.getElementName(start)),
                path,
                XmlResourceValues.parseStyle(eventReader, start)));
        break;
      case DECLARE_STYLEABLE:
        // Styleables are special, as they produce multiple overwrite and non-overwrite values,
        // so we let the value handle the assignments.
        XmlResourceValues.parseDeclareStyleable(
            fqnFactory, path, nonOverwritable, overwritable, eventReader, start);
        break;
      case ARRAY:
        overwritable.add(
            XmlDataResource.of(
                fqnFactory.create(ResourceType.ARRAY, XmlResourceValues.getElementName(start)),
                path,
                ArrayXmlResourceValue.parseArray(eventReader, start)));
        break;
      case PLURALS:
        overwritable.add(
            XmlDataResource.of(
                fqnFactory.create(ResourceType.PLURALS, XmlResourceValues.getElementName(start)),
                path,
                XmlResourceValues.parsePlurals(eventReader)));
        break;
      case ATTR:
        overwritable.add(
            XmlDataResource.of(
                fqnFactory.create(ResourceType.ATTR, XmlResourceValues.getElementName(start)),
                path,
                XmlResourceValues.parseAttr(eventReader, start)));
        break;
      case ID:
        nonOverwritable.add(
            XmlDataResource.of(
                fqnFactory.create(ResourceType.ID, XmlResourceValues.getElementName(start)),
                path,
                XmlResourceValues.parseId()));
        break;
      case STRING:
      case BOOL:
      case COLOR:
      case DIMEN:
        overwritable.add(
            XmlDataResource.of(
                fqnFactory.create(resourceType, XmlResourceValues.getElementName(start)),
                path,
                XmlResourceValues.parseSimple(eventReader, resourceType)));
        break;
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

  private FullyQualifiedName fqn;
  private Path source;
  private XmlResourceValue xml;

  private XmlDataResource(FullyQualifiedName fqn, Path source, XmlResourceValue xmlValue) {
    this.fqn = fqn;
    this.source = source;
    this.xml = xmlValue;
  }

  public static XmlDataResource of(FullyQualifiedName fqn, Path source, XmlResourceValue xml) {
    return new XmlDataResource(fqn, source, xml);
  }

  @Override
  public int hashCode() {
    return Objects.hash(fqn, source, xml);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof XmlDataResource)) {
      return false;
    }
    XmlDataResource other = (XmlDataResource) obj;
    return Objects.equals(fqn, other.fqn)
        && Objects.equals(source, other.source)
        && Objects.equals(xml, other.xml);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("fqn", fqn)
        .add("source", source)
        .add("xml", xml)
        .toString();
  }

  @Override
  public void write(Path newResourceDirectory) throws IOException {
    // TODO(corysmith): Implement write.
    throw new UnsupportedOperationException();
  }
}
