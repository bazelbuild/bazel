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
import com.google.common.base.Preconditions;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.ParsedAndroidData.KeyValueConsumer;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.xml.ArrayXmlResourceValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.IdXmlResourceValue;
import com.google.devtools.build.android.xml.PluralXmlResourceValue;
import com.google.devtools.build.android.xml.SimpleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import com.google.protobuf.InvalidProtocolBufferException;

import com.android.resources.ResourceType;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
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
 * <p>
 * Basically, if the resource is defined inside a &lt;resources&gt; tag, this class will handle it.
 * Layouts are treated separately as they don't declare anything besides ids.
 */
public class DataResourceXml implements DataResource {

  /**
   * Parses xml resources from a Path to the provided overwritable and combining collections.
   *
   * This method is a bit tricky in the service of performance -- creating several collections and
   * merging them was more expensive than writing to mutable collections directly.
   *
   * @param xmlInputFactory Used to create an XMLEventReader from the supplied resource path.
   * @param path The path to the xml resource to be parsed.
   * @param fqnFactory Used to create {@link FullyQualifiedName}s from the resource names.
   * @param overwritingConsumer A consumer for overwritable {@link DataResourceXml}s.
   * @param combiningConsumer A consumer for combining {@link DataResourceXml}s.
   * @throws XMLStreamException Thrown with the resource format is invalid.
   * @throws FactoryConfigurationError Thrown with the {@link XMLInputFactory} is misconfigured.
   * @throws IOException Thrown when there is an error reading a file.
   */
  public static void parse(
      XMLInputFactory xmlInputFactory,
      Path path,
      Factory fqnFactory,
      KeyValueConsumer<DataKey, DataResource> overwritingConsumer,
      KeyValueConsumer<DataKey, DataResource> combiningConsumer)
      throws XMLStreamException, FactoryConfigurationError, IOException {
    XMLEventReader eventReader =
        xmlInputFactory.createXMLEventReader(Files.newBufferedReader(path, StandardCharsets.UTF_8));
    try {
      // TODO(corysmith): Make the xml parsing more readable.
      while (XmlResourceValues.moveToResources(eventReader)) {
        for (StartElement start = XmlResourceValues.findNextStart(eventReader);
            start != null;
            start = XmlResourceValues.findNextStart(eventReader)) {
          if (XmlResourceValues.isEatComment(start) || XmlResourceValues.isSkip(start)) {
            continue;
          }
          ResourceType resourceType = getResourceType(start);
          if (resourceType == null) {
            throw new XMLStreamException(
                path + " contains an unrecognized resource type:" + start, start.getLocation());
          }
          if (resourceType == DECLARE_STYLEABLE) {
            // Styleables are special, as they produce multiple overwrite and combining values,
            // so we let the value handle the assignments.
            XmlResourceValues.parseDeclareStyleable(
                fqnFactory, path, overwritingConsumer, combiningConsumer, eventReader, start);
          } else {
            // Of simple resources, only IDs are combining.
            KeyValueConsumer<DataKey, DataResource> consumer =
                resourceType == ID ? combiningConsumer : overwritingConsumer;
            FullyQualifiedName key =
                fqnFactory.create(resourceType, XmlResourceValues.getElementName(start));
            consumer.consume(
                key, DataResourceXml.of(path, parseXmlElements(resourceType, eventReader, start)));
          }
        }
      }
    } catch (XMLStreamException e) {
      throw new XMLStreamException(path + ":" + e.getMessage(), e.getLocation(), e);
    } catch (RuntimeException e) {
      throw new RuntimeException("Error parsing " + path, e);
    }
  }

  public static DataValue from(SerializeFormat.DataValue protoValue, FileSystem currentFileSystem)
      throws InvalidProtocolBufferException {
    return of(
        currentFileSystem.getPath(protoValue.getSource().getFilename()),
        valueFromProto(protoValue.getXmlValue()));
  }

  private static XmlResourceValue valueFromProto(SerializeFormat.DataValueXml proto)
      throws InvalidProtocolBufferException {
    Preconditions.checkArgument(proto.hasType());
    switch (proto.getType()) {
      case ARRAY:
        return ArrayXmlResourceValue.from(proto);
      case SIMPLE:
        return SimpleXmlResourceValue.from(proto);
      case ATTR:
        return AttrXmlResourceValue.from(proto);
      case ID:
        return IdXmlResourceValue.of();
      case PLURAL:
        return PluralXmlResourceValue.from(proto);
      case STYLE:
        return StyleXmlResourceValue.from(proto);
      case STYLEABLE:
        return StyleableXmlResourceValue.from(proto);
      default:
        throw new IllegalArgumentException();
    }
  }

  private static XmlResourceValue parseXmlElements(
      ResourceType resourceType, XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    // Handle ids first, as they are a special kind of item.
    if (resourceType == ID) {
      return XmlResourceValues.parseId();
    }
    // Handle item stubs.
    if (XmlResourceValues.isItem(start)) {
      return XmlResourceValues.parseSimple(eventReader, resourceType, start); 
    }
    switch (resourceType) {
      case STYLE:
        return XmlResourceValues.parseStyle(eventReader, start);
      case ARRAY:
        return ArrayXmlResourceValue.parseArray(eventReader, start);
      case PLURALS:
        return XmlResourceValues.parsePlurals(eventReader);
      case ATTR:
        return XmlResourceValues.parseAttr(eventReader, start);
      case LAYOUT:
      case DIMEN:
      case STRING:
      case BOOL:
      case COLOR:
      case FRACTION:
      case INTEGER:
      case DRAWABLE:
      case ANIM:
      case ANIMATOR:
      case DECLARE_STYLEABLE:
      case INTERPOLATOR:
      case MENU:
      case MIPMAP:
      case PUBLIC:
      case RAW:
      case STYLEABLE:
      case TRANSITION:
      case XML:
        return XmlResourceValues.parseSimple(eventReader, resourceType, start);
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

  private DataResourceXml(Path source, XmlResourceValue xmlValue) {
    this.source = source;
    this.xml = xmlValue;
  }

  public static DataResourceXml of(Path source, XmlResourceValue xml) {
    return new DataResourceXml(source, xml);
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
    if (!(obj instanceof DataResourceXml)) {
      return false;
    }
    DataResourceXml other = (DataResourceXml) obj;
    return Objects.equals(source, other.source) && Objects.equals(xml, other.xml);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("source", source).add("xml", xml).toString();
  }

  @Override
  public void writeResource(FullyQualifiedName key, AndroidDataWritingVisitor mergedDataWriter) {
    xml.write(key, source, mergedDataWriter);
  }

  @Override
  public int serializeTo(DataKey key, OutputStream outStream) throws IOException {
    return xml.serializeTo(source, outStream);
  }

  // TODO(corysmith): Clean up all the casting. The type structure is unclean.
  @Override
  public DataResource combineWith(DataResource resource) {
    if (!(resource instanceof DataResourceXml)) {
      throw new IllegalArgumentException(resource + " is not a combinable with " + this);
    }
    DataResourceXml xmlResource = (DataResourceXml) resource;
    // TODO(corysmith): Combine the sources so that we know both of the originating files.
    // For right now, use the current source.
    return of(source, xml.combineWith(xmlResource.xml));
  }
}
