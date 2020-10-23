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
import static com.android.resources.ResourceType.PUBLIC;

import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.Value;
import com.android.resources.ResourceType;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.AndroidCompiledDataDeserializer.ReferenceResolver;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.FullyQualifiedName.VirtualType;
import com.google.devtools.build.android.ParsedAndroidData.KeyValueConsumer;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml;
import com.google.devtools.build.android.resources.Visibility;
import com.google.devtools.build.android.xml.ArrayXmlResourceValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.IdXmlResourceValue;
import com.google.devtools.build.android.xml.Namespaces;
import com.google.devtools.build.android.xml.PluralXmlResourceValue;
import com.google.devtools.build.android.xml.PublicXmlResourceValue;
import com.google.devtools.build.android.xml.ResourcesAttribute;
import com.google.devtools.build.android.xml.SimpleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.Objects;
import javax.xml.stream.FactoryConfigurationError;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;

/**
 * Represents an Android Resource defined in the xml and value folder.
 *
 * <p>Basically, if the resource is defined inside a &lt;resources&gt; tag, this class will handle
 * it. Layouts are treated separately as they don't declare anything besides ids.
 */
public class DataResourceXml implements DataResource {

  /**
   * Parses xml resources from a Path to the provided overwritable and combining collections.
   *
   * <p>This method is a bit tricky in the service of performance -- creating several collections
   * and merging them was more expensive than writing to mutable collections directly.
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
        xmlInputFactory.createXMLEventReader(
            new BufferedInputStream(Files.newInputStream(path)), StandardCharsets.UTF_8.toString());
    try {
      // TODO(corysmith): Make the xml parsing more readable.
      for (StartElement resources = XmlResourceValues.moveToResources(eventReader);
          resources != null;
          resources = XmlResourceValues.moveToResources(eventReader)) {
        // Record attributes on the <resources> tag.
        Iterator<Attribute> attributes = XmlResourceValues.iterateAttributesFrom(resources);
        while (attributes.hasNext()) {
          Attribute attribute = attributes.next();
          Namespaces namespaces = Namespaces.from(attribute.getName());
          String attributeName =
              attribute.getName().getNamespaceURI().isEmpty()
                  ? attribute.getName().getLocalPart()
                  // This is intentionally putting the "package" in the wrong place!
                  // TODO: FullyQualifiedName.Factory#create has a overload which accepts "package".
                  : attribute.getName().getPrefix() + ":" + attribute.getName().getLocalPart();
          FullyQualifiedName fqn =
              fqnFactory.create(VirtualType.RESOURCES_ATTRIBUTE, attribute.getName().toString());
          ResourcesAttribute resourceAttribute =
              ResourcesAttribute.of(fqn, attributeName, attribute.getValue());
          DataResourceXml resource =
              DataResourceXml.createWithNamespaces(path, resourceAttribute, namespaces);
          if (resourceAttribute.isCombining()) {
            combiningConsumer.accept(fqn, resource);
          } else {
            overwritingConsumer.accept(fqn, resource);
          }
        }
        // Process resource declarations.
        for (StartElement start = XmlResourceValues.findNextStart(eventReader);
            start != null;
            start = XmlResourceValues.findNextStart(eventReader)) {
          Namespaces.Collector namespacesCollector = Namespaces.collector();
          if (XmlResourceValues.isEatComment(start) || XmlResourceValues.isSkip(start)) {
            continue;
          }
          ResourceType resourceType = getResourceType(start);
          if (resourceType == null) {
            throw new XMLStreamException(
                path + " contains an unrecognized resource type: " + start, start.getLocation());
          }
          if (resourceType == DECLARE_STYLEABLE) {
            // Styleables are special, as they produce multiple overwrite and combining values,
            // so we let the value handle the assignments.
            XmlResourceValues.parseDeclareStyleable(
                fqnFactory, path, overwritingConsumer, combiningConsumer, eventReader, start);
          } else {
            // Of simple resources, only IDs and Public are combining.
            KeyValueConsumer<DataKey, DataResource> consumer =
                (resourceType == ID || resourceType == PUBLIC)
                    ? combiningConsumer
                    : overwritingConsumer;
            String elementName = XmlResourceValues.getElementName(start);
            if (elementName == null) {
              throw new XMLStreamException(
                  String.format("resource name is required for %s", resourceType),
                  start.getLocation());
            }
            FullyQualifiedName key = fqnFactory.create(resourceType, elementName);
            XmlResourceValue xmlResourceValue =
                parseXmlElements(resourceType, eventReader, start, namespacesCollector);
            consumer.accept(
                key,
                DataResourceXml.createWithNamespaces(
                    path, xmlResourceValue, namespacesCollector.toNamespaces()));
          }
        }
      }
    } catch (XMLStreamException e) {
      throw new XMLStreamException(path + ": " + e.getMessage(), e.getLocation(), e);
    } catch (RuntimeException e) {
      throw new RuntimeException("Error parsing " + path, e);
    }
  }

  @SuppressWarnings("deprecation")
  // TODO(corysmith): Update proto to use get<>Map
  public static DataValue from(SerializeFormat.DataValue protoValue, DataSource source)
      throws InvalidProtocolBufferException {
    DataValueXml xmlValue = protoValue.getXmlValue();
    return createWithNamespaces(
        source, valueFromProto(xmlValue), Namespaces.from(xmlValue.getNamespaceMap()));
  }

  public static DataResourceXml from(
      Value protoValue,
      Visibility visibility,
      DataSource source,
      ResourceType resourceType,
      ReferenceResolver packageResolver)
      throws InvalidProtocolBufferException {
    DataResourceXml dataResourceXml =
        createWithNamespaces(
            source,
            valueFromProto(protoValue, visibility, resourceType, packageResolver),
            Namespaces.empty());
    return dataResourceXml;
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
      case PUBLIC:
        return PublicXmlResourceValue.from(proto);
      case STYLE:
        return StyleXmlResourceValue.from(proto);
      case STYLEABLE:
        return StyleableXmlResourceValue.from(proto);
      case RESOURCES_ATTRIBUTE:
        return ResourcesAttribute.from(proto);
      default:
        throw new IllegalArgumentException();
    }
  }

  private static XmlResourceValue valueFromProto(
      Value proto,
      Visibility visibility,
      ResourceType resourceType,
      ReferenceResolver packageResolver)
      throws InvalidProtocolBufferException {
    switch (resourceType) {
      case STYLE:
        return StyleXmlResourceValue.from(proto, visibility);
      case ARRAY:
        return ArrayXmlResourceValue.from(proto, visibility);
      case PLURALS:
        return PluralXmlResourceValue.from(proto, visibility);
      case ATTR:
        return AttrXmlResourceValue.from(proto, visibility);
      case STYLEABLE:
        return StyleableXmlResourceValue.from(proto, visibility, packageResolver);
      case ID:
        return IdXmlResourceValue.from(proto, visibility);
      case DIMEN:
      case LAYOUT:
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
      case NAVIGATION:
      case RAW:
      case TRANSITION:
      case FONT:
      case XML:
        return SimpleXmlResourceValue.from(proto, visibility, resourceType);
      default:
        throw new IllegalArgumentException("Unhandled type " + resourceType + " from " + proto);
    }
  }

  public static DataResourceXml fromPublic(DataSource source, ResourceType resourceType, int id) {
    DataResourceXml dataResourceXml =
        createWithNamespaces(
            source, PublicXmlResourceValue.from(resourceType, id), Namespaces.empty());
    return dataResourceXml;
  }

  private static XmlResourceValue parseXmlElements(
      ResourceType resourceType,
      XMLEventReader eventReader,
      StartElement start,
      Namespaces.Collector namespacesCollector)
      throws XMLStreamException {
    // Handle ids first, as they are a special kind of item.
    if (resourceType == ID) {
      return XmlResourceValues.parseId(eventReader, start, namespacesCollector);
    }
    // Handle item stubs.
    if (XmlResourceValues.isItem(start)) {
      return XmlResourceValues.parseSimple(eventReader, resourceType, start, namespacesCollector);
    }
    switch (resourceType) {
      case STYLE:
        return XmlResourceValues.parseStyle(eventReader, start);
      case ARRAY:
        return ArrayXmlResourceValue.parseArray(eventReader, start, namespacesCollector);
      case PLURALS:
        return XmlResourceValues.parsePlurals(eventReader, start, namespacesCollector);
      case ATTR:
        return XmlResourceValues.parseAttr(eventReader, start);
      case PUBLIC:
        return XmlResourceValues.parsePublic(eventReader, start, namespacesCollector);
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
      case RAW:
      case STYLEABLE:
      case TRANSITION:
      case XML:
        return XmlResourceValues.parseSimple(eventReader, resourceType, start, namespacesCollector);
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

  private final DataSource source;
  private final XmlResourceValue xml;
  private final Namespaces namespaces;

  private DataResourceXml(DataSource source, XmlResourceValue xmlValue, Namespaces namespaces) {
    this.source = source;
    this.xml = xmlValue;
    this.namespaces = namespaces;
  }

  public static DataResourceXml createWithNoNamespace(Path sourcePath, XmlResourceValue xml) {
    return createWithNamespaces(sourcePath, xml, ImmutableMap.<String, String>of());
  }

  public static DataResourceXml createWithNoNamespace(DataSource source, XmlResourceValue xml) {
    return new DataResourceXml(source, xml, Namespaces.empty());
  }

  public static DataResourceXml createWithNamespaces(
      Path sourcePath, XmlResourceValue xml, ImmutableMap<String, String> prefixToUri) {
    return createWithNamespaces(sourcePath, xml, Namespaces.from(prefixToUri));
  }

  public static DataResourceXml createWithNamespaces(
      DataSource source, XmlResourceValue xml, Namespaces namespaces) {
    return new DataResourceXml(source, xml, namespaces);
  }

  public static DataResourceXml createWithNamespaces(
      Path sourcePath, XmlResourceValue xml, Namespaces namespaces) {
    return createWithNamespaces(DataSource.of(DependencyInfo.UNKNOWN, sourcePath), xml, namespaces);
  }

  @Override
  public DataSource source() {
    return source;
  }

  @Override
  public int hashCode() {
    return Objects.hash(source, xml, namespaces);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof DataResourceXml)) {
      return false;
    }
    DataResourceXml other = (DataResourceXml) obj;
    return Objects.equals(source, other.source)
        && Objects.equals(xml, other.xml)
        && Objects.equals(namespaces, other.namespaces);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("source", source)
        .add("xml", xml)
        .add("namespaces", namespaces)
        .toString();
  }

  @Override
  public void writeResource(FullyQualifiedName key, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.defineNamespacesFor(key, namespaces);
    xml.write(key, source, mergedDataWriter);
  }

  @Override
  public void writeResourceToClass(FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    xml.writeResourceToClass(source().getDependencyInfo(), key, sink);
  }

  @Override
  public int serializeTo(DataSourceTable sourceTable, OutputStream outStream)
      throws IOException {
    return xml.serializeTo(sourceTable.getSourceId(source), namespaces, outStream);
  }

  // TODO(corysmith): Clean up all the casting. The type structure is unclean.
  @Override
  public DataResource combineWith(DataResource resource) {
    if (!(resource instanceof DataResourceXml)) {
      throw new IllegalArgumentException(resource + " is not combinable with " + this);
    }
    DataResourceXml xmlResource = (DataResourceXml) resource;
    return createWithNamespaces(
        source.combine(xmlResource.source),
        xml.combineWith(xmlResource.xml),
        namespaces.union(xmlResource.namespaces));
  }

  @Override
  public DataResource overwrite(DataResource resource) {
    if (equals(resource)) {
      return this;
    }
    return createWithNamespaces(source.overwrite(resource.source()), xml, namespaces);
  }

  @Override
  public DataValue update(DataSource source) {
    return createWithNamespaces(source, xml, namespaces);
  }

  @Override
  public String asConflictString() {
    return xml.asConflictStringWith(source);
  }

  @Override
  public boolean valueEquals(DataValue value) {
    if (!(value instanceof DataResourceXml)) {
      return false;
    }
    DataResourceXml other = (DataResourceXml) value;
    return Objects.equals(xml, other.xml);
  }

  @Override
  public int compareMergePriorityTo(DataValue value) {
    Preconditions.checkNotNull(value);
    if (!(value instanceof DataResourceXml)) {
      // This is an ambiguous conflict; return 0 meaning neither has priority.
      return 0;
    }
    DataResourceXml other = (DataResourceXml) value;
    return xml.compareMergePriorityTo(other.xml);
  }

  @Override
  public Visibility getVisibility() {
    return xml.getVisibility();
  }

  @Override
  public ImmutableList<Reference> getReferencedResources() {
    return xml.getReferencedResources();
  }
}
