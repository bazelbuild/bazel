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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.ParsedAndroidData.KeyValueConsumer;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.IdXmlResourceValue;
import com.google.devtools.build.android.xml.PluralXmlResourceValue;
import com.google.devtools.build.android.xml.SimpleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import com.google.protobuf.CodedOutputStream;

import com.android.resources.ResourceType;

import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import javax.annotation.Nullable;
import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventFactory;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLEventWriter;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.Characters;
import javax.xml.stream.events.EndElement;
import javax.xml.stream.events.Namespace;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/**
 * {@link XmlResourceValues} provides methods for getting {@link XmlResourceValue} derived classes.
 *
 * <p>Acts a static factory class containing the general xml parsing logic for resources that are
 * declared inside the &lt;resources&gt; tag.
 */
public class XmlResourceValues {

  private static final Logger logger = Logger.getLogger(XmlResourceValues.class.getCanonicalName());

  private static final QName TAG_EAT_COMMENT = QName.valueOf("eat-comment");
  private static final QName TAG_PLURALS = QName.valueOf("plurals");
  private static final QName ATTR_QUANTITY = QName.valueOf("quantity");
  private static final QName TAG_ATTR = QName.valueOf("attr");
  private static final QName TAG_DECLARE_STYLEABLE = QName.valueOf("declare-styleable");
  private static final QName TAG_ITEM = QName.valueOf("item");
  private static final QName TAG_STYLE = QName.valueOf("style");
  private static final QName TAG_SKIP = QName.valueOf("skip");
  private static final QName TAG_RESOURCES = QName.valueOf("resources");
  private static final QName ATTR_FORMAT = QName.valueOf("format");
  private static final QName ATTR_NAME = QName.valueOf("name");
  private static final QName ATTR_VALUE = QName.valueOf("value");
  private static final QName ATTR_PARENT = QName.valueOf("parent");
  private static final QName ATTR_TYPE = QName.valueOf("type");

  private static final XMLOutputFactory XML_OUTPUT_FACTORY = XMLOutputFactory.newInstance();
  private static final XMLEventFactory XML_EVENT_FACTORY = XMLEventFactory.newInstance();

  static final String XLIFF_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2";
  static final String XLIFF_PREFIX = "xliff";

  static XmlResourceValue parsePlurals(XMLEventReader eventReader) throws XMLStreamException {
    ImmutableMap.Builder<String, String> values = ImmutableMap.builder();
    for (XMLEvent element = nextTag(eventReader);
        !isEndTag(element, TAG_PLURALS);
        element = nextTag(eventReader)) {
      if (isItem(element)) {
        if (!element.isStartElement()) {
          throw new XMLStreamException(
              String.format("Expected start element %s", element), element.getLocation());
        }
        String contents = readContentsAsString(eventReader, element.asStartElement().getName());
        values.put(
            getElementAttributeByName(element.asStartElement(), ATTR_QUANTITY),
            contents == null ? "" : contents);
      }
    }
    return PluralXmlResourceValue.of(values.build());
  }

  static XmlResourceValue parseStyle(XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    Map<String, String> values = new HashMap<>();
    for (XMLEvent element = nextTag(eventReader);
        !isEndTag(element, TAG_STYLE);
        element = nextTag(eventReader)) {
      if (isItem(element)) {
        values.put(getElementName(element.asStartElement()), eventReader.getElementText());
      }
    }
    // Parents can be declared as:
    //   ?style/parent
    //   @style/parent
    //   <Parent>
    // And, in the resource name <parent>.<resource name>
    // Here, we take a garbage in, garbage out approach and just read the xml value raw.
    return StyleXmlResourceValue.of(getElementAttributeByName(start, ATTR_PARENT),
        values);
  }

  static void parseDeclareStyleable(
      FullyQualifiedName.Factory fqnFactory,
      Path path,
      KeyValueConsumer<DataKey, DataResource> overwritingConsumer,
      KeyValueConsumer<DataKey, DataResource> combiningConsumer,
      XMLEventReader eventReader,
      StartElement start)
      throws XMLStreamException {
    Map<FullyQualifiedName, Boolean> members = new LinkedHashMap<>();
    for (XMLEvent element = nextTag(eventReader);
        !isEndTag(element, TAG_DECLARE_STYLEABLE);
        element = nextTag(eventReader)) {
      if (isStartTag(element, TAG_ATTR)) {
        StartElement attr = element.asStartElement();
        FullyQualifiedName attrName = fqnFactory.create(ResourceType.ATTR, getElementName(attr));
        // If there is format and the next tag is a starting tag, treat it as an attr definition.
        // Without those, it will be an attr reference.
        if (XmlResourceValues.getElementAttributeByName(attr, ATTR_FORMAT) != null
            || (XmlResourceValues.peekNextTag(eventReader) != null
                && XmlResourceValues.peekNextTag(eventReader).isStartElement())) {
          overwritingConsumer.consume(
              attrName, DataResourceXml.of(path, parseAttr(eventReader, attr)));
          members.put(attrName, Boolean.TRUE);
        } else {
          members.put(attrName, Boolean.FALSE);
        }
      }
    }
    combiningConsumer.consume(
        fqnFactory.create(ResourceType.STYLEABLE, getElementName(start)),
        DataResourceXml.of(path, StyleableXmlResourceValue.of(members)));
  }

  static XmlResourceValue parseAttr(XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    XmlResourceValue value =
        AttrXmlResourceValue.from(
            start, getElementAttributeByName(start, ATTR_FORMAT), eventReader);
    return value;
  }

  static XmlResourceValue parseId(XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    if (XmlResourceValues.isEndTag(eventReader.peek(), start.getName())) {
      return IdXmlResourceValue.of();
    } else {
      return IdXmlResourceValue.of(readContentsAsString(eventReader, start.getName()));
    }
  }

  static XmlResourceValue parseSimple(
      XMLEventReader eventReader, ResourceType resourceType, StartElement start)
      throws XMLStreamException {
    String contents;
    // Check that the element is unary. If it is, the contents is null
    if (isEndTag(eventReader.peek(), start.getName())) {
      contents = null;
    } else {
      contents = readContentsAsString(eventReader, start.getName());
    }
    return SimpleXmlResourceValue.of(
        start.getName().equals(TAG_ITEM)
            ? SimpleXmlResourceValue.Type.ITEM
            : SimpleXmlResourceValue.Type.from(resourceType),
        ImmutableMap.copyOf(parseTagAttributes(start)),
        contents);
  }

  public static Map<String, String> parseTagAttributes(StartElement start) {
    // Using a map to deduplicate xmlns declarations on the attributes.
    Map<String, String> attributeMap = new LinkedHashMap<>();
    Iterator<Attribute> attributes = iterateAttributesFrom(start);
    while (attributes.hasNext()) {
      Attribute attribute = attributes.next();
      QName name = attribute.getName();
      // Name used as the resource key, so skip it here.
      if (ATTR_NAME.equals(name)) {
        continue;
      }
      String value = escapeXmlValues(attribute.getValue()).replace("\"", "&quot;");
      if (!name.getNamespaceURI().isEmpty()) {
        // xliff is always provided on resources to help with file size
        if (!(XLIFF_NAMESPACE.equals(name.getNamespaceURI())
            && XLIFF_PREFIX.equals(name.getPrefix()))) {
          // Declare the xmlns here, so that the written xml will be semantically correct,
          // if a bit verbose. This allows the resource keys to be written into a <resources>
          // tag without a global namespace.
          attributeMap.put("xmlns:" + name.getPrefix(), name.getNamespaceURI());
        }
        attributeMap.put(name.getPrefix() + ":" + attribute.getName().getLocalPart(), value);
      } else {
        attributeMap.put(attribute.getName().getLocalPart(), value);
      }
    }
    return attributeMap;
  }

  // TODO(corysmith): Replace this with real escaping system, preferably a performant high level xml
  //writing library. See AndroidDataWritingVisitor TODO.
  private static String escapeXmlValues(String data) {
    return data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
  }

  /**
   * Reads the xml events as a string until finding a closing tag.
   *
   * @param eventReader The current xml stream.
   * @param startTag The name of the tag to close on.
   * @return A xml escaped string representation of the xml stream
   */
  @Nullable
  public static String readContentsAsString(XMLEventReader eventReader, QName startTag)
      throws XMLStreamException {
    StringWriter contents = new StringWriter();
    XMLEventWriter writer = XML_OUTPUT_FACTORY.createXMLEventWriter(contents);
    while (!isEndTag(eventReader.peek(), startTag)) {
      XMLEvent xmlEvent = eventReader.nextEvent();
      if (xmlEvent.isStartElement()) {
        // TODO(corysmith): Replace this with a proper representation of the contents that can be
        // serialized and reconstructed appropriately without modification.
        StartElement start = xmlEvent.asStartElement();
        QName name = start.getName();
        // Lazy with the list for memory usage -- this code path happens a lot.
        List<Namespace> namespaces = maybeAddNamespace(name, null);
        Iterator<Attribute> attributes = iterateAttributesFrom(start);
        while (attributes.hasNext()) {
          Attribute attribute = attributes.next();
          namespaces = maybeAddNamespace(attribute.getName(), namespaces);
        }
        if (namespaces != null) {
          writer.add(
              XML_EVENT_FACTORY.createStartElement(
                  name, iterateAttributesFrom(start), namespaces.iterator()));
        } else {
          writer.add(xmlEvent);
        }
      } else {
        writer.add(xmlEvent);
      }
    }
    // Verify the end element.
    EndElement endElement = eventReader.nextEvent().asEndElement();
    Preconditions.checkArgument(endElement.getName().equals(startTag));
    return contents.toString();
  }

  @SuppressWarnings({
    "cast",
    "unchecked"
  }) // The interface returns Iterator, force casting based on documentation.
  private static Iterator<Attribute> iterateAttributesFrom(StartElement start) {
    return (Iterator<Attribute>) start.getAttributes();
  }

  private static List<Namespace> maybeAddNamespace(QName name, List<Namespace> namespaces) {
    // The xliff namespace is provided by default.
    if (name.getNamespaceURI() != null
        && !name.getNamespaceURI().isEmpty()
        && !(XLIFF_NAMESPACE.equals(name.getNamespaceURI())
            && XLIFF_PREFIX.equals(name.getPrefix()))) {
      if (namespaces == null) {
        namespaces = new ArrayList<>();
      }
      namespaces.add(XML_EVENT_FACTORY.createNamespace(name.getPrefix(), name.getNamespaceURI()));
    }
    return namespaces;
  }

  /* XML helper methods follow. */
  // TODO(corysmith): Move these to a wrapper class for XMLEventReader.

  @Nullable
  public static String getElementAttributeByName(StartElement element, QName name) {
    Attribute attribute = element.getAttributeByName(name);
    return attribute == null ? null : attribute.getValue();
  }

  public static String getElementValue(StartElement start) {
    return getElementAttributeByName(start, ATTR_VALUE);
  }

  public static String getElementName(StartElement start) {
    return getElementAttributeByName(start, ATTR_NAME);
  }

  public static String getElementType(StartElement start) {
    return getElementAttributeByName(start, ATTR_TYPE);
  }

  public static boolean isTag(XMLEvent event, QName subTagType) {
    if (event.isStartElement()) {
      return isStartTag(event, subTagType);
    }
    if (event.isEndElement()) {
      return isEndTag(event, subTagType);
    }
    return false;
  }

  public static boolean isItem(XMLEvent start) {
    return isTag(start, TAG_ITEM);
  }

  public static boolean isEndTag(XMLEvent event, QName subTagType) {
    if (event.isEndElement()) {
      return subTagType.equals(event.asEndElement().getName());
    }
    return false;
  }

  public static boolean isStartTag(XMLEvent event, QName subTagType) {
    if (event.isStartElement()) {
      return subTagType.equals(event.asStartElement().getName());
    }
    return false;
  }

  public static XMLEvent nextTag(XMLEventReader eventReader) throws XMLStreamException {
    while (eventReader.hasNext()
        && !(eventReader.peek().isEndElement() || eventReader.peek().isStartElement())) {
      XMLEvent nextEvent = eventReader.nextEvent();
      if (nextEvent.isCharacters() && !nextEvent.asCharacters().isIgnorableWhiteSpace()) {
        Characters characters = nextEvent.asCharacters();
        // TODO(corysmith): Turn into a warning with the Path is available to add to it.
        // This case is when unexpected characters are thrown into the xml. Best case, it's a
        // incorrect comment type...
        logger.fine(
            String.format(
                "Invalid characters [%s] found at %s",
                characters.getData(), characters.getLocation().getLineNumber()));
      }
    }
    return eventReader.nextEvent();
  }

  public static XMLEvent peekNextTag(XMLEventReader eventReader) throws XMLStreamException {
    while (eventReader.hasNext()
        && !(eventReader.peek().isEndElement() || eventReader.peek().isStartElement())) {
      eventReader.nextEvent();
    }
    return eventReader.peek();
  }

  @Nullable
  static StartElement findNextStart(XMLEventReader eventReader) throws XMLStreamException {
    while (eventReader.hasNext()) {
      XMLEvent event = eventReader.nextEvent();
      if (event.isStartElement()) {
        return event.asStartElement();
      }
    }
    return null;
  }

  static boolean moveToResources(XMLEventReader eventReader) throws XMLStreamException {
    while (eventReader.hasNext()) {
      StartElement next = findNextStart(eventReader);
      if (next != null && next.getName().equals(TAG_RESOURCES)) {
        return true;
      }
    }
    return false;
  }

  public static SerializeFormat.DataValue.Builder newSerializableDataValueBuilder(Path source) {
    SerializeFormat.DataValue.Builder builder = SerializeFormat.DataValue.newBuilder();
    return builder.setSource(builder.getSourceBuilder().setFilename(source.toString()));
  }

  public static int serializeProtoDataValue(
      OutputStream output, SerializeFormat.DataValue.Builder builder) throws IOException {
    SerializeFormat.DataValue value = builder.build();
    value.writeDelimitedTo(output);
    return CodedOutputStream.computeUInt32SizeNoTag(value.getSerializedSize())
        + value.getSerializedSize();
  }

  public static boolean isEatComment(StartElement start) {
    return isTag(start, TAG_EAT_COMMENT);
  }

  public static boolean isSkip(StartElement start) {
    return isTag(start, TAG_SKIP);
  }
}
