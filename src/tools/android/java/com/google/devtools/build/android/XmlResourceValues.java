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

import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.IdXmlResourceValue;
import com.google.devtools.build.android.xml.PluralXmlResourceValue;
import com.google.devtools.build.android.xml.SimpleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;

import com.android.resources.ResourceType;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/**
 * XmlValues provides methods for getting XmlValue derived classes.
 *
 * <p>Acts a static factory class containing the general xml parsing logic for resources
 * that are declared inside the &lt;resources&gt; tag.
 */
public class XmlResourceValues {

  private static final QName TAG_PLURALS = QName.valueOf("plurals");
  private static final QName ATTR_QUANTITY = QName.valueOf("quantity");
  private static final QName TAG_ATTR = QName.valueOf("attr");
  private static final QName TAG_DECLARE_STYLEABLE = QName.valueOf("declare-styleable");
  private static final QName TAG_ITEM = QName.valueOf("item");
  private static final QName TAG_STYLE = QName.valueOf("style");
  private static final QName TAG_RESOURCES = QName.valueOf("resources");
  private static final QName ATTR_FORMAT = QName.valueOf("format");
  private static final QName ATTR_NAME = QName.valueOf("name");
  private static final QName ATTR_VALUE = QName.valueOf("value");
  private static final QName ATTR_PARENT = QName.valueOf("parent");
  private static final QName ATTR_TYPE = QName.valueOf("type");

  static XmlResourceValue parsePlurals(XMLEventReader eventReader) throws XMLStreamException {
    Map<String, String> values = new HashMap<>();
    for (XMLEvent element = eventReader.nextTag();
        !isEndTag(element, TAG_PLURALS);
        element = eventReader.nextTag()) {
      if (isItem(element)) {
        values.put(
            getElementAttributeByName(element.asStartElement(), ATTR_QUANTITY),
            eventReader.getElementText());
      }
    }
    return PluralXmlResourceValue.of(values);
  }

  static XmlResourceValue parseStyle(XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    Map<String, String> values = new HashMap<>();
    for (XMLEvent element = eventReader.nextTag();
        !isEndTag(element, TAG_STYLE);
        element = eventReader.nextTag()) {
      if (isItem(element)) {
        values.put(getElementName(element.asStartElement()), eventReader.getElementText());
      }
    }
    return StyleXmlResourceValue.of(parseReferenceFromElementAttribute(start, ATTR_PARENT), values);
  }

  static void parseDeclareStyleable(
      FullyQualifiedName.Factory fqnFactory,
      Path path,
      List<DataResource> nonOverwritable,
      List<DataResource> overwritable,
      XMLEventReader eventReader,
      StartElement start)
      throws XMLStreamException {
    String styleableName = getElementName(start);
    List<String> members = new ArrayList<>();
    for (XMLEvent element = eventReader.nextTag();
        !isEndTag(element, TAG_DECLARE_STYLEABLE);
        element = eventReader.nextTag()) {
      if (isStartTag(element, TAG_ATTR)) {
        StartElement attr = element.asStartElement();
        members.add(getElementName(attr));
        String attrName = getElementName(attr);
        overwritable.add(
            XmlDataResource.of(
                fqnFactory.create(ResourceType.ATTR, attrName),
                path,
                parseAttr(eventReader, start)));
      }
    }
    nonOverwritable.add(
        XmlDataResource.of(
            fqnFactory.create(ResourceType.STYLEABLE, styleableName),
            path,
            StyleableXmlResourceValue.of(members)));
  }

  static XmlResourceValue parseAttr(XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    return AttrXmlResourceValue.from(
        start, getElementAttributeByName(start, ATTR_FORMAT), eventReader);
  }

  static XmlResourceValue parseId() {
    return IdXmlResourceValue.of();
  }

  static XmlResourceValue parseSimple(XMLEventReader eventReader, ResourceType resourceType)
      throws XMLStreamException {
    return SimpleXmlResourceValue.of(
        SimpleXmlResourceValue.Type.from(resourceType), eventReader.getElementText());
  }

  /* XML helper methods follow. */
  // TODO(corysmith): Move these to a wrapper class for XMLEventReader.
  private static String parseReferenceFromElementAttribute(StartElement element, QName name)
      throws XMLStreamException {
    String value = getElementAttributeByName(element, name);
    if (value.startsWith("?") || value.startsWith("@")) {
      return value.substring(1);
    }
    throw new XMLStreamException(
        String.format("Invalid resource reference from %s in %s", name, element),
        element.getLocation());
  }

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

  public static XMLEvent peekNextTag(XMLEventReader eventReader) throws XMLStreamException {
    while (!(eventReader.peek().isEndElement() || eventReader.peek().isStartElement())) {
      eventReader.nextEvent();
    }
    return eventReader.peek();
  }

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
      if (findNextStart(eventReader).getName().equals(TAG_RESOURCES)) {
        return true;
      }
    }
    return false;
  }
}
