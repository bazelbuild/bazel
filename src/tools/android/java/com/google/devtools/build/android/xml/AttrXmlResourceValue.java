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
package com.google.devtools.build.android.xml;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;

import java.io.Writer;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import javax.annotation.Nullable;
import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/**
 * Represents an Android Resource custom attribute.
 *
 * <p>Attribute are the most complicated Android resource, and therefore the least documented. Most
 * of the information about them is found by reading the android compatibility library source. An
 * Attribute defines a parameter that can be passed into a view class -- as such you can think of
 * attributes as creating slots for other resources to fit into. Each slot will have at least one
 * format, and can have multiples. Simple attributes (color, boolean, reference, dimension,
 * float, integer, string, and fraction) are defined as &lt;attr name="<em>name</em>"
 * format="<em>format</em>" /&gt; while the complex ones, flag and enum, have sub tags:
 * &lt;attr name="<em>name</em>" &gt&lt;flag name="<em>name</em>" value="<em>value</em>"&gt;
 * &lt;/attr&gt;.
 *
 * <p>Attributes also have a double duty as defining validation logic for layout resources --
 * each layout attribute *must* have a corresponding attribute which will be used to validate the
 * value/resource reference defined in it.
 *
 * <p>AttrXmlValue, due to the mutiple types of attributes is actually a composite class that
 * contains multiple XmlValue instances for each resource.
 */
public class AttrXmlResourceValue implements XmlResourceValue {

  private static final String FRACTION = "fraction";
  private static final String STRING = "string";
  private static final String INTEGER = "integer";
  private static final String FLOAT = "float";
  private static final String DIMENSION = "dimension";
  private static final String BOOLEAN = "boolean";
  private static final String COLOR = "color";
  private static final String REFERENCE = "reference";
  private static final String ENUM = "enum";
  private static final String FLAG = "flag";
  private static final QName TAG_ENUM = QName.valueOf(ENUM);
  private static final QName TAG_FLAG = QName.valueOf(FLAG);
  private final Map<String, XmlResourceValue> formats;

  private AttrXmlResourceValue(Map<String, XmlResourceValue> formats) {
    this.formats = formats;
  }

  private static Map<String, String> readSubValues(XMLEventReader reader, QName subTagType)
      throws XMLStreamException {
    Map<String, String> enumValue = new HashMap<>();
    while (reader.hasNext()
        && XmlResourceValues.isTag(XmlResourceValues.peekNextTag(reader), subTagType)) {
      StartElement element = reader.nextEvent().asStartElement();
      enumValue.put(
          XmlResourceValues.getElementName(element), XmlResourceValues.getElementValue(element));
      XMLEvent endTag = reader.nextEvent();
      if (!XmlResourceValues.isEndTag(endTag, subTagType)) {
        throw new XMLStreamException(
            String.format("Unexpected [%s]; Expected %s", endTag, "</enum>"), endTag.getLocation());
      }
    }
    return enumValue;
  }

  private static void endAttrElement(XMLEventReader reader) throws XMLStreamException {
    XMLEvent endTag = reader.nextTag();
    if (!endTag.isEndElement() || !QName.valueOf("attr").equals(endTag.asEndElement().getName())) {
      throw new XMLStreamException("Unexpected Tag:" + endTag, endTag.getLocation());
    }
  }

  @VisibleForTesting
  private static final class BuilderEntry implements Map.Entry<String, XmlResourceValue> {
    private final String name;
    private final XmlResourceValue value;

    public BuilderEntry(String name, XmlResourceValue value) {
      this.name = name;
      this.value = value;
    }

    @Override
    public String getKey() {
      return name;
    }

    @Override
    public XmlResourceValue getValue() {
      return value;
    }

    @Override
    public XmlResourceValue setValue(XmlResourceValue value) {
      throw new UnsupportedOperationException();
    }
  }

  @SafeVarargs
  @VisibleForTesting
  public static XmlResourceValue fromFormatEntries(Map.Entry<String, XmlResourceValue>... entries) {
    return of(ImmutableMap.copyOf(Arrays.asList(entries)));
  }

  public static XmlResourceValue from(
      StartElement attr, @Nullable String format, XMLEventReader eventReader)
      throws XMLStreamException {
    Set<String> formatNames = new HashSet<>();
    if (format != null) {
      Collections.addAll(formatNames, format.split("\\|"));
    }
    XMLEvent nextTag = XmlResourceValues.peekNextTag(eventReader);
    if (nextTag != null && nextTag.isStartElement()) {
      formatNames.add(nextTag.asStartElement().getName().getLocalPart().toLowerCase());
    }

    Map<String, XmlResourceValue> formats = new HashMap<>();
    for (String formatName : formatNames) {
      switch (formatName) {
        case FLAG:
          Map<String, String> flags = readSubValues(eventReader, TAG_FLAG);
          endAttrElement(eventReader);
          formats.put(formatName, AttrFlagXmlValue.of(flags));
          break;
        case ENUM:
          Map<String, String> enums = readSubValues(eventReader, TAG_ENUM);
          endAttrElement(eventReader);
          formats.put(formatName, AttrEnumXmlValue.of(enums));
          break;
        case REFERENCE:
          formats.put(formatName, AttrReferenceXmlValue.of());
          break;
        case COLOR:
          formats.put(formatName, AttrColorXmlValue.of());
          break;
        case BOOLEAN:
          formats.put(formatName, AttrBooleanXmlValue.of());
          break;
        case DIMENSION:
          formats.put(formatName, AttrDimensionXmlValue.of());
          break;
        case FLOAT:
          formats.put(formatName, AttrFloatXmlValue.of());
          break;
        case INTEGER:
          formats.put(formatName, AttrIntegerXmlValue.of());
          break;
        case STRING:
          formats.put(formatName, AttrStringXmlValue.of());
          break;
        case FRACTION:
          formats.put(formatName, AttrFractionXmlValue.of());
          break;
        default:
          throw new XMLStreamException(
              String.format("Unexpected attr format: %S", formatName), attr.getLocation());
      }
    }
    return of(formats);
  }

  public static XmlResourceValue of(Map<String, XmlResourceValue> formats) {
    return new AttrXmlResourceValue(formats);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    AttrXmlResourceValue other = (AttrXmlResourceValue) o;
    return Objects.equals(formats, other.formats);
  }

  @Override
  public int hashCode() {
    return formats.hashCode();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("formats", formats).toString();
  }

  @Override
  public void write(Writer buffer, FullyQualifiedName name) {
    // TODO(corysmith): Implement write.
    throw new UnsupportedOperationException();
  }

  /** Represents an Android Enum Attribute resource. */
  @VisibleForTesting
  public static class AttrEnumXmlValue implements XmlResourceValue {

    private Map<String, String> values;

    private AttrEnumXmlValue(Map<String, String> values) {
      this.values = values;
    }

    @VisibleForTesting
    public static Map.Entry<String, XmlResourceValue> asEntryOf(String... keyThenValue) {
      Preconditions.checkArgument(keyThenValue.length > 0);
      Preconditions.checkArgument(keyThenValue.length % 2 == 0);
      Builder<String, String> builder = ImmutableMap.<String, String>builder();
      for (int i = 0; i < keyThenValue.length; i += 2) {
        builder.put(keyThenValue[i], keyThenValue[i + 1]);
      }
      return new BuilderEntry(ENUM, of(builder.build()));
    }

    public static XmlResourceValue of(Map<String, String> values) {
      return new AttrEnumXmlValue(values);
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }

    @Override
    public int hashCode() {
      return values.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof AttrEnumXmlValue)) {
        return false;
      }
      AttrEnumXmlValue other = (AttrEnumXmlValue) obj;
      return Objects.equals(values, other.values);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(getClass()).add("values", values).toString();
    }
  }

  /** Represents an Android Flag Attribute resource. */
  @VisibleForTesting
  public static class AttrFlagXmlValue implements XmlResourceValue {

    private Map<String, String> values;

    private AttrFlagXmlValue(Map<String, String> values) {
      this.values = values;
    }

    public static XmlResourceValue of(Map<String, String> values) {
      return new AttrFlagXmlValue(values);
    }

    @VisibleForTesting
    public static Map.Entry<String, XmlResourceValue> asEntryOf(String... keyThenValue) {
      Builder<String, String> builder = ImmutableMap.<String, String>builder();
      Preconditions.checkArgument(keyThenValue.length > 0);
      Preconditions.checkArgument(keyThenValue.length % 2 == 0);
      for (int i = 0; i < keyThenValue.length; i += 2) {
        builder.put(keyThenValue[i], keyThenValue[i + 1]);
      }
      return new BuilderEntry(FLAG, of(builder.build()));
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }

    @Override
    public int hashCode() {
      return values.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof AttrFlagXmlValue)) {
        return false;
      }
      AttrFlagXmlValue other = (AttrFlagXmlValue) obj;
      return Objects.equals(values, other.values);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(getClass()).add("values", values).toString();
    }
  }

  /** Represents an Android Reference Attribute resource. */
  @VisibleForTesting
  public static class AttrReferenceXmlValue implements XmlResourceValue {
    private static final AttrReferenceXmlValue INSTANCE = new AttrReferenceXmlValue();

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(REFERENCE, of());
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }

  /** Represents an Android Color Attribute resource. */
  @VisibleForTesting
  public static class AttrColorXmlValue implements XmlResourceValue {
    private static final AttrColorXmlValue INSTANCE = new AttrColorXmlValue();

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(COLOR, of());
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }

  /** Represents an Android Boolean Attribute resource. */
  @VisibleForTesting
  public static class AttrBooleanXmlValue implements XmlResourceValue {
    private static final AttrBooleanXmlValue INSTANCE = new AttrBooleanXmlValue();

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(BOOLEAN, of());
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }

  /** Represents an Android Float Attribute resource. */
  @VisibleForTesting
  public static class AttrFloatXmlValue implements XmlResourceValue {
    private static final AttrFloatXmlValue INSTANCE = new AttrFloatXmlValue();

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(FLOAT, of());
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }

  /** Represents an Android Dimension Attribute resource. */
  @VisibleForTesting
  public static class AttrDimensionXmlValue implements XmlResourceValue {
    private static final AttrDimensionXmlValue INSTANCE = new AttrDimensionXmlValue();

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(DIMENSION, of());
    }

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }

  /** Represents an Android Integer Attribute resource. */
  @VisibleForTesting
  public static class AttrIntegerXmlValue implements XmlResourceValue {
    private static final AttrIntegerXmlValue INSTANCE = new AttrIntegerXmlValue();

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(INTEGER, of());
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }

  /** Represents an Android String Attribute resource. */
  @VisibleForTesting
  public static class AttrStringXmlValue implements XmlResourceValue {
    private static final AttrStringXmlValue INSTANCE = new AttrStringXmlValue();

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(STRING, of());
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }

  /** Represents an Android Fraction Attribute resource. */
  @VisibleForTesting
  public static class AttrFractionXmlValue implements XmlResourceValue {
    private static final AttrFractionXmlValue INSTANCE = new AttrFractionXmlValue();

    public static XmlResourceValue of() {
      return INSTANCE;
    }

    @VisibleForTesting
    public static BuilderEntry asEntry() {
      return new BuilderEntry(FRACTION, of());
    }

    @Override
    public void write(Writer buffer, FullyQualifiedName name) {
      // TODO(corysmith): Implement write.
      throw new UnsupportedOperationException();
    }
  }
}
