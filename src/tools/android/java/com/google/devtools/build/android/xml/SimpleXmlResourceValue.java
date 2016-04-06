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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;

import com.android.resources.ResourceType;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

import javax.annotation.concurrent.Immutable;
import javax.xml.namespace.QName;

/**
 * Represents a simple Android resource xml value.
 *
 * <p>There is a class of resources that are simple name/value pairs:
 * string (http://developer.android.com/guide/topics/resources/string-resource.html),
 * bool (http://developer.android.com/guide/topics/resources/more-resources.html#Bool),
 * color (http://developer.android.com/guide/topics/resources/more-resources.html#Color), and
 * dimen (http://developer.android.com/guide/topics/resources/more-resources.html#Dimension).
 * These are defined in xml as &lt;<em>resource type</em> name="<em>name</em>"
 * value="<em>value</em>"&gt;. In the interest of keeping the parsing svelte, these are
 * represented by a single class.
 */
@Immutable
public class SimpleXmlResourceValue implements XmlResourceValue {
  static final QName TAG_STRING = QName.valueOf("string");
  static final QName TAG_BOOL = QName.valueOf("bool");
  static final QName TAG_COLOR = QName.valueOf("color");
  static final QName TAG_DIMEN = QName.valueOf("dimen");

  /** Provides an enumeration resource type and simple value validation. */
  public enum Type {
    STRING(TAG_STRING) {
      @Override
      public boolean validate(String value) {
        return true;
      }
    },
    BOOL(TAG_BOOL) {
      @Override
      public boolean validate(String value) {
        final String cleanValue = value.toLowerCase().trim();
        return "true".equals(cleanValue) || "false".equals(cleanValue);
      }
    },
    COLOR(TAG_COLOR) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the hex color.
        return true;
      }
    },
    DIMEN(TAG_DIMEN) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the dimension type.
        return true;
      }
    };
    private QName tagName;

    Type(QName tagName) {
      this.tagName = tagName;
    }

    abstract boolean validate(String value);

    public static Type from(ResourceType resourceType) {
      for (Type valueType : values()) {
        if (valueType.tagName.getLocalPart().equals(resourceType.getName())) {
          return valueType;
        }
      }
      throw new IllegalArgumentException(
          String.format(
              "%s resource type not found in available types: %s",
              resourceType,
              Arrays.toString(values())));
    }
  }

  private final String value;
  private final Type valueType;

  public static XmlResourceValue of(Type valueType, String value) {
    return new SimpleXmlResourceValue(valueType, value);
  }

  private SimpleXmlResourceValue(Type valueType, String value) {
    this.valueType = valueType;
    this.value = value;
  }

  @Override
  public void write(
      FullyQualifiedName key, Path source, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.writeToValuesXml(
        key,
        ImmutableList.of(
            String.format("<!-- %s -->", source),
            String.format(
                "<%s name='%s'>%s</%s>",
                valueType.tagName.getLocalPart(),
                key.name(),
                value,
                valueType.tagName.getLocalPart())));
  }

  @Override
  public int hashCode() {
    return Objects.hash(valueType, value);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof SimpleXmlResourceValue)) {
      return false;
    }
    SimpleXmlResourceValue other = (SimpleXmlResourceValue) obj;
    return Objects.equals(valueType, other.valueType) && Objects.equals(value, other.value);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("valueType", valueType)
        .add("value", value)
        .toString();
  }
}
