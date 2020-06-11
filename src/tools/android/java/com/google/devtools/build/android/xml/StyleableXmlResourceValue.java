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

import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.Styleable;
import com.android.aapt.Resources.Value;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.AndroidCompiledDataDeserializer.ReferenceResolver;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidDataWritingVisitor.ValuesResourceDefinition;
import com.google.devtools.build.android.AndroidResourceSymbolSink;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.DependencyInfo;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.XmlType;
import com.google.devtools.build.android.resources.Visibility;
import java.io.IOException;
import java.io.OutputStream;
import java.util.AbstractMap.SimpleEntry;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import javax.annotation.concurrent.Immutable;

/**
 * Represent an Android styleable resource.
 *
 * <p>Styleable resources are groups of attributes that can be applied to views. They are, for the
 * most part, vaguely documented (http://developer.android.com/training/custom-views/create-view
 * .html#customattr). It is important to note that attributes declared inside
 * &lt;declare-styleable&gt; tags, for example; <code> <declare-styleable name="PieChart"> <attr
 * name="showText" format="boolean" /> </declare-styleable> </code>
 *
 * <p>Can also be seen as: <code> <attr name="showText" format="boolean" /> <declare-styleable
 * name="PieChart"> <attr name="showText"/> </declare-styleable> </code>
 *
 * <p>However, aapt will parse these two cases differently. In order to maintain the expected
 * indexing for the styleable array
 * (http://developer.android.com/reference/android/content/res/Resources.Theme.html
 * #obtainStyledAttributes(android.util.AttributeSet, int[], int, int)) the styleable must track
 * whether the attr is a reference or a definition, as aapt will sort the attributes first by attr
 * format (the absence of format comes first, followed by alphabetical sorting by format, then
 * sorting by declaration order in the source xml.)
 */
@Immutable
public class StyleableXmlResourceValue implements XmlResourceValue {

  static final Function<Map.Entry<FullyQualifiedName, Boolean>, SerializeFormat.DataKey>
      FULLY_QUALIFIED_NAME_TO_DATA_KEY =
          new Function<Map.Entry<FullyQualifiedName, Boolean>, SerializeFormat.DataKey>() {
            @Override
            public SerializeFormat.DataKey apply(Map.Entry<FullyQualifiedName, Boolean> input) {
              return input.getKey().toSerializedBuilder().setReference(input.getValue()).build();
            }
          };

  static final Function<SerializeFormat.DataKey, Map.Entry<FullyQualifiedName, Boolean>>
      DATA_KEY_TO_FULLY_QUALIFIED_NAME =
          new Function<SerializeFormat.DataKey, Map.Entry<FullyQualifiedName, Boolean>>() {
            @Override
            public Map.Entry<FullyQualifiedName, Boolean> apply(SerializeFormat.DataKey input) {
              FullyQualifiedName key = FullyQualifiedName.fromProto(input);
              return new SimpleEntry<FullyQualifiedName, Boolean>(key, input.getReference());
            }
          };

  private final Visibility visibility;
  private final Styleable styleable;
  // TODO(b/145837824,b/112848607): change to a set, if not removing this outright.  Per the Javadoc
  // for this class, the "should inline" bit is used to mimic how AAPT1 assigns IDs.
  private final ImmutableMap<FullyQualifiedName, /*shouldInline=*/ Boolean> attrs;

  private StyleableXmlResourceValue(
      Visibility visibility, Styleable styleable, ImmutableMap<FullyQualifiedName, Boolean> attrs) {
    this.visibility = visibility;
    this.styleable = styleable;
    this.attrs = attrs;
  }

  @VisibleForTesting
  public static XmlResourceValue createAllAttrAsReferences(FullyQualifiedName... attrNames) {
    return of(Visibility.UNKNOWN, createAttrDefinitionMap(attrNames, Boolean.FALSE));
  }

  private static Map<FullyQualifiedName, Boolean> createAttrDefinitionMap(
      FullyQualifiedName[] attrNames, Boolean definitionType) {
    ImmutableMap.Builder<FullyQualifiedName, Boolean> builder = ImmutableMap.builder();
    for (FullyQualifiedName attrName : attrNames) {
      builder.put(attrName, definitionType);
    }
    return builder.build();
  }

  @VisibleForTesting
  public static XmlResourceValue createAllAttrAsDefinitions(FullyQualifiedName... attrNames) {
    return of(createAttrDefinitionMap(attrNames, Boolean.TRUE));
  }

  public static XmlResourceValue of(Map<FullyQualifiedName, Boolean> attrs) {
    return new StyleableXmlResourceValue(
        Visibility.UNKNOWN, Styleable.getDefaultInstance(), ImmutableMap.copyOf(attrs));
  }

  public static XmlResourceValue of(Visibility visibility, Map<FullyQualifiedName, Boolean> attrs) {
    return new StyleableXmlResourceValue(
        visibility, Styleable.getDefaultInstance(), ImmutableMap.copyOf(attrs));
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {
    ValuesResourceDefinition definition =
        mergedDataWriter
            .define(key)
            .derivedFrom(source)
            .startTag("declare-styleable")
            .named(key)
            .closeTag();
    for (Map.Entry<FullyQualifiedName, Boolean> entry : attrs.entrySet()) {
      if (entry.getValue().booleanValue()) {
        // Move the attr definition to this styleable.
        definition = definition.adopt(entry.getKey());
      } else {
        // Make a reference to the attr.
        definition =
            definition
                .startTag("attr")
                .attribute("name")
                .setTo(entry.getKey())
                .closeUnaryTag()
                .addCharactersOf("\n");
      }
    }
    definition.endTag().save();
  }

  @Override
  public void writeResourceToClass(
      DependencyInfo dependencyInfo, FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptStyleableResource(dependencyInfo, visibility, key, attrs);
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    return XmlResourceValues.serializeProtoDataValue(
        output,
        XmlResourceValues.newSerializableDataValueBuilder(sourceId)
            .setXmlValue(
                SerializeFormat.DataValueXml.newBuilder()
                    .setType(XmlType.STYLEABLE)
                    .putAllNamespace(namespaces.asMap())
                    .addAllReferences(
                        Iterables.transform(attrs.entrySet(), FULLY_QUALIFIED_NAME_TO_DATA_KEY))));
  }

  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(
        ImmutableMap.copyOf(
            Iterables.transform(proto.getReferencesList(), DATA_KEY_TO_FULLY_QUALIFIED_NAME)));
  }

  public static XmlResourceValue from(
      Value proto, Visibility visibility, ReferenceResolver packageResolver) {
    Map<FullyQualifiedName, Boolean> attributes = new LinkedHashMap<>();

    Styleable styleable = proto.getCompoundValue().getStyleable();
    for (Styleable.Entry entry : styleable.getEntryList()) {
      final FullyQualifiedName reference = packageResolver.parse(entry.getAttr().getName());
      final boolean shouldInline = packageResolver.shouldInline(reference);
      attributes.put(reference, shouldInline);
      if (shouldInline) {
        packageResolver.markInlined(reference);
      }
    }

    return new StyleableXmlResourceValue(visibility, styleable, ImmutableMap.copyOf(attributes));
  }

  @Override
  public int hashCode() {
    return Objects.hash(visibility, attrs);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof StyleableXmlResourceValue)) {
      return false;
    }
    // TODO(b/112848607): include the "styleable" proto in comparison; right now it's redundant.
    StyleableXmlResourceValue other = (StyleableXmlResourceValue) obj;
    return Objects.equals(visibility, other.visibility) && Objects.equals(attrs, other.attrs);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("attrs", attrs).toString();
  }

  /**
   * Combines this instance with another {@link StyleableXmlResourceValue}.
   *
   * <p>Defining two Styleables (undocumented in the official Android Docs) with the same {@link
   * FullyQualifiedName} results in a single Styleable containing a union of all the attribute
   * references.
   *
   * @param value Another {@link StyleableXmlResourceValue} with the same {@link
   *     FullyQualifiedName}.
   * @return {@link StyleableXmlResourceValue} containing a sorted union of the attribute
   *     references.
   * @throws IllegalArgumentException if value is not an {@link StyleableXmlResourceValue}.
   */
  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    if (!(value instanceof StyleableXmlResourceValue)) {
      throw new IllegalArgumentException(value + "is not combinable with " + this);
    }
    StyleableXmlResourceValue other = (StyleableXmlResourceValue) value;
    Map<FullyQualifiedName, Boolean> combined = new LinkedHashMap<>();
    combined.putAll(attrs);
    for (Map.Entry<FullyQualifiedName, Boolean> attr : other.attrs.entrySet()) {
      if (combined.containsKey(attr.getKey())) {
        // if either attr is defined in the styleable, the attr will be defined in the styleable.
        if (attr.getValue() || combined.get(attr.getKey())) {
          combined.put(attr.getKey(), Boolean.TRUE);
        } else {
          combined.put(attr.getKey(), Boolean.FALSE);
        }
      } else {
        combined.put(attr.getKey(), attr.getValue());
      }
    }
    // TODO(b/26297204): test that this makes sense and works
    return new StyleableXmlResourceValue(
        Visibility.merge(visibility, other.visibility),
        styleable.toBuilder().mergeFrom(other.styleable).build(),
        ImmutableMap.copyOf(combined));
  }

  @Override
  public int compareMergePriorityTo(XmlResourceValue value) {
    return 0;
  }

  @Override
  public String asConflictStringWith(DataSource source) {
    return source.asConflictString();
  }

  @Override
  public Visibility getVisibility() {
    return visibility;
  }

  @Override
  public ImmutableList<Reference> getReferencedResources() {
    return styleable.getEntryList().stream()
        .map(entry -> entry.getAttr())
        .collect(ImmutableList.toImmutableList());
  }
}
