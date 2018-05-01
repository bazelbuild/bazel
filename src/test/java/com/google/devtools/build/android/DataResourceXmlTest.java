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

import static com.google.common.truth.Truth.assertAbout;
import static com.google.common.truth.Truth.assertThat;

import com.android.resources.ResourceType;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.jimfs.Jimfs;
import com.google.common.truth.Subject;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.xml.ArrayXmlResourceValue;
import com.google.devtools.build.android.xml.ArrayXmlResourceValue.ArrayType;
import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.BooleanResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.ColorResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.DimensionResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.EnumResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.FlagResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.FloatResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.FractionResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.IntegerResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.ReferenceResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.StringResourceXmlAttrValue;
import com.google.devtools.build.android.xml.IdXmlResourceValue;
import com.google.devtools.build.android.xml.Namespaces;
import com.google.devtools.build.android.xml.PluralXmlResourceValue;
import com.google.devtools.build.android.xml.PublicXmlResourceValue;
import com.google.devtools.build.android.xml.ResourcesAttribute;
import com.google.devtools.build.android.xml.SimpleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashMap;
import java.util.Map;
import javax.xml.stream.XMLStreamException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DataResourceXml}. */
@RunWith(JUnit4.class)
public class DataResourceXmlTest {
  static final ImmutableMap<String, String> XLIFF_NAMESPACES =
      ImmutableMap.of("xliff", "urn:oasis:names:tc:xliff:document:1.2");
  static final String END_RESOURCES = new String(AndroidDataWriter.END_RESOURCES);

  private FullyQualifiedName.Factory fqnFactory;
  private FileSystem fs;

  @Before
  public void createCleanEnvironment() {
    fs = Jimfs.newFileSystem();
    fqnFactory = FullyQualifiedName.Factory.from(ImmutableList.<String>of());
  }

  private Path writeResourceXml(String... xml) throws IOException {
    return writeResourceXml(ImmutableMap.<String, String>of(), ImmutableMap.<String, String>of(),
        xml);
  }

  private Path writeResourceXml(Map<String, String> namespaces, Map<String, String> attributes,
      String... xml) throws IOException {
    Path values = fs.getPath("root/values/values.xml");
    Files.createDirectories(values.getParent());
    StringBuilder builder = new StringBuilder();
    builder.append(AndroidDataWriter.PRELUDE).append("<resources");
    for (Map.Entry<String, String> entry : namespaces.entrySet()) {
      builder
          .append(" xmlns:")
          .append(entry.getKey())
          .append("=\"")
          .append(entry.getValue())
          .append("\"");
    }
    for (Map.Entry<String, String> entry : attributes.entrySet()) {
      builder
          .append(" ")
          .append(entry.getKey())
          .append("=\"")
          .append(entry.getValue())
          .append("\"");
    }
    builder.append(">");
    Joiner.on("\n").appendTo(builder, xml);
    builder.append(END_RESOURCES);
    Files.write(values, builder.toString().getBytes(StandardCharsets.UTF_8));
    return values;
  }

  private void parseResourcesFrom(
      Path path, Map<DataKey, DataResource> toOverwrite, Map<DataKey, DataResource> toCombine)
      throws XMLStreamException, IOException {
    DataResourceXml.parse(
        XmlResourceValues.getXmlInputFactory(),
        path,
        fqnFactory,
        new FakeConsumer(toOverwrite),
        new FakeConsumer(toCombine));
  }

  private FullyQualifiedName fqn(String raw) {
    return fqnFactory.parse(raw);
  }

  @Test
  public void simpleXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "<string name='exit' description=\"&amp; egress -&gt; &quot;\">way out</string>",
            "<bool name='canExit'>false</bool>",
            "<color name='exitColor'>#FF000000</color>",
            "<dimen name='exitSize'>20sp</dimen>",
            "<integer name='exitInt'>20</integer>",
            "<fraction name='exitFraction'>%20</fraction>",
            "<drawable name='exitBackground'>#99000000</drawable>",
            "<item name='some_id' type='id'/>",
            "<item name='reference_id' type='id'>@id/some_id</item>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("string/exit"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.of(
                    SimpleXmlResourceValue.Type.STRING,
                    ImmutableMap.of("description", "&amp; egress -&gt; &quot;"),
                    "way out")), // Value
            fqn("bool/canExit"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.BOOL, "false")), // Value
            fqn("color/exitColor"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.COLOR, "#FF000000")), // Value
            fqn("dimen/exitSize"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.DIMEN, "20sp")), // Value
            fqn("integer/exitInt"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.INTEGER, "20")), // Value
            fqn("fraction/exitFraction"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.FRACTION, "%20")), // Value
            fqn("drawable/exitBackground"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.DRAWABLE, "#99000000")) // Value
            );
    assertThat(toCombine)
        .containsExactly(
            fqn("id/some_id"), // Key
            DataResourceXml.createWithNoNamespace(path, IdXmlResourceValue.of()), // Value
            fqn("id/reference_id"), // Key
            DataResourceXml.createWithNoNamespace(
                path, IdXmlResourceValue.of("@id/some_id")) // Value
            );
  }

  @Test
  public void skipIgnored() throws Exception {
    Path path = writeResourceXml("<skip/>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite).isEmpty();
    assertThat(toCombine).isEmpty();
  }

  @Test
  public void eatCommentIgnored() throws Exception {
    Path path = writeResourceXml("<eat-comment/>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite).isEmpty();
    assertThat(toCombine).isEmpty();
  }

  @Test
  public void itemSimpleXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "<item type='dimen' name='exitSizePercent'>20%</item>",
            "<item type='dimen' format='float' name='exitSizeFloat'>20.0</item>",
            "<item type='fraction' name='denom'>%5</item>",
            "<item name='subtype_id' type='id'>0x6f972360</item>",
            "<item type='array' name='oboes'/>",
            "<item type='drawable' name='placeholder'/>");
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("dimen/exitSizePercent"), // Key
            DataResourceXml.createWithNoNamespace(
                path, SimpleXmlResourceValue.itemWithValue(ResourceType.DIMEN, "20%")), // Value
            fqn("dimen/exitSizeFloat"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.itemWithFormattedValue(
                    ResourceType.DIMEN, "float", "20.0")), // Value
            fqn("drawable/placeholder"), // Key
            DataResourceXml.createWithNoNamespace(
                path, SimpleXmlResourceValue.itemPlaceHolderFor(ResourceType.DRAWABLE)), // Value
            fqn("array/oboes"), // Key
            DataResourceXml.createWithNoNamespace(
                path, SimpleXmlResourceValue.itemPlaceHolderFor(ResourceType.ARRAY)), // Value
            fqn("fraction/denom"), // Key
            DataResourceXml.createWithNoNamespace(
                path, SimpleXmlResourceValue.itemWithValue(ResourceType.FRACTION, "%5")) // Value
            );
  }

  @Test
  public void styleableXmlResourcesEnum() throws Exception {
    Path path =
        writeResourceXml(
            "<declare-styleable name='Theme'>",
            "  <attr name=\"labelPosition\" format=\"enum\">",
            "    <enum name=\"left\" value=\"0\"/>",
            "    <enum name=\"right\" value=\"1\"/>",
            "  </attr>",
            "</declare-styleable>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("attr/labelPosition"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                AttrXmlResourceValue.fromFormatEntries(
                    EnumResourceXmlAttrValue.asEntryOf("left", "0", "right", "1"))) // Value
            );
    assertThat(toCombine)
        .containsExactly(
            fqn("styleable/Theme"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                StyleableXmlResourceValue.createAllAttrAsDefinitions(
                    fqnFactory.parse("attr/labelPosition"))) // Value
            );
  }

  @Test
  public void styleableXmlResourcesString() throws Exception {
    Path path =
        writeResourceXml(
            "<declare-styleable name='UnusedStyleable'>",
            "  <attr name='attribute1'/>",
            "  <attr name='attribute2'/>",
            "</declare-styleable>",
            "<attr name='attribute1' format='string'/>",
            "<attr name='attribute2' format='string'/>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("attr/attribute1"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                AttrXmlResourceValue.fromFormatEntries(
                    StringResourceXmlAttrValue.asEntry())), // Value
            fqn("attr/attribute2"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                AttrXmlResourceValue.fromFormatEntries(
                    StringResourceXmlAttrValue.asEntry())) // Value
            );
    assertThat(toCombine)
        .containsExactly(
            fqn("styleable/UnusedStyleable"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                StyleableXmlResourceValue.createAllAttrAsReferences(
                    fqnFactory.parse("attr/attribute1"),
                    fqnFactory.parse("attr/attribute2"))) // Value
            );
  }

  @Test
  public void pluralXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "<plurals name='numberOfSongsAvailable'>",
            "  <item quantity='one'>%d song found.</item>",
            "  <item quantity='other'>%d songs found.</item>",
            "</plurals>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("plurals/numberOfSongsAvailable"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                PluralXmlResourceValue.createWithoutAttributes(
                    ImmutableMap.of(
                        "one", "%d song found.",
                        "other", "%d songs found."))) // Value
            );
  }

  @Test
  public void pluralXmlResourcesWithTrailingCharacters() throws Exception {
    Path path =
        writeResourceXml(
            "<plurals name='numberOfSongsAvailable'>",
            "  <item quantity='one'>%d song found.</item> // this is an invalid comment.",
            "  <item quantity='other'>%d songs found.</item>typo",
            "</plurals>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("plurals/numberOfSongsAvailable"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                PluralXmlResourceValue.createWithoutAttributes(
                    ImmutableMap.of(
                        "one", "%d song found.",
                        "other", "%d songs found."))) // Value
            );
  }

  @Test
  public void styleResources() throws Exception {
    Path path =
        writeResourceXml(
            "<style name=\"candlestick_maker\" parent=\"@style/wax_maker\">\n"
                + "  <item name=\"texture\">waxy</item>\n"
                + "</style>");
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("style/candlestick_maker"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                StyleXmlResourceValue.of(
                    "@style/wax_maker", ImmutableMap.of("texture", "waxy"))) // Value
            );
  }

  @Test
  public void styleResourcesNoParent() throws Exception {
    Path path =
        writeResourceXml(
            "<style name=\"CandlestickMaker\">\n"
                + "  <item name=\"texture\">waxy</item>\n"
                + "</style>");
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("style/CandlestickMaker"), // Key
            DataResourceXml.createWithNoNamespace(
                path, StyleXmlResourceValue.of(null, ImmutableMap.of("texture", "waxy"))) // Value
            );
  }

  @Test
  public void styleResourcesForceNoParent() throws Exception {
    Path path =
        writeResourceXml(
            // using a '.' implies a parent Candlestick, parent='' corrects to no parent.
            "<style name=\"Candlestick.Maker\" parent=\"\">\n"
                + "  <item name=\"texture\">waxy</item>\n"
                + "</style>");
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("style/Candlestick.Maker"), // Key
            DataResourceXml.createWithNoNamespace(
                path, StyleXmlResourceValue.of("", ImmutableMap.of("texture", "waxy"))) // Value
            );
  }

  @Test
  public void styleResourcesLazyReference() throws Exception {
    Path path =
        writeResourceXml(
            "<style name=\"candlestick_maker\" parent=\"AppTheme\">\n"
                + "  <item name=\"texture\">waxy</item>\n"
                + "</style>");
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("style/candlestick_maker"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                StyleXmlResourceValue.of("AppTheme", ImmutableMap.of("texture", "waxy"))) // Value
            );
  }

  @Test
  public void arrayXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "<array name='icons'>",
            "  <item>@drawable/home</item>",
            "  <item>@drawable/settings</item>",
            "  <item>@drawable/logout</item>",
            "</array>",
            "<array name='colors'>",
            "  <item>#FFFF0000</item>",
            "  <item>#FF00FF00</item>",
            "  <item>#FF0000FF</item>",
            "</array>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("array/icons"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                ArrayXmlResourceValue.of(
                    ArrayType.ARRAY,
                    "@drawable/home",
                    "@drawable/settings",
                    "@drawable/logout")), // Value
            fqn("array/colors"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                ArrayXmlResourceValue.of(
                    ArrayType.ARRAY, "#FFFF0000", "#FF00FF00", "#FF0000FF")) // Value
            );
  }

  @Test
  public void stringArrayXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "<string-array name='characters'>",
            "  <item>boojum</item>",
            "  <item>snark</item>",
            "  <item>bellman</item>",
            "  <item>barrister</item>",
            "  <item>\\\"billiard-marker\\\"</item>",
            "</string-array>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("array/characters"),
            DataResourceXml.createWithNoNamespace(
                path,
                ArrayXmlResourceValue.of(
                    ArrayType.STRING_ARRAY,
                    "boojum",
                    "snark",
                    "bellman",
                    "barrister",
                    "\\\"billiard-marker\\\"")));
    assertThat(toCombine).isEmpty();
  }

  @Test
  public void integerArrayXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "<integer-array name='bits'>",
            "  <item>4</item>",
            "  <item>8</item>",
            "  <item>16</item>",
            "  <item>32</item>",
            "</integer-array>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("array/bits"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                ArrayXmlResourceValue.of(ArrayType.INTEGER_ARRAY, "4", "8", "16", "32")) // Value
            );
  }

  @Test
  public void attrFlagXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "  <attr name=\"labelPosition\">",
            "    <flag name=\"left\" value=\"0\"/>",
            "    <flag name=\"right\" value=\"1\"/>",
            "  </attr>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("attr/labelPosition"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                AttrXmlResourceValue.fromFormatEntries(
                    FlagResourceXmlAttrValue.asEntryOf(
                        "left", "0",
                        "right", "1"))) // Value
            );
  }

  @Test
  public void attrMultiFormatImplicitFlagXmlResources() throws Exception {
    Path path =
        writeResourceXml(
            "  <attr name='labelPosition' format='reference'>",
            "    <flag name='left' value='0'/>",
            "    <flag name='right' value='1'/>",
            "  </attr>");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("attr/labelPosition"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                AttrXmlResourceValue.fromFormatEntries(
                    ReferenceResourceXmlAttrValue.asEntry(),
                    FlagResourceXmlAttrValue.asEntryOf(
                        "left", "0",
                        "right", "1"))) // Value
            );
  }

  @Test
  public void attrMultiFormatResources() throws Exception {
    Path path =
        writeResourceXml(
            "<attr name='labelPosition' ",
            "format='color|boolean|dimension|float|integer|string|fraction|reference' />");

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    assertThat(toOverwrite)
        .containsExactly(
            fqn("attr/labelPosition"), // Key
            DataResourceXml.createWithNoNamespace(
                path,
                AttrXmlResourceValue.fromFormatEntries(
                    ColorResourceXmlAttrValue.asEntry(),
                    BooleanResourceXmlAttrValue.asEntry(),
                    ReferenceResourceXmlAttrValue.asEntry(),
                    DimensionResourceXmlAttrValue.asEntry(),
                    FloatResourceXmlAttrValue.asEntry(),
                    IntegerResourceXmlAttrValue.asEntry(),
                    StringResourceXmlAttrValue.asEntry(),
                    FractionResourceXmlAttrValue.asEntry())) // Value
            );
  }

  @Test
  public void publicXmlResource() throws Exception {
    Path path =
        writeResourceXml(
            "<string name='exit'>way out</string>",
            "<public type='string' name='exit' id='0x123'/>");
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);
    assertThat(toOverwrite)
        .containsExactly(
            fqn("string/exit"),
            DataResourceXml.createWithNoNamespace(
                path,
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.STRING, "way out")));
    assertThat(toCombine)
        .containsExactly(
            fqn("public/exit"),
            DataResourceXml.createWithNoNamespace(
                path, PublicXmlResourceValue.create(ResourceType.STRING, Optional.of(0x123))));
  }

  @Test
  public void resourcesAttribute() throws Exception {
    Namespaces namespaces = Namespaces.from(
        ImmutableMap.of("tools", "http://schemas.android.com/tools"));

    Path path = writeResourceXml(
        namespaces.asMap(),
        ImmutableMap.of("tools:foo", "fooVal", "tools:ignore", "ignoreVal"));

    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);

    FullyQualifiedName fooFqn = fqn("<resources>/{http://schemas.android.com/tools}foo");
    assertThat(toOverwrite)
        .containsExactly(
            fooFqn,
            DataResourceXml.createWithNamespaces(
                path,
                ResourcesAttribute.of(fooFqn, "tools:foo", "fooVal"),
                namespaces)
            );

    FullyQualifiedName ignoreFqn = fqn("<resources>/{http://schemas.android.com/tools}ignore");
    assertThat(toCombine)
        .containsExactly(
            ignoreFqn,
            DataResourceXml.createWithNamespaces(
                path,
                ResourcesAttribute.of(ignoreFqn, "tools:ignore", "ignoreVal"),
                namespaces)
            );

  }

  @Test
  public void writeSimpleXmlResources() throws Exception {
    Path source =
        writeResourceXml(
            "<string name='exit'>way <a href=\"out.html\">out</a></string>",
            "<bool name='canExit'>false</bool>",
            "<color name='exitColor'>#FF000000</color>",
            "<integer name='exitInt'>5</integer>",
            "<drawable name='reference'/>");
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("string/exit")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                source, "<string name='exit'>way <a href=\"out.html\">out</a></string>"));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("bool/canExit")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, "<bool name='canExit'>false</bool>"));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("color/exitColor")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(source, "<color name='exitColor'>#FF000000</color>"));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("integer/exitInt")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, "<integer name='exitInt'>5</integer>"));
  }

  @Test
  public void writeItemResources() throws Exception {
    Path source =
        writeResourceXml(
            "<item type='dimen' name='exitSizePercent'>20%</item>",
            "<item type='dimen' format='float' name='exitSizeFloat'>20.0</item>",
            "<item name='exitId' type='id'/>",
            "<item name='frac' type='fraction'>5%</item>");
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("dimen/exitSizePercent")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(source, "<item type='dimen' name='exitSizePercent'>20%</item>"));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("dimen/exitSizeFloat")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                source, "<item type='dimen' format='float' name='exitSizeFloat'>20.0</item>"));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("fraction/frac")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(source, "<item name='frac' type='fraction'>5%</item>"));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("id/exitId")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, "<item name='exitId' type='id'/>"));
  }

  @Test
  public void writeStringResourceWithXliffNamespace() throws Exception {
    Path source = fs.getPath("root/values/values.xml");
    Files.createDirectories(source.getParent());
    Files.write(
        source,
        ("<resources xmlns:xliff=\"urn:oasis:names:tc:xliff:document:1.2\">"
                + "<string name=\"star_rating\">Check out our 5\n"
                + "    <xliff:g id=\"star\">\\u2605</xliff:g>\n"
                + "</string>"
                + "</resources>")
            .getBytes(StandardCharsets.UTF_8));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("string/star_rating")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                XLIFF_NAMESPACES,
                ImmutableMap.<String, String>of(),
                source,
                "<string name=\"star_rating\">Check out our 5",
                "    <xliff:g id=\"star\">\\u2605</xliff:g>",
                "</string>"));
  }

  @Test
  public void writeStringResourceCData() throws Exception {
    String[] xml = {
      "<string name='cdata'><![CDATA[<b>Jabber, Jabber</b><br><br>\n Wock!]]></string>"
    };
    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("string/cdata")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeStringResourceWithNamespace() throws Exception {
    Path source = fs.getPath("root/values/values.xml");
    Files.createDirectories(source.getParent());
    Files.write(
        source,
        ("<resources xmlns:ns1=\"urn:oasis:names:tc:xliff:document:1.2\">"
                + "<string name=\"star_rating\">Check out our 5\n"
                + "    <ns1:g xmlns:foglebert=\"defogle\" foglebert:id=\"star\">\\u2605</ns1:g>\n"
                + "</string>"
                + "</resources>")
            .getBytes(StandardCharsets.UTF_8));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("string/star_rating")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                ImmutableMap.of("ns1", "urn:oasis:names:tc:xliff:document:1.2"),
                ImmutableMap.<String, String>of(),
                source,
                "<string name=\"star_rating\">Check out our 5    ",
                "<ns1:g xmlns:foglebert=\"defogle\" " + "foglebert:id=\"star\">\\u2605</ns1:g>",
                "</string>"));
  }

  @Test
  public void writeStringResourceWithUnusedNamespace() throws Exception {
    Path source = fs.getPath("root/values/values.xml");
    Files.createDirectories(source.getParent());
    Files.write(
        source,
        ("<resources xmlns:ns1=\"urn:oasis:names:tc:xliff:document:1.2\">"
                + "<string name=\"star_rating\">"
                + "not yet implemented\n"
                + "</string>"
                + "</resources>")
            .getBytes(StandardCharsets.UTF_8));
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("string/star_rating")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                source, "<string name=\"star_rating\">", "not yet implemented\n", "</string>"));
  }

  @Test
  public void writeStringResourceWithEscapedValues() throws Exception {
    String[] xml = {
      "<string name=\"AAP_SUGGEST_ACCEPT_SUGGESTION\">",
      "        &lt;b&gt;<xliff:g id=\"name\" example=\"Pizza hut\">%1$s</xliff:g>&lt;/b&gt; "
      + "already exists at &lt;b&gt;<xliff:g id=\"address\" "
      + "example=\"123 main street\">%2$s</xliff:g>&lt;/b&gt;&lt;br&gt;",
      "        &lt;br&gt;",
      "        Is this the place you\\'re trying to add?\n",
      "</string>"
    };

    Path source = writeResourceXml(XLIFF_NAMESPACES, ImmutableMap.<String, String>of(), xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("string/AAP_SUGGEST_ACCEPT_SUGGESTION")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(XLIFF_NAMESPACES, ImmutableMap.<String, String>of(), source, xml));
  }

  @Test
  public void writeStyleableXmlResource() throws Exception {
    String[] xml = {
      "<declare-styleable name='Theme'>",
      "  <attr name=\"labelPosition\" />",
      "</declare-styleable>"
    };

    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("styleable/Theme")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeStyleableXmlResourceReference() throws Exception {
    String[] xml = {
      "<declare-styleable name='Theme'>",
      "  <attr name=\"labelColor\" format=\"color\" />",
      "</declare-styleable>"
    };

    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("styleable/Theme"), fqn("attr/labelColor")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writePluralXmlResources() throws Exception {
    String[] xml = {
      "<plurals name='numberOfSongsAvailable' format='none'>",
      "  <item quantity='one'>%d song found.</item>",
      "  <item quantity='other'>%d songs found.</item>",
      "</plurals>"
    };
    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("plurals/numberOfSongsAvailable")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writePublicXmlResources() throws Exception {
    String[] xml = {
        "<public name='bar' type='dimen' id='0x7f030003' />",
        "<public name='foo' type='dimen' />",
    };
    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("public/bar"), fqn("public/foo")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeArrayXmlResources() throws Exception {
    String[] xml = {
      "<array name='icons'>",
      "  <item>@drawable/home</item>",
      "  <item>@drawable/settings</item>",
      "  <item>@drawable/logout</item>",
      "</array>",
    };

    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("array/icons")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeStringArrayXmlResources() throws Exception {
    String[] xml = {
      "<string-array name='rosebud' translatable='false'>",
      "  <item>Howard Hughes</item>",
      "  <item>Randolph Hurst</item>",
      "</string-array>",
    };

    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("array/rosebud")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeIntegerArrayXmlResources() throws Exception {
    String[] xml = {
      "<integer-array name='bits'>",
      "  <item>4</item>",
      "  <item>8</item>",
      "  <item>16</item>",
      "  <item>32</item>",
      "</integer-array>"
    };
    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("array/bits")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeAttrFlagXmlResources() throws Exception {
    String[] xml = {
      "  <attr name=\"labelPosition\">",
      "    <flag name=\"left\" value=\"0\"/>",
      "    <flag name=\"right\" value=\"1\"/>",
      "  </attr>"
    };
    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("attr/labelPosition")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                source,
                "  <attr name=\"labelPosition\">",
                "    <flag name=\"left\" value=\"0\"/>",
                "    <flag name=\"right\" value=\"1\"/>",
                "  </attr>"));
  }

  @Test
  public void writeAttrMultiFormatImplicitFlagXmlResources() throws Exception {
    String[] xml = {
      "  <attr name='labelPosition' format='reference'>",
      "    <flag name='left' value='0'/>",
      "    <flag name='right' value='1'/>",
      "  </attr>"
    };
    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("attr/labelPosition")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                source,
                "  <attr name='labelPosition' format='reference'>",
                "    <flag name='left' value='0'/>",
                "    <flag name='right' value='1'/>",
                "  </attr>"));
  }

  @Test
  public void writeAttrMultiFormatResources() throws Exception {
    String[] xml = {
      "<attr name='labelPosition' ",
      "format='boolean|color|dimension|float|fraction|integer|reference|string'/>"
    };
    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("attr/labelPosition")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeStyle() throws Exception {
    String[] xml = {
      "<style name='candlestick_maker' parent='@style/wax_maker'>\n"
          + "  <item name='texture'>waxy</item>\n"
          + "</style>"
    };

    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("style/candlestick_maker")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeForceNoParentStyle() throws Exception {
    String[] xml = {
      "<style name='candlestick_maker' parent=''>\n",
      "  <item name='texture'>waxy</item>\n",
      "</style>"
    };

    Path source = writeResourceXml(xml);
    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("style/candlestick_maker")))
        .xmlContentsIsEqualTo(resourcesXmlFrom(source, xml));
  }

  @Test
  public void writeResourceAttributes() throws Exception {
    Path source = writeResourceXml(
        ImmutableMap.of("tools", "http://schemas.android.com/tools"),
        ImmutableMap.of("tools:foo", "fooVal"));

    assertAbout(resourcePaths)
        .that(parsedAndWritten(source, fqn("<resources>/{http://schemas.android.com/tools}foo")))
        .xmlContentsIsEqualTo(
            resourcesXmlFrom(
                ImmutableMap.of("tools", "http://schemas.android.com/tools"),
                ImmutableMap.of("tools:foo", "fooVal"),
                null));
  }

  @Test
  public void serializeMultipleSimpleXmlResources() throws Exception {
    Path serialized = fs.getPath("out/out.bin");
    Path source = fs.getPath("res/values/values.xml");
    FullyQualifiedName stringKey = fqn("string/exit");
    DataResourceXml stringValue =
        DataResourceXml.createWithNoNamespace(
            source,
            SimpleXmlResourceValue.createWithValue(SimpleXmlResourceValue.Type.STRING, "way out"));
    FullyQualifiedName nullStringKey = fqn("string/nullexit");
    DataResourceXml nullStringValue =
        DataResourceXml.createWithNoNamespace(
            source,
            SimpleXmlResourceValue.createWithValue(SimpleXmlResourceValue.Type.STRING, null));
    FullyQualifiedName boolKey = fqn("bool/canExit");
    DataResourceXml boolValue =
        DataResourceXml.createWithNoNamespace(
            source,
            SimpleXmlResourceValue.createWithValue(SimpleXmlResourceValue.Type.BOOL, "false"));
    FullyQualifiedName colorKey = fqn("color/exitColor");
    DataResourceXml colorValue =
        DataResourceXml.createWithNoNamespace(
            source,
            SimpleXmlResourceValue.createWithValue(SimpleXmlResourceValue.Type.COLOR, "#FF000000"));
    FullyQualifiedName dimenKey = fqn("dimen/exitSize");
    DataResourceXml dimenValue =
        DataResourceXml.createWithNoNamespace(
            source,
            SimpleXmlResourceValue.createWithValue(SimpleXmlResourceValue.Type.DIMEN, "20sp"));

    AndroidDataSerializer serializer = AndroidDataSerializer.create();
    serializer.queueForSerialization(stringKey, stringValue);
    serializer.queueForSerialization(nullStringKey, nullStringValue);
    serializer.queueForSerialization(boolKey, boolValue);
    serializer.queueForSerialization(colorKey, colorValue);
    serializer.queueForSerialization(dimenKey, dimenValue);
    serializer.flushTo(serialized);

    AndroidDataDeserializer deserializer = AndroidParsedDataDeserializer.create();
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    deserializer.read(
        serialized,
        KeyValueConsumers.of(new FakeConsumer(toOverwrite), new FakeConsumer(toCombine), null));
    assertThat(toOverwrite)
        .containsExactly(
            stringKey, stringValue,
            boolKey, boolValue,
            colorKey, colorValue,
            nullStringKey, nullStringValue,
            dimenKey, dimenValue);
    assertThat(toCombine).isEmpty();
  }

  @Test
  public void serializeItemXmlResources() throws Exception {
    Path source = fs.getPath("res/values/values.xml");
    assertSerialization(
        fqn("dimen/exitSizePercent"),
        DataResourceXml.createWithNoNamespace(
            source, SimpleXmlResourceValue.itemWithValue(ResourceType.DIMEN, "20%")));
    assertSerialization(
        fqn("dimen/exitSizeFloat"),
        DataResourceXml.createWithNoNamespace(
            source,
            SimpleXmlResourceValue.itemWithFormattedValue(ResourceType.DIMEN, "float", "20.0")));
    assertSerialization(
        fqn("fraction/denom"),
        DataResourceXml.createWithNoNamespace(
            source,
            SimpleXmlResourceValue.createWithValue(SimpleXmlResourceValue.Type.COLOR, "5%")));
    assertSerialization(
        fqn("id/subtype_afrikaans"),
        DataResourceXml.createWithNoNamespace(
            source, SimpleXmlResourceValue.itemWithValue(ResourceType.ID, "0x6f972360")));
  }

  @Test
  public void serializeStyleableXmlResource() throws Exception {
    Path serialized = fs.getPath("out/out.bin");
    Path source = fs.getPath("res/values/values.xml");
    FullyQualifiedName attrKey = fqn("attr/labelPosition");
    DataResourceXml attrValue =
        DataResourceXml.createWithNoNamespace(
            source,
            AttrXmlResourceValue.fromFormatEntries(
                EnumResourceXmlAttrValue.asEntryOf("left", "0", "right", "1")));
    FullyQualifiedName themeKey = fqn("styleable/Theme");
    DataResourceXml themeValue =
        DataResourceXml.createWithNoNamespace(
            source,
            StyleableXmlResourceValue.createAllAttrAsReferences(
                fqnFactory.parse("attr/labelPosition")));

    AndroidDataSerializer serializer = AndroidDataSerializer.create();
    serializer.queueForSerialization(attrKey, attrValue);
    serializer.queueForSerialization(themeKey, themeValue);
    serializer.flushTo(serialized);

    AndroidDataDeserializer deserializer = AndroidParsedDataDeserializer.create();
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    deserializer.read(
        serialized,
        KeyValueConsumers.of(new FakeConsumer(toOverwrite), new FakeConsumer(toCombine), null));
    assertThat(toOverwrite).containsEntry(attrKey, attrValue);
    assertThat(toCombine).containsEntry(themeKey, themeValue);
  }

  @Test
  public void serializePlurals() throws Exception {
    Path path = fs.getPath("res/values/values.xml");
    assertSerialization(
        fqn("plurals/numberOfSongsAvailable"),
        DataResourceXml.createWithNoNamespace(
            path,
            PluralXmlResourceValue.createWithoutAttributes(
                ImmutableMap.of(
                    "one", "%d song found.",
                    "other", "%d songs found."))));
  }

  @Test
  public void serializeArrays() throws Exception {
    Path path = fs.getPath("res/values/values.xml");
    assertSerialization(
        fqn("plurals/numberOfSongsAvailable"),
        DataResourceXml.createWithNoNamespace(
            path,
            PluralXmlResourceValue.createWithoutAttributes(
                ImmutableMap.of(
                    "one", "%d song found.",
                    "other", "%d songs found."))));
    assertSerialization(
        fqn("array/icons"),
        DataResourceXml.createWithNoNamespace(
            path,
            ArrayXmlResourceValue.of(
                ArrayType.ARRAY, "@drawable/home", "@drawable/settings", "@drawable/logout")));
    assertSerialization(
        fqn("array/colors"),
        DataResourceXml.createWithNoNamespace(
            path,
            ArrayXmlResourceValue.of(ArrayType.ARRAY, "#FFFF0000", "#FF00FF00", "#FF0000FF")));
    assertSerialization(
        fqn("array/characters"),
        DataResourceXml.createWithNoNamespace(
            path,
            ArrayXmlResourceValue.of(
                ArrayType.STRING_ARRAY,
                "boojum",
                "snark",
                "bellman",
                "barrister",
                "\\\"billiard-marker\\\"")));
  }

  @Test
  public void serializeAttrFlag() throws Exception {
    assertSerialization(
        fqn("attr/labelPosition"),
        DataResourceXml.createWithNoNamespace(
            fs.getPath("res/values/values.xml"),
            AttrXmlResourceValue.fromFormatEntries(
                FlagResourceXmlAttrValue.asEntryOf(
                    "left", "0",
                    "right", "1"))));
  }

  @Test
  public void serializeId() throws Exception {
    assertSerialization(
        fqn("id/squark"),
        DataResourceXml.createWithNoNamespace(
            fs.getPath("res/values/values.xml"), IdXmlResourceValue.of()));
  }

  @Test
  public void serializePublic() throws Exception {
    assertSerialization(
        fqn("public/park"),
        DataResourceXml.createWithNoNamespace(
            fs.getPath("res/values/public.xml"), PublicXmlResourceValue.of(
                ImmutableMap.of(
                    ResourceType.DIMEN, Optional.of(0x7f040000),
                    ResourceType.STRING, Optional.of(0x7f050000))
            )));
  }

  @Test
  public void serializeStyle() throws Exception {
    assertSerialization(
        fqn("style/snark"),
        DataResourceXml.createWithNoNamespace(
            fs.getPath("res/values/styles.xml"),
            StyleXmlResourceValue.of(null, ImmutableMap.of("look", "boojum"))));
  }

  @Test
  public void assertMultiFormatAttr() throws Exception {
    assertSerialization(
        fqn("attr/labelPosition"),
        DataResourceXml.createWithNoNamespace(
            fs.getPath("res/values/values.xml"),
            AttrXmlResourceValue.fromFormatEntries(
                ColorResourceXmlAttrValue.asEntry(),
                BooleanResourceXmlAttrValue.asEntry(),
                ReferenceResourceXmlAttrValue.asEntry(),
                DimensionResourceXmlAttrValue.asEntry(),
                FloatResourceXmlAttrValue.asEntry(),
                IntegerResourceXmlAttrValue.asEntry(),
                StringResourceXmlAttrValue.asEntry(),
                FractionResourceXmlAttrValue.asEntry())));
  }

  private void assertSerialization(FullyQualifiedName key, DataValue value) throws Exception {
    Path serialized = fs.getPath("out/out.bin");
    Files.deleteIfExists(serialized);
    Path manifestPath = fs.getPath("AndroidManifest.xml");
    Files.deleteIfExists(manifestPath);
    Files.createFile(manifestPath);
    AndroidDataSerializer serializer = AndroidDataSerializer.create();
    serializer.queueForSerialization(key, value);
    serializer.flushTo(serialized);

    AndroidDataDeserializer deserializer = AndroidParsedDataDeserializer.create();
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    deserializer.read(
        serialized,
        KeyValueConsumers.of(new FakeConsumer(toOverwrite), new FakeConsumer(toCombine), null));
    if (key.isOverwritable()) {
      assertThat(toOverwrite).containsEntry(key, value);
      assertThat(toCombine).isEmpty();
    } else {
      assertThat(toCombine).containsEntry(key, value);
      assertThat(toOverwrite).isEmpty();
    }
  }

  private String[] resourcesXmlFrom(Path source, String... lines) {
    return resourcesXmlFrom(ImmutableMap.<String, String>of(), ImmutableMap.<String, String>of(),
        source, lines);
  }

  private String[] resourcesXmlFrom(Map<String, String> namespaces, Map<String, String> attributes,
      Path source, String... lines) {
    FluentIterable<String> xml =
        FluentIterable.of(new String(AndroidDataWriter.PRELUDE))
            .append("<resources")
            .append(
                FluentIterable.from(namespaces.entrySet())
                    .transform(
                        new Function<Map.Entry<String, String>, String>() {
                          @Override
                          public String apply(Map.Entry<String, String> input) {
                            return String.format(
                                " xmlns:%s=\"%s\"", input.getKey(), input.getValue());
                          }
                        })
                    .join(Joiner.on("")))
            .append(
                FluentIterable.from(attributes.entrySet())
                    .transform(
                        new Function<Map.Entry<String, String>, String>() {
                          @Override
                          public String apply(Map.Entry<String, String> input) {
                            return String.format(" %s=\"%s\"", input.getKey(), input.getValue());
                          }
                        })
                    .join(Joiner.on("")));
    if (source == null && (lines == null || lines.length == 0)) {
      xml = xml.append("/>");
    } else {
      xml = xml.append(">");
      if (source != null) {
        xml = xml.append(String.format("<!-- %s --> <eat-comment/>", source));
      }
      if (lines != null) {
        xml = xml.append(lines);
      }
      xml = xml.append(END_RESOURCES);
    }
    return xml.toArray(String.class);
  }

  private Path parsedAndWritten(Path path, FullyQualifiedName... fqns)
      throws XMLStreamException, IOException, MergingException {
    final Map<DataKey, DataResource> toOverwrite = new HashMap<>();
    final Map<DataKey, DataResource> toCombine = new HashMap<>();
    parseResourcesFrom(path, toOverwrite, toCombine);
    Path out = fs.getPath("out");
    if (Files.exists(out)) {
      Files.walkFileTree(
          out,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
                throws IOException {
              Files.delete(file);
              return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path directory, IOException e)
                throws IOException {
              if (e != null) {
                throw e;
              }
              Files.delete(directory);
              return FileVisitResult.CONTINUE;
            }
          });
    }
    // find and write the resource -- the categorization is tested during parsing.
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(out);
    for (FullyQualifiedName fqn : fqns) {
      if (toOverwrite.containsKey(fqn)) {
        toOverwrite.get(fqn).writeResource(fqn, mergedDataWriter);
      } else if (toCombine.containsKey(fqn)) {
        toCombine.get(fqn).writeResource(fqn, mergedDataWriter);
      }
    }
    mergedDataWriter.flush();
    return mergedDataWriter.resourceDirectory().resolve("values/values.xml");
  }

  private static final Subject.Factory<PathsSubject, Path> resourcePaths = PathsSubject::new;

  private static class FakeConsumer
      implements ParsedAndroidData.KeyValueConsumer<DataKey, DataResource> {
    private final Map<DataKey, DataResource> target;

    FakeConsumer(Map<DataKey, DataResource> target) {
      this.target = target;
    }

    @Override
    public void accept(DataKey key, DataResource value) {
      target.put(key, value);
    }
  }
}
