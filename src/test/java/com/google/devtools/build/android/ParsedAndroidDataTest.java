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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.jimfs.Jimfs;
import com.google.common.truth.Subject;
import com.google.common.truth.Truth;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.resources.Visibility;
import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.IdXmlResourceValue;
import com.google.devtools.build.android.xml.ResourcesAttribute;
import com.google.devtools.build.android.xml.SimpleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for ParsedAndroidData */
@RunWith(JUnit4.class)
public class ParsedAndroidDataTest {
  private FileSystem fs;
  private Factory fqnFactory;

  @Before
  public void createCleanEnvironment() {
    fs = Jimfs.newFileSystem();
    fqnFactory = FullyQualifiedName.Factory.from(ImmutableList.<String>of());
  }

  @Test
  public void assets() throws Exception {
    Path root = fs.getPath("transDep");
    ParsedAndroidData dataSet =
        ParsedAndroidData.from(
            ImmutableList.of(
                AndroidDataBuilder.of(root)
                    .addAsset("bin/boojum")
                    .createManifest("AndroidManifest.xml", "com.google.foo", "")
                    .buildDependency()));

    Path assetPath = root.resolve("assets/bin/boojum");
    RelativeAssetPath key = RelativeAssetPath.Factory.of(root.resolve("assets")).create(assetPath);
    assertThat(dataSet.getAssets()).containsEntry(key, DataValueFile.of(assetPath));

    assertThat(dataSet.getCombiningResources()).isEmpty();
    assertThat(dataSet.getOverwritingResources()).isEmpty();
  }

  @Test
  public void assetsConflict() throws Exception {
    Path root = fs.getPath("root");
    Path otherRoot = fs.getPath("otherRoot");

    ParsedAndroidData dataSet =
        ParsedAndroidData.from(
            ImmutableList.of(
                AndroidDataBuilder.of(root)
                    .addAsset("bin/boojum")
                    .createManifest("AndroidManifest.xml", "com.google.foo", "")
                    .buildDependency(),
                AndroidDataBuilder.of(otherRoot)
                    .addAsset("bin/boojum")
                    .createManifest("AndroidManifest.xml", "com.google.foo", "")
                    .buildDependency()));

    DataSource assetSource =
        DataSource.of(DependencyInfo.UNKNOWN, root.resolve("assets/bin/boojum"));
    DataSource otherAssetSource =
        DataSource.of(DependencyInfo.UNKNOWN, otherRoot.resolve("assets/bin/boojum"));
    RelativeAssetPath key =
        RelativeAssetPath.Factory.of(root.resolve("assets")).create(assetSource.getPath());

    Truth.assertAbout(parsedAndroidData)
        .that(dataSet)
        .isEqualTo(
            ParsedAndroidData.of(
                ImmutableSet.of(
                    MergeConflict.of(
                        key,
                        DataValueFile.of(
                            Visibility.UNKNOWN,
                            assetSource,
                            /*fingerprint=*/ null,
                            /*rootXmlNode=*/ null),
                        DataValueFile.of(
                            Visibility.UNKNOWN,
                            otherAssetSource,
                            /*fingerprint=*/ null,
                            /*rootXmlNode=*/ null))),
                ImmutableMap.<DataKey, DataResource>of(),
                ImmutableMap.<DataKey, DataResource>of(),
                ImmutableMap.<DataKey, DataAsset>of(
                    key,
                    DataValueFile.of(
                        Visibility.UNKNOWN,
                        otherAssetSource.overwrite(assetSource),
                        /*fingerprint=*/ null,
                        /*rootXmlNode=*/ null))));
  }

  @Test
  public void fileResources() throws Exception {
    Path root = fs.getPath("transDep");
    ParsedAndroidData dataSet =
        ParsedAndroidData.from(
            ImmutableList.of(
                AndroidDataBuilder.of(root)
                    .addResource("layout/foo.xml", AndroidDataBuilder.ResourceType.LAYOUT, "")
                    .addResourceBinary(
                        "drawable/menu.png", Files.createFile(fs.getPath("menu.png")))
                    .createManifest("AndroidManifest.xml", "com.google.foo", "")
                    .buildDependency()));

    FullyQualifiedName layoutFoo = fqnFactory.parse("layout/foo");
    FullyQualifiedName drawableMenu = fqnFactory.parse("drawable/menu");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(
            layoutFoo, DataValueFile.of(root.resolve("res/layout/foo.xml")),
            drawableMenu, DataValueFile.of(root.resolve("res/drawable/menu.png")));

    assertThat(dataSet.getCombiningResources()).isEmpty();
  }

  @Test
  public void ninePatchResourceName() throws Exception {
    // drawable/foo.9.png should give a resource named "R.drawable.foo"
    Path root = fs.getPath("transDep");
    ParsedAndroidData dataSet =
        ParsedAndroidData.from(
            ImmutableList.of(
                AndroidDataBuilder.of(root)
                    .addResourceBinary(
                        "drawable-hdpi/seven_eight.9.png",
                        Files.createFile(fs.getPath("seven_eight.9.png")))
                    .createManifest("AndroidManifest.xml", "com.google.foo", "")
                    .buildDependency()));

    FullyQualifiedName.Factory fqnFactory =
        FullyQualifiedName.Factory.from(ImmutableList.<String>of("hdpi", "v4"));
    FullyQualifiedName drawable = fqnFactory.parse("drawable/seven_eight");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(
            drawable, DataValueFile.of(root.resolve("res/drawable-hdpi/seven_eight.9.png")));

    assertThat(dataSet.getCombiningResources()).isEmpty();
  }

  @Test
  public void fileResourcesSkipInvalidQualifier() throws Exception {
    Path root = fs.getPath("transDep");
    ParsedAndroidData dataSet =
        ParsedAndroidData.from(
            ImmutableList.of(
                AndroidDataBuilder.of(root)
                    .addResource("layout/foo.xml", AndroidDataBuilder.ResourceType.LAYOUT, "")
                    .addResourceBinary(
                        "drawable-scooby-doo/menu.png", Files.createFile(fs.getPath("menu.png")))
                    .createManifest("AndroidManifest.xml", "com.google.foo", "")
                    .buildDependency()));

    FullyQualifiedName layoutFoo = fqnFactory.parse("layout/foo");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(layoutFoo, DataValueFile.of(root.resolve("res/layout/foo.xml")));

    assertThat(dataSet.getCombiningResources()).isEmpty();
  }

  @Test
  public void nestedFileResources() throws Exception {
    Path parent = fs.getPath("parent");
    Path root = parent.resolve("transDep");
    AndroidDataBuilder.of(root)
        .addResource("layout/foo.xml", AndroidDataBuilder.ResourceType.LAYOUT, "")
        .addResourceBinary("drawable/menu.png", Files.createFile(fs.getPath("menu.png")))
        .createManifest("AndroidManifest.xml", "com.google.foo", "")
        .buildDependency();
    ParsedAndroidData dataSet =
        ParsedAndroidData.from(
            new UnvalidatedAndroidData(
                ImmutableList.of(root),
                ImmutableList.<Path>of(),
                root.resolve("AndroidManifest.xml")));

    FullyQualifiedName layoutFoo = fqnFactory.parse("layout/foo");
    FullyQualifiedName drawableMenu = fqnFactory.parse("drawable/menu");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(
            layoutFoo, DataValueFile.of(root.resolve("res/layout/foo.xml")),
            drawableMenu, DataValueFile.of(root.resolve("res/drawable/menu.png")));

    assertThat(dataSet.getCombiningResources()).isEmpty();
  }

  @Test
  public void xmlAndFileResources() throws Exception {
    Path root = fs.getPath("root");
    DependencyAndroidData dep =
        AndroidDataBuilder.of(root)
            .addResource("layout/foo.xml", AndroidDataBuilder.ResourceType.LAYOUT, "")
            .addResourceBinary("drawable/menu.png", Files.createFile(fs.getPath("menu.png")))
            .addValuesWithAttributes(
                "values/attr.xml",
                ImmutableMap.of("foo", "fooVal"),
                "<string name='exit'>way out</string>",
                "<declare-styleable name='Theme'>",
                "  <attr name=\"labelPosition\" format=\"enum\">",
                "    <enum name=\"left\" value=\"0\"/>",
                "    <enum name=\"right\" value=\"1\"/>",
                "  </attr>",
                "</declare-styleable>",
                "<item name='some_id' type='id'/>")
            .createManifest("AndroidManifest.xml", "com.google.foo", "")
            .buildDependency();

    ParsedAndroidData dataSet = ParsedAndroidData.from(ImmutableList.of(dep));

    FullyQualifiedName layoutFoo = fqnFactory.parse("layout/foo");
    FullyQualifiedName drawableMenu = fqnFactory.parse("drawable/menu");
    FullyQualifiedName attributeFoo = fqnFactory.parse("<resources>/foo");
    FullyQualifiedName stringExit = fqnFactory.parse("string/exit");
    FullyQualifiedName attrLabelPosition = fqnFactory.parse("attr/labelPosition");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(
            layoutFoo, // key
            DataValueFile.of(root.resolve("res/layout/foo.xml")), // value
            drawableMenu, // key
            DataValueFile.of(root.resolve("res/drawable/menu.png")), // value
            attributeFoo, // key
            DataResourceXml.createWithNoNamespace(
                root.resolve("res/values/attr.xml"),
                ResourcesAttribute.of(attributeFoo, "foo", "fooVal")), // value
            stringExit, // key
            DataResourceXml.createWithNoNamespace(
                root.resolve("res/values/attr.xml"),
                SimpleXmlResourceValue.createWithValue(
                    SimpleXmlResourceValue.Type.STRING, "way out")), // value
            attrLabelPosition, // key
            DataResourceXml.createWithNoNamespace(
                root.resolve("res/values/attr.xml"), // value
                AttrXmlResourceValue.of(
                    ImmutableMap.of(
                        "enum",
                        AttrXmlResourceValue.EnumResourceXmlAttrValue.of(
                            ImmutableMap.of("left", "0", "right", "1"))))));
    FullyQualifiedName styleableTheme = fqnFactory.parse("styleable/Theme");
    FullyQualifiedName idSomeId = fqnFactory.parse("id/some_id");
    assertThat(dataSet.getCombiningResources())
        .containsExactly(
            styleableTheme, // key
            DataResourceXml.createWithNoNamespace(
                root.resolve("res/values/attr.xml"),
                StyleableXmlResourceValue.createAllAttrAsDefinitions(
                    fqnFactory.parse("attr/labelPosition"))), // value
            idSomeId, // key
            DataResourceXml.createWithNoNamespace(
                root.resolve("res/values/attr.xml"), IdXmlResourceValue.of())); // value
    assertThat(dataSet.getAssets()).isEmpty();
  }

  @Test
  public void combiningResources() throws Exception {
    Path root = fs.getPath("root");
    DependencyAndroidData dep =
        AndroidDataBuilder.of(root)
            .addResource(
                "values/attr.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<declare-styleable name='Doors'>",
                "  <attr name='egress' />",
                "</declare-styleable>",
                "<declare-styleable name='Doors'>",
                "  <attr name='exit' />",
                "</declare-styleable>",
                "<item name='exit_id' type='id'/>")
            .createManifest("AndroidManifest.xml", "com.google.foo", "")
            .buildDependency();

    ParsedAndroidData dataSet = ParsedAndroidData.from(ImmutableList.of(dep));

    assertThat(dataSet.getOverwritingResources()).isEmpty();
    FullyQualifiedName styleableDoors = fqnFactory.parse("styleable/Doors");
    FullyQualifiedName idExitId = fqnFactory.parse("id/exit_id");
    assertThat(dataSet.getCombiningResources())
        .containsExactly(
            styleableDoors, // key
            DataResourceXml.createWithNoNamespace(
                root.resolve("res/values/attr.xml"),
                StyleableXmlResourceValue.createAllAttrAsReferences(
                    fqnFactory.parse("attr/egress"), fqnFactory.parse("attr/exit"))), // value
            idExitId, // key
            DataResourceXml.createWithNoNamespace(
                root.resolve("res/values/attr.xml"), IdXmlResourceValue.of())); // value
    assertThat(dataSet.getAssets()).isEmpty();
  }

  @Test
  public void xmlAndFileResourcesConflict() throws Exception {
    Path root = fs.getPath("root");
    Path otherRoot = fs.getPath("otherRoot");
    DependencyAndroidData dep =
        AndroidDataBuilder.of(root)
            .addResourceBinary("drawable/menu.png", Files.createFile(fs.getPath("menu1.png")))
            .addValuesWithAttributes(
                "values/attr.xml",
                ImmutableMap.of("foo", "fooVal"),
                "<string name='exit'>way out</string>",
                "<item name='some_id' type='id'/>")
            .createManifest("AndroidManifest.xml", "com.google.foo", "")
            .buildDependency();
    DependencyAndroidData otherDep =
        AndroidDataBuilder.of(otherRoot)
            .addResourceBinary("drawable/menu.png", Files.createFile(fs.getPath("menu2.png")))
            .addValuesWithAttributes(
                "values/attr.xml",
                ImmutableMap.of("foo", "fooVal"),
                "<string name='exit'>way out</string>",
                "<item name='some_id' type='id'/>")
            .createManifest("AndroidManifest.xml", "com.google.foo", "")
            .buildDependency();

    ParsedAndroidData dataSet = ParsedAndroidData.from(ImmutableList.of(dep, otherDep));

    FullyQualifiedName drawableMenu = fqnFactory.parse("drawable/menu");
    FullyQualifiedName stringExit = fqnFactory.parse("string/exit");
    FullyQualifiedName attributeFoo = fqnFactory.parse("<resources>/foo");
    DataSource rootDrawableMenuPath =
        DataSource.of(DependencyInfo.UNKNOWN, root.resolve("res/drawable/menu.png"));
    DataSource otherRootDrawableMenuPath =
        DataSource.of(DependencyInfo.UNKNOWN, otherRoot.resolve("res/drawable/menu.png"));
    DataSource rootValuesPath =
        DataSource.of(DependencyInfo.UNKNOWN, root.resolve("res/values/attr.xml"));
    DataSource otherRootValuesPath =
        DataSource.of(DependencyInfo.UNKNOWN, otherRoot.resolve("res/values/attr.xml"));
    FullyQualifiedName idSomeId = fqnFactory.parse("id/some_id");

    Truth.assertAbout(parsedAndroidData)
        .that(dataSet)
        .isEqualTo(
            ParsedAndroidData.of(
                ImmutableSet.of(
                    MergeConflict.of(
                        drawableMenu,
                        DataValueFile.of(
                            Visibility.UNKNOWN,
                            rootDrawableMenuPath,
                            /*fingerprint=*/ null,
                            /*rootXmlNode=*/ null),
                        DataValueFile.of(
                            Visibility.UNKNOWN,
                            otherRootDrawableMenuPath,
                            /*fingerprint=*/ null,
                            /*rootXmlNode=*/ null)),
                    MergeConflict.of(
                        stringExit,
                        DataResourceXml.createWithNoNamespace(
                            rootValuesPath,
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out")),
                        DataResourceXml.createWithNoNamespace(
                            otherRootValuesPath,
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out"))),
                    MergeConflict.of(
                        attributeFoo,
                        DataResourceXml.createWithNoNamespace(
                            rootValuesPath, ResourcesAttribute.of(attributeFoo, "foo", "fooVal")),
                        DataResourceXml.createWithNoNamespace(
                            otherRootValuesPath,
                            ResourcesAttribute.of(attributeFoo, "foo", "fooVal")))),
                ImmutableMap.<DataKey, DataResource>of(
                    drawableMenu, // key
                    DataValueFile.of(
                        Visibility.UNKNOWN,
                        otherRootDrawableMenuPath.overwrite(rootDrawableMenuPath),
                        /*fingerprint=*/ null,
                        /*rootXmlNode=*/ null), // value
                    attributeFoo, // key
                    DataResourceXml.createWithNoNamespace(
                        otherRootValuesPath.overwrite(rootValuesPath),
                        ResourcesAttribute.of(attributeFoo, "foo", "fooVal")), // value
                    stringExit, // key
                    DataResourceXml.createWithNoNamespace(
                        otherRootValuesPath.overwrite(rootValuesPath),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, /* value= */ "way out"))),
                ImmutableMap.<DataKey, DataResource>of(
                    idSomeId, // key
                    DataResourceXml.createWithNoNamespace(
                        rootValuesPath, IdXmlResourceValue.of()) // value
                    ),
                ImmutableMap.<DataKey, DataAsset>of()));
  }

  @Test
  public void layoutWithIDsForwardDeclared() throws Exception {
    Path root = fs.getPath("root");
    ParsedAndroidData dataSet =
        AndroidDataBuilder.of(root)
            .addResource(
                "layout/some_layout.xml",
                AndroidDataBuilder.ResourceType.LAYOUT,
                "<TextView android:id=\"@+id/MyTextView\"",
                "          android:text=\"@string/walrus\"",
                // Test forward declaration in an attribute other than android:id
                "          android:layout_above=\"@+id/AnotherTextView\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />",
                "<TextView android:id=\"@id/AnotherTextView\"",
                "          android:text=\"@string/walrus\"",
                // Test redundantly having a "+id/MyTextView" in a different attribute.
                "          android:layout_below=\"@+id/MyTextView\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />")
            .addResource(
                "values/strings.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<string name=\"walrus\">I has a bucket</string>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    FullyQualifiedName layoutKey = fqnFactory.parse("layout/some_layout");
    FullyQualifiedName stringKey = fqnFactory.parse("string/walrus");
    Path layoutPath = root.resolve("res/layout/some_layout.xml");
    Path valuesPath = root.resolve("res/values/strings.xml");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(
            layoutKey, DataValueFile.of(layoutPath),
            stringKey,
                DataResourceXml.createWithNoNamespace(
                    valuesPath,
                    SimpleXmlResourceValue.createWithValue(
                        SimpleXmlResourceValue.Type.STRING, "I has a bucket")));
    FullyQualifiedName idTextView = fqnFactory.parse("id/MyTextView");
    FullyQualifiedName idTextView2 = fqnFactory.parse("id/AnotherTextView");
    assertThat(dataSet.getCombiningResources())
        .containsExactly(
            idTextView, DataResourceXml.createWithNoNamespace(layoutPath, IdXmlResourceValue.of()),
            idTextView2,
                DataResourceXml.createWithNoNamespace(layoutPath, IdXmlResourceValue.of()));
  }

  @Test
  public void databindingWithIDs() throws Exception {
    Path root = fs.getPath("root");
    ParsedAndroidData dataSet =
        AndroidDataBuilder.of(root)
            .addResource(
                "layout/databinding_layout.xml",
                AndroidDataBuilder.ResourceType.UNFORMATTED,
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<layout xmlns:android=\"http://schemas.android.com/apk/res/android\">",
                "  <data>",
                "    <variable name=\"handlers\" type=\"com.xyz.Handlers\"/>",
                "    <variable name=\"user\" type=\"com.xyz.User\"/>",
                "  </data>",
                "  <LinearLayout",
                "    android:orientation=\"vertical\"",
                "    android:layout_width=\"match_parent\"",
                "    android:layout_height=\"match_parent\">",
                "    <TextView android:id=\"@+id/MyTextView\"",
                "          android:text=\"@{user.id}\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\"",
                "          android:onClick=\"@{user.isX ? handlers.onClickX : handlers.onClickY}\"",
                "    />",
                "    <TextView android:id=\"@+id/AnotherTextView\"",
                "          android:text=\"@{user.nickName}\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />",
                "  </LinearLayout>",
                "</layout>")
            .createManifest("AndroidManifest.xml", "com.xyz", "")
            .buildParsed();
    FullyQualifiedName layoutKey = fqnFactory.parse("layout/databinding_layout");
    Path layoutPath = root.resolve("res/layout/databinding_layout.xml");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(layoutKey, DataValueFile.of(layoutPath));
    FullyQualifiedName idTextView = fqnFactory.parse("id/MyTextView");
    FullyQualifiedName idTextView2 = fqnFactory.parse("id/AnotherTextView");
    assertThat(dataSet.getCombiningResources())
        .containsExactly(
            idTextView, DataResourceXml.createWithNoNamespace(layoutPath, IdXmlResourceValue.of()),
            idTextView2,
                DataResourceXml.createWithNoNamespace(layoutPath, IdXmlResourceValue.of()));
  }

  @Test
  public void menuWithByteOrderMark() throws Exception {
    Path root = fs.getPath("root");
    ParsedAndroidData dataSet =
        AndroidDataBuilder.of(root)
            .addResource(
                "menu/some_menu.xml",
                AndroidDataBuilder.ResourceType.UNFORMATTED,
                "\uFEFF<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<menu xmlns:android=\"http://schemas.android.com/apk/res/android\">",
                "  <group android:id=\"@+id/some_group\"",
                "     android:checkableBehavior=\"none\"",
                "     android:visible=\"true\"",
                "     android:enabled=\"true\"",
                "     android:menuCategory=\"container\"",
                "     android:orderInCategory=\"50\" >",
                "    <item",
                "       android:orderInCategory=\"100\"",
                "       android:showAsAction=\"never\"",
                "       android:id=\"@+id/action_settings\"",
                "       android:title=\"foo\"",
                "    />",
                "  </group>",
                "</menu>")
            .createManifest("AndroidManifest.xml", "com.xyz", "")
            .buildParsed();
    FullyQualifiedName menuKey = fqnFactory.parse("menu/some_menu");
    Path menuPath = root.resolve("res/menu/some_menu.xml");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(menuKey, DataValueFile.of(menuPath));
    FullyQualifiedName groupIdKey = fqnFactory.parse("id/some_group");
    FullyQualifiedName itemIdKey = fqnFactory.parse("id/action_settings");
    assertThat(dataSet.getCombiningResources())
        .containsExactly(
            groupIdKey, DataResourceXml.createWithNoNamespace(menuPath, IdXmlResourceValue.of()),
            itemIdKey, DataResourceXml.createWithNoNamespace(menuPath, IdXmlResourceValue.of()));
  }

  @Test
  public void drawableWithIDs() throws Exception {
    Path root = fs.getPath("root");
    ParsedAndroidData dataSet =
        AndroidDataBuilder.of(root)
            .addResource(
                "drawable-v21/some_drawable.xml",
                AndroidDataBuilder.ResourceType.UNFORMATTED,
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<animated-selector",
                "    xmlns:android=\"http://schemas.android.com/apk/res/android\">",
                "  <item android:id=\"@+id/focused_state\"",
                "        android:drawable=\"@drawable/a_drawable\"",
                "        android:state_focused=\"true\"/>",
                "  <item android:id=\"@+id/default_state\"",
                "        android:drawable=\"@drawable/a_drawable\"/>",
                "  <!-- specify a transition -->",
                "  <transition",
                "        android:fromId=\"@id/default_state\"",
                // test forward decl
                "        android:toId=\"@+id/pressed_state\">",
                "    <animation-list>",
                "      <item android:duration=\"15\" android:drawable=\"@drawable/a_drawable\"/>",
                "      <item android:duration=\"15\" android:drawable=\"@drawable/a_drawable\"/>",
                "    </animation-list>",
                "  </transition>",
                "  <item android:id=\"@id/pressed_state\"",
                "        android:drawable=\"@drawable/a_drawable\"",
                "        android:state_pressed=\"true\"/>",
                "</animated-selector>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    FullyQualifiedName.Factory fqnV21Factory =
        FullyQualifiedName.Factory.from(ImmutableList.<String>of("v21"));
    FullyQualifiedName drawableKey = fqnV21Factory.parse("drawable/some_drawable");
    Path drawablePath = root.resolve("res/drawable-v21/some_drawable.xml");
    assertThat(dataSet.getOverwritingResources())
        .containsExactly(drawableKey, DataValueFile.of(drawablePath));
    FullyQualifiedName focusedState = fqnV21Factory.parse("id/focused_state");
    FullyQualifiedName defaultState = fqnV21Factory.parse("id/default_state");
    FullyQualifiedName pressedState = fqnV21Factory.parse("id/pressed_state");
    assertThat(dataSet.getCombiningResources())
        .containsExactly(
            focusedState,
                DataResourceXml.createWithNoNamespace(drawablePath, IdXmlResourceValue.of()),
            defaultState,
                DataResourceXml.createWithNoNamespace(drawablePath, IdXmlResourceValue.of()),
            pressedState,
                DataResourceXml.createWithNoNamespace(drawablePath, IdXmlResourceValue.of()));
  }

  @Test
  public void testParseErrorsHaveContext() throws Exception {
    Path root = fs.getPath("root");
    AndroidDataBuilder builder =
        AndroidDataBuilder.of(root)
            .addResource(
                "values/unique_strings.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<string name=\"hello\">Hello</string>",
                "<not_string name=\"invalid_string\">Unrecognized tag</not_string>")
            .addResource(
                "layout/unique_layout.xml",
                AndroidDataBuilder.ResourceType.UNFORMATTED,
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<LinearLayout xmlns:android=\"http://schemas.android.com/apk/res/android\"",
                "    android:orientation=\"vertical\"",
                "    android:layout_width=\"match_parent\"",
                "    android:layout_height=\"match_parent\">",
                "</WrongCloseTag>")
            .addResource(
                "menu/unique_menu.xml",
                AndroidDataBuilder.ResourceType.UNFORMATTED,
                "<?xml version=\"not_a_version\" encoding=\"utf-8\"?>",
                "<menu xmlns:android=\"http://schemas.android.com/apk/res/android\">",
                "</menu>")
            .createManifest("AndroidManifest.xml", "com.xyz", "");
    MergingException e = assertThrows(MergingException.class, () -> builder.buildParsed());
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(MergingException.withMessage("3 Parse Error(s)").getMessage());
      String combinedSuberrors = Joiner.on('\n').join(e.getSuppressed());
      assertThat(combinedSuberrors)
          .contains(fs.getPath("values/unique_strings.xml") + ": ParseError at [row,col]:[3,35]");
      assertThat(combinedSuberrors)
          .contains("unrecognized resource type: <not_string name='invalid_string'>");
      assertThat(combinedSuberrors)
          .contains(fs.getPath("layout/unique_layout.xml") + ": ParseError at [row,col]:[6,3]");
      assertThat(combinedSuberrors).contains("must be terminated by the matching end-tag");
      assertThat(combinedSuberrors)
          .contains(fs.getPath("menu/unique_menu.xml") + ": ParseError at [row,col]:[1,30]");
    assertThat(combinedSuberrors)
        .contains("XML version \"not_a_version\" is not supported, only XML 1.0 is supported");
  }

  @Test
  public void publicResourceValidation() throws Exception {
    Path root = fs.getPath("root");
    AndroidDataBuilder builder =
        AndroidDataBuilder.of(root)
            .addResource(
                "values/missing_name.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<public flame=\"not_name\" type=\"string\" id=\"0x7f050000\" />")
            .addResource(
                "values/missing_type.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<public name=\"yarn\" id=\"0x7f050000\" />")
            .addResource(
                "values/bad_type.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<public name=\"twine\" type=\"tring\" id=\"0x7f050000\" />")
            .addResource(
                "values/invalid_id.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<public name=\"wire\" type=\"string\" id=\"0z7f050000\" />")
            .addResource(
                "values/overflow_id.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<public name=\"hemp\" type=\"string\" id=\"0x7f0500000\" />")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "");
    MergingException e = assertThrows(MergingException.class, () -> builder.buildParsed());
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(MergingException.withMessage("5 Parse Error(s)").getMessage());
      String combinedSuberrors = Joiner.on('\n').join(e.getSuppressed());
      assertThat(combinedSuberrors).contains(fs.getPath("values/missing_name.xml").toString());
      assertThat(combinedSuberrors).contains("resource name is required for public");
      assertThat(combinedSuberrors).contains(fs.getPath("values/missing_type.xml").toString());
      assertThat(combinedSuberrors).contains("missing type attribute");
      assertThat(combinedSuberrors).contains(fs.getPath("values/bad_type.xml").toString());
      assertThat(combinedSuberrors).contains("has invalid type attribute");
      assertThat(combinedSuberrors).contains(fs.getPath("values/invalid_id.xml").toString());
      assertThat(combinedSuberrors).contains("has invalid id number");
      assertThat(combinedSuberrors).contains(fs.getPath("values/overflow_id.xml").toString());
    assertThat(combinedSuberrors).contains("has invalid id number");
  }

  final Subject.Factory<ParsedAndroidDataSubject, ParsedAndroidData> parsedAndroidData =
      ParsedAndroidDataSubject::new;
}
