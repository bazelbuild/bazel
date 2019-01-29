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
import static com.google.devtools.build.android.ParsedAndroidDataBuilder.file;
import static com.google.devtools.build.android.ParsedAndroidDataBuilder.xml;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.jimfs.Jimfs;
import com.google.common.truth.Subject;
import com.google.devtools.build.android.AndroidDataBuilder.ResourceType;
import com.google.devtools.build.android.AndroidDataMerger.MergeConflictException;
import com.google.devtools.build.android.AndroidDataMerger.SourceChecker;
import com.google.devtools.build.android.xml.IdXmlResourceValue;
import com.google.devtools.build.android.xml.PublicXmlResourceValue;
import com.google.devtools.build.android.xml.SimpleXmlResourceValue;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidDataMerger}. */
@RunWith(JUnit4.class)
public class AndroidDataMergerTest {

  static final String XLIFF_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2";
  static final String XLIFF_PREFIX = "xliff";

  private FileSystem fileSystem;
  private FullyQualifiedName.Factory fqnFactory;
  private TestLoggingHandler loggingHandler;
  private Logger mergerLogger;

  @Before
  public void setUp() throws Exception {
    fileSystem = Jimfs.newFileSystem();
    fqnFactory = FullyQualifiedName.Factory.from(ImmutableList.<String>of());
    mergerLogger = Logger.getLogger(AndroidDataMerger.class.getCanonicalName());
    loggingHandler = new TestLoggingHandler();
    mergerLogger.addHandler(loggingHandler);
  }

  @After
  public void removeLoggingHandler() {
    mergerLogger.removeHandler(loggingHandler);
  }

  @Test
  public void mergeDirectDeps() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");

    DataSource primaryStrings = DataSource.of(primaryRoot.resolve("res/values/resources.xml"));
    DataSource directStrings = DataSource.of(directRoot.resolve("res/values/strings.xml"));

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(
                file("layout/exit").source("res/layout/exit.xml"),
                xml("string/exit")
                    .source(directStrings)
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .combining(xml("id/exit").source("values/ids.xml").value(IdXmlResourceValue.of()))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/resources.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    xml("string/exit")
                        .source(primaryStrings.overwrite(directStrings))
                        .value(
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out")))
                .build(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(file("layout/exit").root(directRoot).source("res/layout/exit.xml"))
                .combining(
                    xml("id/exit")
                        .root(directRoot)
                        .source("values/ids.xml")
                        .value(IdXmlResourceValue.of()))
                .build());

    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeDirectAndTransitiveDeps() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    DataSource directString = DataSource.of(directRoot.resolve("res/values/resources.xml"));
    DataSource primaryString =
        DataSource.of(primaryRoot.resolve("res").resolve("values/resources.xml"));

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(file("layout/enter").source("res/layout/enter.xml"))
            .combining(xml("id/exit").source("values/ids.xml").value(IdXmlResourceValue.of()))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(
                file("layout/exit").source("res/layout/exit.xml"),
                xml("string/exit")
                    .source(directString)
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .combining(xml("id/exit").source("values/ids.xml").value(IdXmlResourceValue.of()))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/resources.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    xml("string/exit")
                        .root(primaryRoot)
                        .source(primaryString.overwrite(directString))
                        .value(
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out")))
                .build(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    file("layout/enter").root(transitiveRoot).source("res/layout/enter.xml"),
                    file("layout/exit").root(directRoot).source("res/layout/exit.xml"))
                .combining(
                    xml("id/exit")
                        .root(directRoot)
                        .source("values/ids.xml")
                        .value(IdXmlResourceValue.of()))
                .build());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeWithOverwriteInTransitiveDeps() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");
    Path descendentRoot = fileSystem.getPath("descendent");

    DataSource descendentLayout = DataSource.of(descendentRoot.resolve("res/layout/enter.xml"));
    DataSource transitiveLayout = DataSource.of(transitiveRoot.resolve("res/layout/enter.xml"));

    DataSource primaryString = DataSource.of(primaryRoot.resolve("res/values/resources.xml"));
    DataSource directStrings = DataSource.of(directRoot.resolve("res/values/strings.xml"));

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(file("layout/enter").source(descendentLayout))
            .overwritable(file("layout/enter").source(transitiveLayout.overwrite(descendentLayout)))
            .combining(xml("id/exit").source("values/ids.xml").value(IdXmlResourceValue.of()))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(
                file("layout/exit").source("res/layout/exit.xml"),
                xml("string/exit")
                    .source(directStrings)
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .combining(xml("id/exit").source("values/ids.xml").value(IdXmlResourceValue.of()))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/resources.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData transitive =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    xml("string/exit")
                        .source(primaryString.overwrite(directStrings))
                        .value(
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out")))
                .build(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    file("layout/enter").source(transitiveLayout.overwrite(descendentLayout)),
                    file("layout/exit").root(directRoot).source("res/layout/exit.xml"))
                .combining(
                    xml("id/exit")
                        .root(directRoot)
                        .source("values/ids.xml")
                        .value(IdXmlResourceValue.of()))
                .build());
    assertAbout(unwrittenMergedAndroidData).that(transitive).isEqualTo(expected);
  }

  @Test
  public void mergeDirectConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            // Two string/exit will create conflict.
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")),
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);

    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    fqnFactory.parse("string/exit"),
                    DataResourceXml.createWithNoNamespace(
                        directRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")),
                    DataResourceXml.createWithNoNamespace(
                        directRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")))
                .toConflictMessage());
  }

  @Test
  public void mergeDirectConflictDuplicated() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            // Two string/exit will create conflict.
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "way out")),
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger =
        AndroidDataMerger.createWithDefaultThreadPool(
            new SourceChecker() {
              @Override
              public boolean checkEquality(DataSource one, DataSource two) throws IOException {
                return one.equals(two);
              }
            });

    assertAbout(unwrittenMergedAndroidData)
        .that(merger.merge(transitiveDependency, directDependency, primary, false, true))
        .isEqualTo(
            UnwrittenMergedAndroidData.of(
                primary.getManifest(),
                ParsedAndroidDataBuilder.empty(),
                ParsedAndroidDataBuilder.buildOn(fqnFactory)
                    .overwritable(
                        xml("string/exit")
                            .root(directRoot)
                            .source("values/strings.xml")
                            .value(
                                SimpleXmlResourceValue.createWithValue(
                                    SimpleXmlResourceValue.Type.STRING, "way out")))
                    .build()));
  }

  @Test
  public void mergeDirectConflictDuplicatedWithDifferentSources() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            // Two string/exit will create conflict.
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "way out")),
                xml("string/exit")
                    .source("values/more_strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "way out")),
                xml("string/another_key")
                    .source("values/more_strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "another way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();
    merger.merge(transitiveDependency, directDependency, primary, false, true);

    assertThat(loggingHandler.warnings).isEmpty();
  }

  @Test
  public void mergeDirectConflictWithPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    DataSource primaryStrings = DataSource.of(primaryRoot.resolve("res/values/strings.xml"));
    DataSource directStrings = DataSource.of(directRoot.resolve("res/values/strings.xml"));

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            // Two string/exit will create conflict.
            .overwritable(
                xml("string/exit")
                    .source(directStrings)
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")),
                xml("string/exit")
                    .source(directStrings)
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/strings.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();
    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, true, true);
    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    xml("string/exit")
                        .source(primaryStrings.overwrite(directStrings))
                        .value(
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out")))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeTransitiveConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            // Two string/exit will create conflict.
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")),
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    ParsedAndroidData directDependency = ParsedAndroidDataBuilder.empty();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);

    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    fqnFactory.parse("string/exit"),
                    DataResourceXml.createWithNoNamespace(
                        transitiveRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")),
                    DataResourceXml.createWithNoNamespace(
                        transitiveRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")))
                .toConflictMessage());
  }

  @Test
  public void mergeTransitiveConflictWithPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            // Two string/exit will create conflict.
            .overwritable(
                xml("string/exit")
                    .root(transitiveRoot.resolve("1"))
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")),
                xml("string/exit")
                    .root(transitiveRoot.resolve("2"))
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory).build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/strings.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();
    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, true, true);
    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    xml("string/exit")
                        .root(primaryRoot)
                        .source("values/strings.xml")
                        .value(
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out")))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeDirectAndTransitiveConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);
    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    fqnFactory.parse("string/exit"),
                    DataResourceXml.createWithNoNamespace(
                        directRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")),
                    DataResourceXml.createWithNoNamespace(
                        transitiveRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
                .toConflictMessage());
  }

  @Test
  public void mergeDirectTransitivePrimaryConflictWithoutPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/strings.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);

    FullyQualifiedName fullyQualifiedName = fqnFactory.parse("string/exit");
    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    fullyQualifiedName,
                    DataResourceXml.createWithNoNamespace(
                        transitiveRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")),
                    DataResourceXml.createWithNoNamespace(
                        directRoot.resolve("res/values/strings.xml"),
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")))
                .toConflictMessage());
  }

  @Test
  public void mergeDirectTransitivePrimaryConflictWithThrowOnConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source("values/strings.xml")
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/strings.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    try {
      merger.merge(transitiveDependency, directDependency, primary, false, true);
      throw new Exception("Expected a MergeConflictException!");
    } catch (MergeConflictException e) {
      return;
    }
  }

  @Test
  public void mergeDirectTransitivePrimaryConflictWithPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    DataSource primaryStrings = DataSource.of(primaryRoot.resolve("res/values/strings.xml"));
    DataSource directStrings = DataSource.of(directRoot.resolve("res/values/strings.xml"));
    DataSource transitiveStrings = DataSource.of(transitiveRoot.resolve("res/values/strings.xml"));

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source(transitiveStrings)
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "no way out")))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(
                xml("string/exit")
                    .source(directStrings)
                    .value(
                        SimpleXmlResourceValue.createWithValue(
                            SimpleXmlResourceValue.Type.STRING, "wrong way out")))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/strings.xml", ResourceType.VALUE, "<string name='exit'>way out</string>")
            .buildUnvalidated();

    UnwrittenMergedAndroidData data =
        AndroidDataMerger.createWithDefaults()
            .merge(transitiveDependency, directDependency, primary, true, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(fqnFactory)
                .overwritable(
                    xml("string/exit")
                        .source(primaryStrings.overwrite(directStrings))
                        .value(
                            SimpleXmlResourceValue.createWithValue(
                                SimpleXmlResourceValue.Type.STRING, "way out")))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeDirectAndTransitiveNinepatchConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    // A drawable nine-patch png and plain png with the same base filename will conflict.
    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(file("drawable/rounded_corners").source("drawable/rounded_corners.9.png"))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(file("drawable/rounded_corners").source("drawable/rounded_corners.png"))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();
    merger.merge(transitiveDependency, directDependency, primary, false, false);

    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    fqnFactory.parse("drawable/rounded_corners"),
                    DataValueFile.of(directRoot.resolve("res/drawable/rounded_corners.png")),
                    DataValueFile.of(transitiveRoot.resolve("res/drawable/rounded_corners.9.png")))
                .toConflictMessage());
  }

  @Test
  public void mergeAssetsDirectDeps() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addAsset("bin/boojum")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.builder()
                .assets(file().root(primaryRoot).source("bin/boojum"))
                .build(),
            ParsedAndroidDataBuilder.builder()
                .assets(file().root(directRoot).source("hunting/of/the/snark.txt"))
                .build());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeCombiningResources() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .combining(
                xml("styleable/RubADubDub")
                    .source("values/attrs.xml")
                    .value(
                        StyleableXmlResourceValue.createAllAttrAsReferences(
                            fqnFactory.parse("attr/baker"))))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .combining(
                xml("styleable/RubADubDub")
                    .source("values/attrs.xml")
                    .value(
                        StyleableXmlResourceValue.createAllAttrAsReferences(
                            fqnFactory.parse("attr/candlestickmaker"))))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/attrs.xml",
                ResourceType.VALUE,
                "<declare-styleable name='RubADubDub'>",
                "  <attr name='butcher'/>",
                "</declare-styleable>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(primaryRoot, fqnFactory)
                .combining(
                    xml("styleable/RubADubDub")
                        .source("values/attrs.xml")
                        .value(
                            StyleableXmlResourceValue.createAllAttrAsReferences(
                                fqnFactory.parse("attr/baker"),
                                fqnFactory.parse("attr/butcher"),
                                fqnFactory.parse("attr/candlestickmaker"))))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeCombiningResourcesWithNamespaces() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    // TODO(corysmith): Make conflict uris for a single prefix and error
    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .combining(
                xml("styleable/RubADubDub")
                    .source("values/attrs.xml")
                    .namespace(XLIFF_PREFIX, XLIFF_NAMESPACE)
                    .namespace("tools", "http://schemas.android.com/tools")
                    .value(
                        StyleableXmlResourceValue.createAllAttrAsReferences(
                            fqnFactory.parse("attr/baker"))))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .combining(
                xml("styleable/RubADubDub")
                    .source("values/attrs.xml")
                    .namespace(XLIFF_PREFIX, "wrong uri")
                    .value(
                        StyleableXmlResourceValue.createAllAttrAsReferences(
                            fqnFactory.parse("attr/candlestickmaker"))))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/attrs.xml",
                ResourceType.VALUE,
                "<declare-styleable name='RubADubDub'>",
                "  <attr name='butcher'/>",
                "</declare-styleable>")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(primaryRoot, fqnFactory)
                .combining(
                    xml("styleable/RubADubDub")
                        .source("values/attrs.xml")
                        .namespace(XLIFF_PREFIX, "wrong uri")
                        .namespace("tools", "http://schemas.android.com/tools")
                        .value(
                            StyleableXmlResourceValue.createAllAttrAsReferences(
                                fqnFactory.parse("attr/baker"),
                                fqnFactory.parse("attr/butcher"),
                                fqnFactory.parse("attr/candlestickmaker"))))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeCombiningLayoutIDResources() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    DataSource transitiveLayout =
        DataSource.of(transitiveRoot.resolve("res/layout/transitive.xml"));
    DataSource directLayout = DataSource.of(directRoot.resolve("res/layout/direct.xml"));
    DataSource primaryLayout = DataSource.of(primaryRoot.resolve("res/layout/primary.xml"));

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .overwritable(file("layout/transitive").source(transitiveLayout))
            .combining(
                xml("id/back_door").source(transitiveLayout).value(IdXmlResourceValue.of()),
                xml("id/door").source("values/ids.xml").value(IdXmlResourceValue.of()),
                xml("id/slide").source(transitiveLayout).value(IdXmlResourceValue.of()))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .overwritable(file("layout/zzDirect").source(directLayout))
            .combining(
                xml("id/door").source(directLayout).value(IdXmlResourceValue.of()),
                xml("id/slide").source(directLayout).value(IdXmlResourceValue.of()),
                xml("id/window").source(directLayout).value(IdXmlResourceValue.of()))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "layout/primary.xml",
                ResourceType.LAYOUT,
                "<TextView android:id=\"@+id/door\"",
                "          android:text=\"this way ---> \"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />",
                "<TextView android:id=\"@+id/window\"",
                "          android:text=\"no, not that way\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(primaryRoot, fqnFactory)
                .overwritable(file("layout/primary").source(primaryLayout))
                .combining(
                    xml("id/window").source("layout/primary.xml").value(IdXmlResourceValue.of()),
                    xml("id/door")
                        .root(transitiveRoot)
                        .source("values/ids.xml")
                        .value(IdXmlResourceValue.of()))
                .build(),
            ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
                .overwritable(
                    file("layout/transitive").source(transitiveLayout),
                    file("layout/zzDirect").source(directLayout))
                .combining(
                    xml("id/back_door").source(transitiveLayout).value(IdXmlResourceValue.of()),
                    xml("id/slide").source(directLayout).value(IdXmlResourceValue.of()))
                .build());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeCombiningPublicResources() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot, fqnFactory)
            .combining(
                xml("public/RubADubDub")
                    .source("values/public.xml")
                    .value(
                        PublicXmlResourceValue.create(
                            com.android.resources.ResourceType.STRING, Optional.of(0x7f040000))))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot, fqnFactory)
            .combining(
                xml("public/RubADubDub")
                    .source("values/public.xml")
                    .value(
                        PublicXmlResourceValue.create(
                            com.android.resources.ResourceType.DIMEN, Optional.of(0x7f020000))))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addResource(
                "values/public.xml",
                ResourceType.VALUE,
                "<public name='RubADubDub' type='color' id='0x7f030000' />")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(primaryRoot, fqnFactory)
                .combining(
                    xml("public/RubADubDub")
                        .source("values/public.xml")
                        .value(
                            PublicXmlResourceValue.of(
                                ImmutableMap.of(
                                    com.android.resources.ResourceType.COLOR,
                                    Optional.of(0x7f030000),
                                    com.android.resources.ResourceType.DIMEN,
                                    Optional.of(0x7f020000),
                                    com.android.resources.ResourceType.STRING,
                                    Optional.of(0x7f040000)))))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeAssetsDirectAndTransitiveDeps() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot)
            .assets(file().source("hunting/of/the/jubjub.bird"))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addAsset("bin/boojum")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, true);

    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.builder()
                .assets(file().root(primaryRoot).source("bin/boojum"))
                .build(),
            ParsedAndroidDataBuilder.builder()
                .assets(
                    file().root(directRoot).source("hunting/of/the/snark.txt"),
                    file().root(transitiveRoot).source("hunting/of/the/jubjub.bird"))
                .build());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeAssetsDirectConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRootOne = fileSystem.getPath("directOne");
    Path directRootTwo = fileSystem.getPath("directTwo");

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.builder()
            .assets(
                file().root(directRootOne).source("hunting/of/the/snark.txt"),
                file().root(directRootTwo).source("hunting/of/the/snark.txt"))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);
    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    RelativeAssetPath.Factory.of(directRootOne.resolve("assets"))
                        .create(directRootOne.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(directRootOne.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(directRootTwo.resolve("assets/hunting/of/the/snark.txt")))
                .toConflictMessage());
  }

  @Test
  public void mergeAssetsDirectConflictWithPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRootOne = fileSystem.getPath("directOne");
    Path directRootTwo = fileSystem.getPath("directTwo");

    ParsedAndroidData transitiveDependency = ParsedAndroidDataBuilder.empty();
    String assetFile = "hunting/of/the/snark.txt";

    DataSource primarySource = DataSource.of(primaryRoot.resolve("assets/" + assetFile));
    DataSource directSourceOne = DataSource.of(directRootOne.resolve("assets/" + assetFile));
    DataSource directSourceTwo = DataSource.of(directRootTwo.resolve("assets/" + assetFile));

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.builder()
            .assets(
                file().root(directRootOne).source(directSourceOne),
                file().root(directRootTwo).source(directSourceTwo))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addAsset(assetFile)
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();
    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, true, true);
    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.builder()
                .assets(
                    file()
                        .root(primaryRoot)
                        .source(primarySource.overwrite(directSourceOne, directSourceTwo)))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeAssetsTransitiveConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path transitiveRootOne = fileSystem.getPath("transitiveOne");
    Path transitiveRootTwo = fileSystem.getPath("transitiveTwo");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.builder()
            .assets(
                file().root(transitiveRootOne).source("hunting/of/the/snark.txt"),
                file().root(transitiveRootTwo).source("hunting/of/the/snark.txt"))
            .build();

    ParsedAndroidData directDependency = ParsedAndroidDataBuilder.empty();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);
    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    RelativeAssetPath.Factory.of(transitiveRootOne.resolve("assets"))
                        .create(transitiveRootOne.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(transitiveRootOne.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(transitiveRootTwo.resolve("assets/hunting/of/the/snark.txt")))
                .toConflictMessage());
  }

  @Test
  public void mergeAssetsTransitiveConflictWithPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    ParsedAndroidData directDependency = ParsedAndroidDataBuilder.empty();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addAsset("hunting/of/the/snark.txt")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();
    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, true, true);
    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(primaryRoot)
                .assets(file().source("hunting/of/the/snark.txt"))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  @Test
  public void mergeAssetsDirectAndTransitiveConflict() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);
    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    RelativeAssetPath.Factory.of(directRoot.resolve("assets"))
                        .create(directRoot.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(directRoot.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(transitiveRoot.resolve("assets/hunting/of/the/snark.txt")))
                .toConflictMessage());
  }

  @Test
  public void mergeAssetsDirectTransitivePrimaryConflictWithoutPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addAsset("hunting/of/the/snark.txt")
            .buildUnvalidated();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    merger.merge(transitiveDependency, directDependency, primary, false, false);
    assertThat(loggingHandler.warnings)
        .containsExactly(
            MergeConflict.of(
                    RelativeAssetPath.Factory.of(directRoot.resolve("assets"))
                        .create(directRoot.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(directRoot.resolve("assets/hunting/of/the/snark.txt")),
                    DataValueFile.of(transitiveRoot.resolve("assets/hunting/of/the/snark.txt")))
                .toConflictMessage());
  }

  @Test
  public void mergeAssetsDirectTransitivePrimaryConflictWithPrimaryOverride() throws Exception {
    Path primaryRoot = fileSystem.getPath("primary");
    Path directRoot = fileSystem.getPath("direct");
    Path transitiveRoot = fileSystem.getPath("transitive");

    DataSource primarySource =
        DataSource.of(primaryRoot.resolve("assets/hunting/of/the/snark.txt"));
    DataSource directSource = DataSource.of(directRoot.resolve("assets/hunting/of/the/snark.txt"));

    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(transitiveRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    ParsedAndroidData directDependency =
        ParsedAndroidDataBuilder.buildOn(directRoot)
            .assets(file().source("hunting/of/the/snark.txt"))
            .build();

    UnvalidatedAndroidData primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .addAsset("hunting/of/the/snark.txt")
            .buildUnvalidated();

    UnwrittenMergedAndroidData data =
        AndroidDataMerger.createWithDefaults()
            .merge(transitiveDependency, directDependency, primary, true, true);
    
    UnwrittenMergedAndroidData expected =
        UnwrittenMergedAndroidData.of(
            primary.getManifest(),
            ParsedAndroidDataBuilder.buildOn(primaryRoot)
                .assets(file().source(primarySource.overwrite(directSource)))
                .build(),
            ParsedAndroidDataBuilder.empty());
    assertAbout(unwrittenMergedAndroidData).that(data).isEqualTo(expected);
  }

  final Subject.Factory<UnwrittenMergedAndroidDataSubject, UnwrittenMergedAndroidData>
      unwrittenMergedAndroidData = UnwrittenMergedAndroidDataSubject::new;

  private static final class TestLoggingHandler extends Handler {
    public final List<String> warnings = new ArrayList<String>();

    @Override
    public void publish(LogRecord record) {
      if (record.getLevel().equals(Level.WARNING)) {
        warnings.add(record.getMessage());
      }
    }

    @Override
    public void flush() {}

    @Override
    public void close() throws SecurityException {}
  }
}
