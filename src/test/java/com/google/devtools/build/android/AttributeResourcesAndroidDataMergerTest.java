// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.android.ParsedAndroidDataBuilder.xml;

import com.android.aapt.Resources.Attribute;
import com.android.aapt.Resources.CompoundValue;
import com.android.aapt.Resources.Value;
import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.jimfs.Jimfs;
import com.google.common.truth.Subject;
import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.ReferenceResourceXmlAttrValue;
import com.google.devtools.build.android.xml.AttrXmlResourceValue.StringResourceXmlAttrValue;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/** Parameterized tests for {@link AndroidDataMerger} focusing on merging attribute resoutces. */
@RunWith(Parameterized.class)
public final class AttributeResourcesAndroidDataMergerTest {

  /** For test data readability, represent strength as a 2-state enum rather than a boolean. */
  enum Strength {
    STRONG,
    WEAK
  }

  @Parameters(name = "{index}: {0}")
  public static Collection<Object[]> data() {
    return Arrays.asList(
        new Object[][] {
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 1)
                .set2(Strength.STRONG, 0xFFFF)
                .setExpectedMergeConflict(
                    ctx ->
                        MergeConflict.of(
                            ctx.fqnFactory.parse("attr/ambiguousName"),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot1.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.fromFormatEntries(
                                    ReferenceResourceXmlAttrValue.asEntry())),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot2.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.fromFormatEntries())))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 1)
                .set2(Strength.WEAK, 0xFFFF)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr1.overwrite(ctx.transitiveAttr2))
                                        .value(
                                            AttrXmlResourceValue.weakFromFormatEntries(
                                                ReferenceResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 0xFFFF)
                .set2(Strength.WEAK, 1)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(
                                            AttrXmlResourceValue.weakFromFormatEntries(
                                                ReferenceResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 1)
                .set2(Strength.STRONG, 1)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(
                                            AttrXmlResourceValue.fromFormatEntries(
                                                ReferenceResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 1)
                .set2(Strength.WEAK, 1)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(
                                            AttrXmlResourceValue.weakFromFormatEntries(
                                                ReferenceResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 1)
                .set2(Strength.WEAK, 0xFFFF)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr1.overwrite(ctx.transitiveAttr2))
                                        .value(
                                            AttrXmlResourceValue.fromFormatEntries(
                                                ReferenceResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 0xFFFF)
                .set2(Strength.STRONG, 2)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(
                                            AttrXmlResourceValue.fromFormatEntries(
                                                StringResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 1)
                .set2(Strength.WEAK, 2)
                .setExpectedMergeConflict(
                    ctx ->
                        MergeConflict.of(
                            ctx.fqnFactory.parse("attr/ambiguousName"),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot1.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.fromFormatEntries(
                                    ReferenceResourceXmlAttrValue.asEntry())),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot2.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.weakFromFormatEntries(
                                    StringResourceXmlAttrValue.asEntry()))))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 1)
                .set2(Strength.STRONG, 2)
                .setExpectedMergeConflict(
                    ctx ->
                        MergeConflict.of(
                            ctx.fqnFactory.parse("attr/ambiguousName"),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot1.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.weakFromFormatEntries(
                                    ReferenceResourceXmlAttrValue.asEntry())),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot2.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.fromFormatEntries(
                                    StringResourceXmlAttrValue.asEntry()))))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 1)
                .set2(Strength.WEAK, 2)
                .setExpectedMergeConflict(
                    ctx ->
                        MergeConflict.of(
                            ctx.fqnFactory.parse("attr/ambiguousName"),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot1.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.weakFromFormatEntries(
                                    ReferenceResourceXmlAttrValue.asEntry())),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot2.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.weakFromFormatEntries(
                                    StringResourceXmlAttrValue.asEntry()))))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 1)
                .set2(Strength.WEAK, 1)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr1.overwrite(ctx.transitiveAttr2))
                                        .value(
                                            AttrXmlResourceValue.fromFormatEntries(
                                                ReferenceResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 1)
                .set2(Strength.STRONG, 1)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(
                                            AttrXmlResourceValue.fromFormatEntries(
                                                ReferenceResourceXmlAttrValue.asEntry())))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 0xFFFF)
                .set2(Strength.STRONG, 0xFFFF)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(AttrXmlResourceValue.fromFormatEntries()))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 0xFFFF)
                .set2(Strength.WEAK, 0xFFFF)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(AttrXmlResourceValue.weakFromFormatEntries()))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 0xFFFF)
                .set2(Strength.WEAK, 0xFFFF)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr1.overwrite(ctx.transitiveAttr2))
                                        .value(AttrXmlResourceValue.fromFormatEntries()))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 0xFFFF)
                .set2(Strength.STRONG, 0xFFFF)
                .setExpectedMergedAndroidData(
                    ctx ->
                        UnwrittenMergedAndroidData.of(
                            ctx.primary.getManifest(),
                            ParsedAndroidDataBuilder.empty(),
                            ParsedAndroidDataBuilder.buildOn(ctx.fqnFactory)
                                .overwritable(
                                    xml("attr/ambiguousName")
                                        .root(ctx.primaryRoot)
                                        .source(ctx.transitiveAttr2.overwrite(ctx.transitiveAttr1))
                                        .value(AttrXmlResourceValue.fromFormatEntries()))
                                .build()))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.STRONG, 0xFFFF)
                .set2(Strength.WEAK, 1)
                .setExpectedMergeConflict(
                    ctx ->
                        MergeConflict.of(
                            ctx.fqnFactory.parse("attr/ambiguousName"),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot1.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.fromFormatEntries()),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot2.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.weakFromFormatEntries(
                                    ReferenceResourceXmlAttrValue.asEntry()))))
                .build()
          },
          {
            TestParameters.builder()
                .set1(Strength.WEAK, 1)
                .set2(Strength.STRONG, 0xFFFF)
                .setExpectedMergeConflict(
                    ctx ->
                        MergeConflict.of(
                            ctx.fqnFactory.parse("attr/ambiguousName"),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot1.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.weakFromFormatEntries(
                                    ReferenceResourceXmlAttrValue.asEntry())),
                            DataResourceXml.createWithNoNamespace(
                                ctx.transitiveRoot2.resolve("res/values/attrs.xml"),
                                AttrXmlResourceValue.fromFormatEntries())))
                .build()
          }
        });
  }

  @Parameter public TestParameters testParameters;

  private FullyQualifiedName.Factory fqnFactory;
  private TestLoggingHandler loggingHandler;
  private Logger mergerLogger;
  private Path primaryRoot;
  private Path transitiveRoot1;
  private Path transitiveRoot2;
  private DataSource transitiveAttr1;
  private DataSource transitiveAttr2;
  private UnvalidatedAndroidData primary;

  @Before
  public void setUp() throws Exception {
    FileSystem fileSystem = Jimfs.newFileSystem();
    fqnFactory = FullyQualifiedName.Factory.from(ImmutableList.of());
    mergerLogger = Logger.getLogger(AndroidDataMerger.class.getCanonicalName());
    loggingHandler = new TestLoggingHandler();
    mergerLogger.addHandler(loggingHandler);
    primaryRoot = fileSystem.getPath("primary");
    transitiveRoot1 = fileSystem.getPath("transitive1");
    transitiveRoot2 = fileSystem.getPath("transitive2");
    transitiveAttr1 = DataSource.of(transitiveRoot1.resolve("res").resolve("values/attrs.xml"));
    transitiveAttr2 = DataSource.of(transitiveRoot2.resolve("res").resolve("values/attrs.xml"));
    primary =
        AndroidDataBuilder.of(primaryRoot)
            .createManifest("AndroidManifest.xml", "com.google.mergetest")
            .buildUnvalidated();
  }

  @After
  public void removeLoggingHandler() {
    mergerLogger.removeHandler(loggingHandler);
  }

  @Test
  public void test() throws Exception {
    ParsedAndroidData transitiveDependency =
        ParsedAndroidDataBuilder.buildOn(fqnFactory)
            .overwritable(
                xml("attr/ambiguousName")
                    .root(transitiveRoot1)
                    .source("values/attrs.xml")
                    .value(
                        AttrXmlResourceValue.from(
                            Value.newBuilder()
                                .setCompoundValue(
                                    CompoundValue.newBuilder()
                                        .setAttr(
                                            Attribute.newBuilder()
                                                .setFormatFlags(testParameters.formatFlags1())))
                                .setWeak(testParameters.strength1() == Strength.WEAK)
                                .build())))
            .overwritable(
                xml("attr/ambiguousName")
                    .root(transitiveRoot2)
                    .source("values/attrs.xml")
                    .value(
                        AttrXmlResourceValue.from(
                            Value.newBuilder()
                                .setCompoundValue(
                                    CompoundValue.newBuilder()
                                        .setAttr(
                                            Attribute.newBuilder()
                                                .setFormatFlags(testParameters.formatFlags2())))
                                .setWeak(testParameters.strength2() == Strength.WEAK)
                                .build())))
            .build();

    ParsedAndroidData directDependency = ParsedAndroidDataBuilder.empty();

    AndroidDataMerger merger = AndroidDataMerger.createWithDefaults();

    UnwrittenMergedAndroidData data =
        merger.merge(transitiveDependency, directDependency, primary, false, false);

    if (testParameters.expectedMergedAndroidData().isPresent()) {
      assertAbout(unwrittenMergedAndroidData)
          .that(data)
          .isEqualTo(testParameters.expectedMergedAndroidData().get().apply(this));
    }

    if (testParameters.expectedMergeConflict().isPresent()) {
      assertThat(loggingHandler.warnings)
          .containsExactly(
              testParameters.expectedMergeConflict().get().apply(this).toConflictMessage());
    } else {
      assertThat(loggingHandler.warnings).isEmpty();
    }
  }

  private final Subject.Factory<UnwrittenMergedAndroidDataSubject, UnwrittenMergedAndroidData>
      unwrittenMergedAndroidData = UnwrittenMergedAndroidDataSubject::new;

  private static final class TestLoggingHandler extends Handler {
    public final List<String> warnings = new ArrayList<>();

    @Override
    public void publish(LogRecord record) {
      if (record.getLevel().equals(Level.WARNING)) {
        warnings.add(record.getMessage());
      }
    }

    @Override
    public void flush() {}

    @Override
    public void close() {}
  }

  @AutoValue
  abstract static class TestParameters {
    abstract Strength strength1();

    abstract int formatFlags1();

    abstract Strength strength2();

    abstract int formatFlags2();

    abstract Optional<Function<AttributeResourcesAndroidDataMergerTest, UnwrittenMergedAndroidData>>
        expectedMergedAndroidData();

    abstract Optional<Function<AttributeResourcesAndroidDataMergerTest, MergeConflict>>
        expectedMergeConflict();

    @Override
    public final String toString() {
      return Joiner.on(", ")
          .join(
              strength1(),
              formatFlags1(),
              strength2(),
              formatFlags2(),
              expectedMergeConflict().isPresent()
                  ? "conflict expected"
                  : "successful merge expected");
    }

    static Builder builder() {
      return new AutoValue_AttributeResourcesAndroidDataMergerTest_TestParameters.Builder();
    }

    @AutoValue.Builder
    abstract static class Builder {

      Builder set1(Strength strength, int formatFlags) {
        return setStrength1(strength).setFormatFlags1(formatFlags);
      }

      Builder set2(Strength strength, int formatFlags) {
        return setStrength2(strength).setFormatFlags2(formatFlags);
      }

      abstract Builder setStrength1(Strength value);

      abstract Builder setFormatFlags1(int value);

      abstract Builder setStrength2(Strength value);

      abstract Builder setFormatFlags2(int value);

      abstract Builder setExpectedMergedAndroidData(
          Function<AttributeResourcesAndroidDataMergerTest, UnwrittenMergedAndroidData> value);

      abstract Builder setExpectedMergeConflict(
          Function<AttributeResourcesAndroidDataMergerTest, MergeConflict> value);

      abstract TestParameters build();
    }
  }
}
