// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder.DefaultPackageSettings;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.List;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Package}. */
@RunWith(JUnit4.class)
public class PackageTest {

  private static final RuleClass FAUX_TEST_CLASS =
      new RuleClass.Builder("faux_test", RuleClassType.TEST, /*starlark=*/ false)
          .addAttribute(
              Attribute.attr("tags", Type.STRING_LIST).nonconfigurable("tags aren't").build())
          .addAttribute(Attribute.attr("size", Type.STRING).nonconfigurable("size isn't").build())
          .addAttribute(Attribute.attr("timeout", Type.STRING).build())
          .addAttribute(Attribute.attr("flaky", Type.BOOLEAN).build())
          .addAttribute(Attribute.attr("shard_count", Type.INTEGER).build())
          .addAttribute(Attribute.attr("local", Type.BOOLEAN).build())
          .setConfiguredTargetFunction(mock(StarlarkCallable.class))
          .build();

  private FileSystem fileSystem;

  @Before
  public void setUp() {
    this.fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
  }

  @Test
  public void testBuildPartialPopulatesImplicitTestSuiteIgnoresManualTests() throws Exception {
    Package.Builder pkgBuilder = pkgBuilder("test_pkg");
    Label testLabel = Label.parseAbsoluteUnchecked("//test_pkg:my_test");
    addRule(pkgBuilder, testLabel, FAUX_TEST_CLASS);

    Label manualTestLabel = Label.parseAbsoluteUnchecked("//test_pkg:my_manual_test");
    Rule tag2Rule = addRule(pkgBuilder, manualTestLabel, FAUX_TEST_CLASS);
    tag2Rule.setAttributeValue(
        FAUX_TEST_CLASS.getAttributeByName("tags"), ImmutableList.of("manual"), /*explicit=*/ true);

    Label taggedTestLabel = Label.parseAbsoluteUnchecked("//test_pkg:my_tagged_test");
    Rule taggedTestRule = addRule(pkgBuilder, taggedTestLabel, FAUX_TEST_CLASS);
    taggedTestRule.setAttributeValue(
        FAUX_TEST_CLASS.getAttributeByName("tags"), ImmutableList.of("tag1"), /*explicit=*/ true);

    Label taggedManualTestLabel = Label.parseAbsoluteUnchecked("//test_pkg:my_tagged_manual_test");
    Rule taggedManualTestRule = addRule(pkgBuilder, taggedManualTestLabel, FAUX_TEST_CLASS);
    taggedManualTestRule.setAttributeValue(
        FAUX_TEST_CLASS.getAttributeByName("tags"),
        ImmutableList.of("manual", "tag1"),
        /*explicit=*/ true);

    List<Label> allTests = pkgBuilder.getTestSuiteImplicitTestsRef(/*tags=*/ ImmutableList.of());
    List<Label> tag1Tests = pkgBuilder.getTestSuiteImplicitTestsRef(ImmutableList.of("tag1"));

    pkgBuilder.buildPartial();

    assertThat(allTests).containsExactly(testLabel, taggedTestLabel);
    assertThat(tag1Tests).containsExactly(taggedTestLabel);
  }

  @Test
  public void testBuildPartialPopulatesImplicitTestSuiteValueOnlyForRequestedTags()
      throws Exception {
    Package.Builder pkgBuilder = pkgBuilder("test_pkg");
    Label tag1Label = Label.parseAbsoluteUnchecked("//test_pkg:my_test_tag_1");
    Rule tag1Rule = addRule(pkgBuilder, tag1Label, FAUX_TEST_CLASS);
    tag1Rule.setAttributeValue(
        FAUX_TEST_CLASS.getAttributeByName("tags"), ImmutableList.of("tag1"), /*explicit=*/ true);

    Label tag2Label = Label.parseAbsoluteUnchecked("//test_pkg:my_test_tag_2");
    Rule tag2Rule = addRule(pkgBuilder, tag2Label, FAUX_TEST_CLASS);
    tag2Rule.setAttributeValue(
        FAUX_TEST_CLASS.getAttributeByName("tags"), ImmutableList.of("tag2"), /*explicit=*/ true);

    List<Label> result = pkgBuilder.getTestSuiteImplicitTestsRef(ImmutableList.of("tag1"));

    pkgBuilder.buildPartial();

    assertThat(result).containsExactly(tag1Label);

    // Neither "tag2" nor empty (all tags) were requested before buildPartial, so they weren't
    // accumulated.
    assertThat(pkgBuilder.getTestSuiteImplicitTestsRef(ImmutableList.of("tag2"))).isEmpty();
    assertThat(pkgBuilder.getTestSuiteImplicitTestsRef(/*tags=*/ ImmutableList.of())).isEmpty();
  }

  @Test
  public void testBuildPartialPopulatesImplicitTestSuitesMatchingTags() throws Exception {
    Package.Builder pkgBuilder = pkgBuilder("test_pkg");
    Label matchingLabel = Label.parseAbsoluteUnchecked("//test_pkg:matching");
    Rule matchingRule = addRule(pkgBuilder, matchingLabel, FAUX_TEST_CLASS);
    matchingRule.setAttributeValue(
        FAUX_TEST_CLASS.getAttributeByName("tags"), ImmutableList.of("tag1"), /*explicit=*/ true);

    Label excludedLabel = Label.parseAbsoluteUnchecked("//test_pkg:excluded");
    Rule excludedRule = addRule(pkgBuilder, excludedLabel, FAUX_TEST_CLASS);
    excludedRule.setAttributeValue(
        FAUX_TEST_CLASS.getAttributeByName("tags"),
        ImmutableList.of("tag1", "tag2"),
        /*explicit=*/ true);

    List<Label> result = pkgBuilder.getTestSuiteImplicitTestsRef(ImmutableList.of("tag1", "-tag2"));

    pkgBuilder.buildPartial();
    assertThat(result).containsExactly(matchingLabel);
  }

  @Test
  public void testBuildPartialPopulatesImplicitTestSuiteValueIdempotently() throws Exception {
    Package.Builder pkgBuilder = pkgBuilder("test_pkg");
    Label testLabel = Label.parseAbsoluteUnchecked("//test_pkg:my_test");
    addRule(pkgBuilder, testLabel, FAUX_TEST_CLASS);

    // Ensure targets are accumulated.
    List<Label> result = pkgBuilder.getTestSuiteImplicitTestsRef(/*tags=*/ ImmutableList.of());

    pkgBuilder.buildPartial();
    assertThat(result).containsExactly(testLabel);

    // Multiple calls are valid - make sure they're safe.
    pkgBuilder.buildPartial();
    assertThat(result).containsExactly(testLabel);
  }

  private Package.Builder pkgBuilder(String name) {
    Package.Builder result =
        new Package.Builder(
            DefaultPackageSettings.INSTANCE,
            PackageIdentifier.createInMainRepo(name),
            "workspace",
            /*noImplicitFileExport=*/ true,
            ImmutableMap.of());
    result.setFilename(
        RootedPath.toRootedPath(
            Root.fromPath(fileSystem.getPath("/irrelevantRoot")), PathFragment.create(name)));
    return result;
  }

  private static Rule addRule(Package.Builder pkgBuilder, Label label, RuleClass ruleClass)
      throws Exception {
    Rule rule =
        pkgBuilder.createRule(
            label,
            ruleClass,
            Location.BUILTIN,
            ImmutableList.of(),
            new AttributeContainer.Mutable(FAUX_TEST_CLASS.getAttributeCount()));
    rule.populateOutputFiles(new StoredEventHandler(), pkgBuilder);
    pkgBuilder.addRule(rule);
    return rule;
  }
}
