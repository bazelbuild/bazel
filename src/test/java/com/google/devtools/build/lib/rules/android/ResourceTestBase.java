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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.junit.After;
import org.junit.Before;

/** Base class for tests that work with resource artifacts. */
public abstract class ResourceTestBase {
  public static final String RESOURCE_ROOT = "java/android/res";

  private static final ArtifactOwner OWNER = () -> {
    try {
      return Label.create("java", "all");
    } catch (LabelSyntaxException e) {
      assertWithMessage(e.getMessage()).fail();
      return null;
    }
  };

  /** A faked {@link RuleErrorConsumer} that validates that only expected errors were reported. */
  public static final class FakeRuleErrorConsumer implements RuleErrorConsumer {
    private String ruleErrorMessage = null;
    private String attributeErrorAttribute = null;
    private String attributeErrorMessage = null;

    private final List<String> ruleWarnings = new ArrayList<>();

    // Use an ArrayListMultimap since it allows duplicates - we'll want to know if a warning is
    // reported twice.
    private final Multimap<String, String> attributeWarnings = ArrayListMultimap.create();

    @Override
    public void ruleWarning(String message) {
      ruleWarnings.add(message);
    }

    @Override
    public void ruleError(String message) {
      ruleErrorMessage = message;
    }

    @Override
    public void attributeWarning(String attrName, String message) {
      attributeWarnings.put(attrName, message);
    }

    @Override
    public void attributeError(String attrName, String message) {
      attributeErrorAttribute = attrName;
      attributeErrorMessage = message;
    }

    @Override
    public RuleErrorException throwWithRuleError(String message) throws RuleErrorException {
      ruleError(message);
      throw new RuleErrorException();
    }

    @Override
    public RuleErrorException throwWithAttributeError(String attrName, String message)
        throws RuleErrorException {
      attributeError(attrName, message);
      throw new RuleErrorException();
    }

    @Override
    public boolean hasErrors() {
      return ruleErrorMessage != null || attributeErrorMessage != null;
    }

    @Override
    public void assertNoErrors() throws RuleErrorException {
      if (hasErrors()) {
        throw new RuleErrorException();
      }
    }

    public Collection<String> getAndClearRuleWarnings() {
      Collection<String> warnings = ImmutableList.copyOf(ruleWarnings);
      ruleWarnings.clear();
      return warnings;
    }

    public void assertNoRuleWarnings() {
      assertThat(ruleWarnings).isEmpty();
    }

    public Collection<String> getAndClearAttributeWarnings(String attrName) {
      if (!attributeWarnings.containsKey(attrName)) {
        return ImmutableList.of();
      }

      return attributeWarnings.removeAll(attrName);
    }

    public void assertNoAttributeWarnings(String attrName) {
      assertThat(attributeWarnings).doesNotContainKey(attrName);
    }

    /**
     * Called at the end of a test to assert that that test produced a rule error
     *
     * @param expectedMessage a substring of the expected message
     */
    public void assertRuleError(String expectedMessage) {
      // Clear the message before asserting so that if we fail here the error is not masked by the
      // @After call to assertNoUnexpectedErrors.

      String message = ruleErrorMessage;
      ruleErrorMessage = null;

      assertThat(message).contains(expectedMessage);
    }

    /**
     * Called at the end of a test to assert that that test produced an attribute error
     *
     * @param expectedAttribute the attribute that caused the error
     * @param expectedMessage a substring of the expected message
     */
    public void assertAttributeError(String expectedAttribute, String expectedMessage) {
      // Clear the message before asserting so that if we fail here the error is not masked by the
      // @After call to assertNoUnexpectedErrors.
      String attr = attributeErrorAttribute;
      String message = attributeErrorMessage;
      attributeErrorAttribute = null;
      attributeErrorMessage = null;

      assertThat(message).contains(expectedMessage);
      assertThat(attr).isEqualTo(expectedAttribute);
    }

    /**
     * Asserts this {@link RuleErrorConsumer} encountered no unexpected errors. To consume an
     * expected error, call {@link #assertRuleError(String)} or {@link #assertAttributeError(String,
     * String)} in your test after the error is produced.
     */
    private void assertNoUnexpectedErrors() {
      assertThat(ruleErrorMessage).isNull();
      assertThat(attributeErrorMessage).isNull();
      assertThat(attributeErrorAttribute).isNull();
    }
  };

  public FakeRuleErrorConsumer errorConsumer;
  public FileSystem fileSystem;
  public Root root;

  @Before
  public void setup() {
    errorConsumer = new FakeRuleErrorConsumer();
    fileSystem = new InMemoryFileSystem();
    root = Root.asDerivedRoot(fileSystem.getRootDirectory());
  }

  @After
  public void assertNoErrors() {
    errorConsumer.assertNoUnexpectedErrors();
  }

  public ImmutableList<Artifact> getResources(String... pathStrings) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (String pathString : pathStrings) {
      builder.add(getResource(pathString));
    }

    return builder.build();
  }

  public Artifact getResource(String pathString) {
    Path path = fileSystem.getPath("/" + RESOURCE_ROOT + "/" + pathString);
    return new Artifact(
        path, root, root.getExecPath().getRelative(path.relativeTo(root.getPath())), OWNER);
  }
}
