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

package com.google.devtools.build.lib.analysis.util;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.Arrays;

/**
 * A writer for a scratch build target and associated source files. Can be parameterized with a rule
 * type for which to write a mock target.
 *
 * <p>For example, the snippet:
 *
 * <pre>{@code
 * new ScratchAttributeWriter(testCase, "cc_library", "//x:x")
 *     .setList("srcs", "a.cc", "b.cc")
 *     .setList("hdrs", "hdr.h")
 *     .write();
 * }</pre>
 *
 * <p>Would create the BUILD file "x/BUILD" with contents:
 *
 * <pre>{@code
 * cc_library(
 *     name = 'x',
 *     srcs = ['a.cc', 'b.cc'],
 *     hdrs = ['hdr.h'],
 * )
 * }</pre>
 */
public class ScratchAttributeWriter {

  private abstract static class ScratchAttribute<T> {
    protected String attributeName;
    protected T attributeValue;

    abstract StringBuilder appendLine(StringBuilder builder);
  }

  /** A plain string attribute. */
  private static class StringAttribute extends ScratchAttribute<String> {
    public StringAttribute(String attributeName, String attributeValue) {
      this.attributeName = attributeName;
      this.attributeValue = attributeValue;
    }

    @Override
    StringBuilder appendLine(StringBuilder builder) {
      return builder.append(String.format("%s=%s,", attributeName, attributeValue));
    }
  }

  /** An integer attribute, such as "alwayslink" */
  private static class IntegerAttribute extends ScratchAttribute<Integer> {

    public IntegerAttribute(String attributeName, Integer attributeValue) {
      this.attributeName = attributeName;
      this.attributeValue = attributeValue;
    }

    @Override
    StringBuilder appendLine(StringBuilder builder) {
      return builder.append(String.format("%s=%d,", attributeName, attributeValue));
    }
  }

  /** A list attribute, such as "srcs" */
  private static class StringListAttribute extends ScratchAttribute<Iterable<String>> {

    public StringListAttribute(String attributeName, Iterable<String> attributeValue) {
      this.attributeName = attributeName;
      this.attributeValue = attributeValue;
    }

    @Override
    StringBuilder appendLine(StringBuilder builder) {
      builder.append(String.format("%s=[", attributeName));
      for (String value : attributeValue) {
        builder.append(String.format("'%s',", value));
      }
      builder.append("],");
      return builder;
    }
  }

  /** The name of the package. */
  private final String packageName;

  /** The name of the target. */
  private final String targetName;

  /** The test case for which to write this target. */
  private final BuildViewTestCase testCase;

  /** The name of the rule for this target */
  private final String ruleName;

  /** An ordered list of the attributes to be written for this scratch target */
  StringBuilder buildString;

  /**
   * Creates a ScratchAttributeWriter for a given test case, package name, and target name. The
   * provided rule name will determine the type of the target written.
   */
  private ScratchAttributeWriter(
      BuildViewTestCase testCase, String ruleName, String packageName, String targetName) {
    this.testCase = checkNotNull(testCase);
    this.ruleName = checkNotNull(ruleName);
    this.packageName = checkNotNull(packageName);
    this.targetName = checkNotNull(targetName);
    this.buildString =
        new StringBuilder()
            .append(String.format("%s(", this.ruleName))
            .append(String.format("name='%s',", this.targetName));
  }

  /**
   * Creates a ScratchAttributeWriter for a given test case and label. The provided rule name will
   * determine the type of the target written.
   */
  public static ScratchAttributeWriter fromLabel(
      BuildViewTestCase testCase, String ruleName, Label label) {
    return new ScratchAttributeWriter(testCase, ruleName, label.getPackageName(), label.getName());
  }

  /**
   * Creates a ScratchAttributeWriter for a given test case and label string. The provided rule name
   * will determine the type of the target written.
   */
  public static ScratchAttributeWriter fromLabelString(
      BuildViewTestCase testCase, String ruleName, String labelString) {
    return fromLabel(testCase, ruleName, Label.parseAbsoluteUnchecked(labelString));
  }

  /**
   * Writes this scratch target to this ScratchAttributeWriter's Scratch instance, and returns the
   * target in the given configuration.
   */
  public ConfiguredTarget write(BuildConfiguration config) throws Exception {
    Scratch scratch = testCase.getScratch();

    buildString.append(")");

    scratch.file(String.format("%s/BUILD", packageName), buildString.toString());
    return testCase.getConfiguredTarget(String.format("//%s:%s", packageName, targetName), config);
  }

  /**
   * Writes this scratch target to this ScratchAttributeWriter's Scratch instance, and returns the
   * target in the target configuration.
   */
  public ConfiguredTarget write() throws Exception {
    return write(testCase.getTargetConfiguration());
  }

  private void createSource(String source) throws IOException {
    testCase.getScratch().file(String.format("%s/%s", packageName, source));
  }

  /** Sets a string attribute (like ios_application.app_icon) for this target. */
  public ScratchAttributeWriter set(String name, String value) {
    new StringAttribute(name, value).appendLine(this.buildString);
    return this;
  }

  /** Sets an integer attribute (like cc_binary.linkstatic) for this target. */
  public ScratchAttributeWriter set(String name, int value) {
    new IntegerAttribute(name, value).appendLine(this.buildString);
    return this;
  }

  /** Sets a list attribute (like cc_library.srcs) for this target. */
  public ScratchAttributeWriter setList(String name, Iterable<String> value) {
    new StringListAttribute(name, value).appendLine(this.buildString);
    return this;
  }

  /** Sets a list attribute (like cc_library.srcs) for this target */
  public ScratchAttributeWriter setList(String name, String... value) {
    return setList(name, Arrays.asList(value));
  }

  /**
   * Sets a list attribute (link cc_library.srcs) for this target. For each string in 'value',
   * writes an empty file to this writer's package with that name.
   *
   * <p>Usually, an analysis-time should not require that referenced files actually be written, in
   * which case ScratchAttributeWriter#set should be used instead.
   */
  public ScratchAttributeWriter setAndCreateFiles(String name, Iterable<String> value)
      throws IOException {
    for (String source : value) {
      createSource(source);
    }
    return setList(name, value);
  }

  /**
   * Sets a list attribute (link cc_library.srcs) for this target. For each string in 'value',
   * writes an empty file to this writer's package with that name.
   *
   * <p>Usually, an analysis-time should not require that referenced files actually be written, in
   * which case ScratchAttributeWriter#set should be used instead.
   */
  public ScratchAttributeWriter setAndCreateFiles(String name, String... value) throws IOException {
    return setAndCreateFiles(name, Arrays.asList(value));
  }
}
