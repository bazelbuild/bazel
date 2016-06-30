// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.List;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 *
 * <p>This class contains the data of the {@code java_toolchain} rules, it is a separate object so
 * it can be shared with other tools.
 */
@Immutable
public class JavaToolchainData {

  public enum SupportsWorkers {
    NO,
    YES
  }

  private final String sourceVersion;
  private final String targetVersion;
  private final Iterable<String> bootclasspath;
  private final Iterable<String> extclasspath;
  private final String encoding;
  private final ImmutableList<String> options;
  private final ImmutableList<String> jvmOpts;
  private boolean javacSupportsWorkers;

  public JavaToolchainData(
      String sourceVersion,
      String targetVersion,
      Iterable<String> bootclasspath,
      Iterable<String> extclasspath,
      String encoding,
      List<String> xlint,
      List<String> misc,
      List<String> jvmOpts,
      SupportsWorkers javacSupportsWorkers) {
    this.sourceVersion = checkNotNull(sourceVersion, "sourceVersion must not be null");
    this.targetVersion = checkNotNull(targetVersion, "targetVersion must not be null");
    this.bootclasspath = checkNotNull(bootclasspath, "bootclasspath must not be null");
    this.extclasspath = checkNotNull(extclasspath, "extclasspath must not be null");
    this.encoding = checkNotNull(encoding, "encoding must not be null");

    this.jvmOpts = ImmutableList.copyOf(jvmOpts);
    this.javacSupportsWorkers = javacSupportsWorkers.equals(SupportsWorkers.YES);
    Builder<String> builder = ImmutableList.<String>builder();
    if (!sourceVersion.isEmpty()) {
      builder.add("-source", sourceVersion);
    }
    if (!targetVersion.isEmpty()) {
      builder.add("-target", targetVersion);
    }
    if (!encoding.isEmpty()) {
      builder.add("-encoding", encoding);
    }
    if (!xlint.isEmpty()) {
      builder.add("-Xlint:" + Joiner.on(",").join(xlint));
    }
    this.options = builder.addAll(misc).build();
  }

  /**
   * @return the list of options as given by the {@code java_toolchain} rule.
   */
  public ImmutableList<String> getJavacOptions() {
    return options;
  }

  /**
   * @return the list of options to be given to the JVM when invoking the java compiler.
   */
  public ImmutableList<String> getJavacJvmOptions() {
    return jvmOpts;
  }

  public String getSourceVersion() {
    return sourceVersion;
  }

  public String getTargetVersion() {
    return targetVersion;
  }

  public Iterable<String> getBootclasspath() {
    return bootclasspath;
  }

  public Iterable<String> getExtclasspath() {
    return extclasspath;
  }

  public String getEncoding() {
    return encoding;
  }

  public boolean getJavacSupportsWorkers() {
    return javacSupportsWorkers;
  }
}
