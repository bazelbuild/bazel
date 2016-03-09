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

import javax.annotation.Nullable;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 *
 * <p>This class contains the data of the {@code java_toolchain} rules, it is a separate object so
 * it can be shared with other tools.
 */
@Immutable
public class JavaToolchainData {

  private final String sourceVersion;
  private final String targetVersion;
  // TODO(cushon): remove @Nullable once migration from --javac_bootclasspath and --javac_extdir
  // is complete, and java_toolchain.{bootclasspath,extclasspath} are mandatory
  @Nullable private final Iterable<String> bootclasspath;
  @Nullable private final Iterable<String> extclasspath;
  private final String encoding;
  private final ImmutableList<String> options;
  private final ImmutableList<String> jvmOpts;

  public JavaToolchainData(
      String sourceVersion,
      String targetVersion,
      @Nullable Iterable<String> bootclasspath,
      @Nullable Iterable<String> extclasspath,
      String encoding,
      List<String> xlint,
      List<String> misc,
      List<String> jvmOpts) {
    this.sourceVersion = checkNotNull(sourceVersion, "sourceVersion must not be null");
    this.targetVersion = checkNotNull(targetVersion, "targetVersion must not be null");
    this.bootclasspath = bootclasspath;
    this.extclasspath = extclasspath;
    this.encoding = checkNotNull(encoding, "encoding must not be null");

    this.jvmOpts = ImmutableList.copyOf(jvmOpts);
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

  @Nullable
  public Iterable<String> getBootclasspath() {
    return bootclasspath;
  }

  @Nullable
  public Iterable<String> getExtclasspath() {
    return extclasspath;
  }

  public String getEncoding() {
    return encoding;
  }
}
