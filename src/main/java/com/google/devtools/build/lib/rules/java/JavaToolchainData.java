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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.List;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 *
 * <p>This class contains the data of the {@code java_toolchain} rules.
 */
// TODO(cushon): inline this into JavaToolchainProvider (it used to be shared with other tools).
@Immutable
public class JavaToolchainData {

  public enum SupportsWorkers {
    NO,
    YES
  }

  private final Iterable<String> bootclasspath;
  private final Iterable<String> extclasspath;
  private final ImmutableList<String> javacopts;
  private final ImmutableList<String> jvmOpts;
  private boolean javacSupportsWorkers;

  public JavaToolchainData(
      Iterable<String> bootclasspath,
      Iterable<String> extclasspath,
      ImmutableList<String> javacopts,
      List<String> jvmOpts,
      SupportsWorkers javacSupportsWorkers) {
    this.bootclasspath = checkNotNull(bootclasspath, "bootclasspath must not be null");
    this.extclasspath = checkNotNull(extclasspath, "extclasspath must not be null");
    this.jvmOpts = ImmutableList.copyOf(jvmOpts);
    this.javacSupportsWorkers = javacSupportsWorkers.equals(SupportsWorkers.YES);
    this.javacopts = checkNotNull(javacopts, "javacopts must not be null");
  }

  /**
   * @return the list of options as given by the {@code java_toolchain} rule.
   */
  public ImmutableList<String> getJavacOptions() {
    return javacopts;
  }

  /**
   * @return the list of options to be given to the JVM when invoking the java compiler and
   *     associated tools.
   */
  public ImmutableList<String> getJvmOptions() {
    return jvmOpts;
  }

  public Iterable<String> getBootclasspath() {
    return bootclasspath;
  }

  public Iterable<String> getExtclasspath() {
    return extclasspath;
  }

  public boolean getJavacSupportsWorkers() {
    return javacSupportsWorkers;
  }
}
