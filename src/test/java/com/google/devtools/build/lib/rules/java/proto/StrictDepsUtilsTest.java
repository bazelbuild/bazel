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

package com.google.devtools.build.lib.rules.java.proto;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link StrictDepsUtils}Test */
@RunWith(JUnit4.class)
public class StrictDepsUtilsTest extends BuildViewTestCase {

  @Test
  public void isStrictDepsJavaProtoLibrary_flagIsFalse_noPackageLevelAttribute() throws Exception {
    useConfiguration("--strict_deps_java_protos=false");

    scratch.file(
        "x/BUILD",
        "java_proto_library(name = 'a')",
        "java_proto_library(name = 'b', strict_deps = 0)",
        "java_proto_library(name = 'c', strict_deps = 1)");

    assertThat(StrictDepsUtils.isStrictDepsJavaProtoLibrary(getRuleContext("//x:a"))).isTrue();
    assertThat(StrictDepsUtils.isStrictDepsJavaProtoLibrary(getRuleContext("//x:b"))).isFalse();
    assertThat(StrictDepsUtils.isStrictDepsJavaProtoLibrary(getRuleContext("//x:c"))).isTrue();
  }

  @Test
  public void isStrictDepsJavaProtoLibrary_flagIsTrue_noPackageLevelAttribute() throws Exception {
    useConfiguration("--strict_deps_java_protos=true");

    scratch.file(
        "x/BUILD",
        "java_proto_library(name = 'a')",
        "java_proto_library(name = 'b', strict_deps = 0)",
        "java_proto_library(name = 'c', strict_deps = 1)");

    assertThat(StrictDepsUtils.isStrictDepsJavaProtoLibrary(getRuleContext("//x:a"))).isTrue();
    assertThat(StrictDepsUtils.isStrictDepsJavaProtoLibrary(getRuleContext("//x:b"))).isTrue();
    assertThat(StrictDepsUtils.isStrictDepsJavaProtoLibrary(getRuleContext("//x:c"))).isTrue();
  }

  private RuleContext getRuleContext(String label) throws Exception {
    return getRuleContext(getConfiguredTarget(label));
  }
}
