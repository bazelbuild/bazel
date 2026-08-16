// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac;

import static com.google.common.truth.Truth.assertThat;

import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Options;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link BlazeJavacMain}. */
@RunWith(JUnit4.class)
public final class BlazeJavacMainTest {

  @Test
  public void setupDefaultJavacOptions_jdk21_addsTypeAnnotationsToSymbol() {
    Options options = Options.instance(new Context());

    BlazeJavacMain.setupDefaultJavacOptions(options, 21);

    assertThat(options.get("addTypeAnnotationsToSymbol")).isEqualTo("true");
  }

  @Test
  public void setupDefaultJavacOptions_jdk22_doesNotAddTypeAnnotationsToSymbol() {
    Options options = Options.instance(new Context());

    BlazeJavacMain.setupDefaultJavacOptions(options, 22);

    assertThat(options.get("addTypeAnnotationsToSymbol")).isNull();
  }

  @Test
  public void setupDefaultJavacOptions_keepsExistingDefaults() {
    Options options = Options.instance(new Context());

    BlazeJavacMain.setupDefaultJavacOptions(options, 21);

    assertThat(options.get("-Xlint:path")).isEqualTo("path");
    assertThat(options.get("expandJarClassPaths")).isEqualTo("false");
  }
}
