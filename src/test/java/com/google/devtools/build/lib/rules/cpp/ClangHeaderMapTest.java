// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.testutil.FoundationTestCase;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;


public class ClangHeaderMapTest extends FoundationTestCase {
  @Test
  public void testNamespacedHeaderMap() {
    Map<String, String> paths = new HashMap();
    paths.put("MyHeader.h", "include/MyHeader.h");
    paths.put("X/MyHeader2.h", "include/MyHeader2.h");
    paths.put("X/MyHeader3.h", "include/MyHeader3.h");

    ClangHeaderMap hmap = new ClangHeaderMap(paths);
    assertThat(hmap.get("MyHeader.h")).isEqualTo("include/MyHeader.h");
    assertThat(hmap.get("X/MyHeader2.h")).isEqualTo("include/MyHeader2.h");
    assertThat(hmap.get("X/MyHeader3.h")).isEqualTo("include/MyHeader3.h");
  }
}

