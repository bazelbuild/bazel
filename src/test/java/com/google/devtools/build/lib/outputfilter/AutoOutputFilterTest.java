// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.outputfilter;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.outputfilter.AutoOutputFilter.ALL;
import static com.google.devtools.build.lib.outputfilter.AutoOutputFilter.NONE;
import static com.google.devtools.build.lib.outputfilter.AutoOutputFilter.PACKAGES;
import static com.google.devtools.build.lib.outputfilter.AutoOutputFilter.SUBPACKAGES;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.OutputFilter;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the {@link AutoOutputFilter} class.
 */
@RunWith(JUnit4.class)
public class AutoOutputFilterTest extends BuildViewTestCase {
  @Test
  public void testNoneAOF() throws Exception {
    assertThat(NONE.getFilter(targets())).isEqualTo(OutputFilter.OUTPUT_EVERYTHING);
    assertThat(NONE.getFilter(targets("//a"))).isEqualTo(OutputFilter.OUTPUT_EVERYTHING);
    assertThat(NONE.getFilter(targets("//a", "//b"))).isEqualTo(OutputFilter.OUTPUT_EVERYTHING);

    assertThat(ALL.getFilter(targets())).isEqualTo(OutputFilter.OUTPUT_NOTHING);
    assertThat(ALL.getFilter(targets("//a"))).isEqualTo(OutputFilter.OUTPUT_NOTHING);
    assertThat(ALL.getFilter(targets("//a", "//b"))).isEqualTo(OutputFilter.OUTPUT_NOTHING);
  }

  @Test
  public void testPackagesAOF() throws Exception {
    assertFilter("^//():", PACKAGES);
    assertFilter("^//(a):", PACKAGES, "//a:b");
    assertFilter("^//(a):", PACKAGES, "//a:b", "//a:c");
    assertFilter("^//(a|b):", PACKAGES, "//a:a", "//a:b", "//b:c");
    assertFilter("^//(a|b):", PACKAGES, "//a:a", "//b:c", "//a:b");
    assertFilter("^//(a/b|a/b/c):", PACKAGES, "//a/b:b", "//a/b/c:c");
    assertFilter("^//(java(tests)?/a):", PACKAGES, "//java/a");
    assertFilter("^//(java(tests)?/a):", PACKAGES, "//javatests/a");
    assertFilter("^//(java(tests)?/a):", PACKAGES, "//java/a", "//javatests/a");
    assertFilter("^//(java(tests)?/a|java(tests)?/b):", PACKAGES, "//java/a", "//javatests/b");

    assertFilter("^//(a/b|a/b/c):", PACKAGES, "//a/b:b", "//a/b/c:c");
    assertFilter("^//(a|a/b|a/b/c|b):", PACKAGES, "//a", "//a/b", "//a/b/c", "//b");
    assertFilter("^//(a|a/b/c|b|b/c/d):", PACKAGES, "//a", "//a/b/c", "//b", "//b/c/d");

    assertFilter("^//(java(tests)?/a|java(tests)?/a/b):", PACKAGES, "//java/a", "//javatests/a/b");
    assertFilter("^//(java(tests)?/a|java(tests)?/a/b/c):", PACKAGES, "//javatests/a",
        "//java/a/b/c");
  }

  @Test
  public void testSubPackagesAOF() throws Exception {
    assertFilter("^//()[/:]", SUBPACKAGES);
    assertFilter("^//(a)[/:]", SUBPACKAGES, "//a:b");
    assertFilter("^//(a)[/:]", SUBPACKAGES, "//a:b", "//a:c");
    assertFilter("^//(a|b)[/:]", SUBPACKAGES, "//a:a", "//a:b", "//b:c");
    assertFilter("^//(a|b)[/:]", SUBPACKAGES, "//a:a", "//b:c", "//a:b");
    assertFilter("^//(java(tests)?/a)[/:]", SUBPACKAGES, "//java/a");
    assertFilter("^//(java(tests)?/a)[/:]", SUBPACKAGES, "//javatests/a");
    assertFilter("^//(java(tests)?/a)[/:]", SUBPACKAGES, "//java/a", "//javatests/a");
    assertFilter("^//(java(tests)?/a|java(tests)?/b)[/:]", SUBPACKAGES, "//java/a",
        "//javatests/b");

    assertFilter("^//(a/b)[/:]", SUBPACKAGES, "//a/b:b", "//a/b/c:c");
    assertFilter("^//(a|b)[/:]", SUBPACKAGES, "//a", "//a/b", "//a/b/c", "//b");
    assertFilter("^//(a|b)[/:]", SUBPACKAGES, "//a", "//a/b/c", "//b", "//b/c/d");

    assertFilter("^//(java(tests)?/a)[/:]", SUBPACKAGES, "//java/a", "//javatests/a/b");
    assertFilter("^//(java(tests)?/a)[/:]", SUBPACKAGES, "//javatests/a", "//java/a/b/c");
  }

  private void assertFilter(String extractedRegex, AutoOutputFilter autoFilter,
      String... targetLabels) throws Exception {
    OutputFilter filter = autoFilter.getFilter(targets(targetLabels));
    String extraRegex = (autoFilter == AutoOutputFilter.NONE) ? "" : "(unknown)|";
    assertWithMessage("output filter " + autoFilter + " returned wrong filter:")
        .that(filter.toString())
        .isEqualTo(extraRegex + extractedRegex);
  }

  private List<Label> targets(String... targetLabels) throws Exception {
    // Sort targets by package
    ListMultimap<String, String> targetsPerPackage = ArrayListMultimap.create();
    for (String targetName : targetLabels) {
      Label label = Label.parseAbsolute(targetName, ImmutableMap.of());
      targetsPerPackage.put(label.getPackageName(), label.getName());
    }

    // Create BUILD file for each package
    for (String pkg : targetsPerPackage.keySet()) {
      StringBuilder contents = new StringBuilder();
      for (String target : targetsPerPackage.get(pkg)) {
        contents.append("sh_library(name='" + target + "');");
      }
      scratch.overwriteFile(pkg + "/BUILD", contents.toString());
    }

    invalidatePackages();

    // Collect targets
    ImmutableList.Builder<Label> targets = ImmutableList.builder();
    for (String targetName : targetLabels) {
      targets.add(Label.parseAbsolute(targetName, ImmutableMap.of()));
    }
    return targets.build();
  }
}
