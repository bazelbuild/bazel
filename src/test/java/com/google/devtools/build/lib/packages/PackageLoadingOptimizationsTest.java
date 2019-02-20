// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for ensuring that optimizations we have during package loading actually occur. */
@RunWith(JUnit4.class)
public class PackageLoadingOptimizationsTest extends PackageLoadingTestCase {
  @Test
  public void attributeListValuesAreDedupedIntraPackage() throws Exception {
    scratch.file(
        "foo/BUILD",
        "L = ['//other:t' + str(i) for i in range(10)]",
        "[sh_library(name = 't' + str(i), deps = L) for i in range(10)]");

    Package fooPkg =
        getPackageManager()
            .getPackage(NullEventHandler.INSTANCE, PackageIdentifier.createInMainRepo("foo"));

    ImmutableList.Builder<ImmutableList<Label>> allListsBuilder = ImmutableList.builder();
    for (Rule ruleInstance : fooPkg.getTargets(Rule.class)) {
      assertThat(ruleInstance.getTargetKind()).isEqualTo("sh_library rule");
      @SuppressWarnings("unchecked")
      ImmutableList<Label> depsList =
          (ImmutableList<Label>) ruleInstance.getAttributeContainer().getAttr("deps");
      allListsBuilder.add(depsList);
    }
    ImmutableList<ImmutableList<Label>> allLists = allListsBuilder.build();
    assertThat(allLists).hasSize(10);
    ImmutableList<Label> firstList = allLists.get(0);
    for (int i = 1; i < allLists.size(); i++) {
      assertThat(allLists.get(i)).isSameAs(firstList);
    }
  }
}
