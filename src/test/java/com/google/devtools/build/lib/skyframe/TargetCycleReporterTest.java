// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TargetCycleReporter}. */
@RunWith(JUnit4.class)
public final class TargetCycleReporterTest extends BuildViewTestCase {

  /**
   * Regression test for b/142966884 : Blaze crashes when building with --aspects and --keep_going
   * on a target where the transitive deps have a genquery which finds a cycle over //foo:c that
   * doesn't happen when actually building //foo:c because of a select() on its deps that skips the
   * path that happens to make the cycle.
   *
   * <p>That results in top-level keys that aren't {@link ConfiguredTargetKey} in {@link
   * TargetCycleReporter#getAdditionalMessageAboutCycle}.
   */
  @Test
  public void loadingPhaseCycleWithDifferentTopLevelKeyTypes() throws Exception {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'a', srcs = [], outs = ['a.o'], cmd = 'echo uh > $@')",
        "genrule(name = 'b', srcs = [], outs = ['b.o'], cmd = 'echo hi > $@', visibility = [':c'])",
        "genrule(name = 'c', srcs = [], outs = ['c.o'], cmd = 'echo hi > $@')");
    TargetCycleReporter cycleReporter = new TargetCycleReporter(getPackageManager());
    CycleInfo cycle =
        new CycleInfo(
            ImmutableList.of(
                TransitiveTargetKey.of(Label.parseCanonicalUnchecked("//foo:b")),
                TransitiveTargetKey.of(Label.parseCanonicalUnchecked("//foo:c"))));

    ConfiguredTargetKey ctKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//foo:a"))
            .setConfiguration(targetConfig)
            .build();
    assertThat(cycleReporter.getAdditionalMessageAboutCycle(reporter, ctKey, cycle))
        .contains(
            "The cycle is caused by a visibility edge from //foo:b to the non-package_group "
                + "target //foo:c");

    SkyKey aspectKey = AspectKeyCreator.createAspectKey(null, ctKey);
    assertThat(cycleReporter.getAdditionalMessageAboutCycle(reporter, aspectKey, cycle))
        .contains(
            "The cycle is caused by a visibility edge from //foo:b to the non-package_group "
                + "target //foo:c");

    SkyKey starlarkAspectKey =
        AspectKeyCreator.createTopLevelAspectsKey(
            ImmutableList.of(
                new StarlarkAspectClass(
                    Label.parseCanonicalUnchecked("//foo:b"), "my Starlark key")),
            Label.parseCanonicalUnchecked("//foo:a"),
            targetConfig,
            /* topLevelAspectsParameters= */ ImmutableMap.of());
    assertThat(cycleReporter.getAdditionalMessageAboutCycle(reporter, starlarkAspectKey, cycle))
        .contains(
            "The cycle is caused by a visibility edge from //foo:b to the non-package_group "
                + "target //foo:c");
  }
}
