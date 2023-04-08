// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assume.assumeFalse;

import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildtool.util.SkyframeIntegrationTestBase;
import com.google.devtools.build.lib.cmdline.Label.LabelInterner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.PackageValue;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LabelInternerIntegrationTest extends SkyframeIntegrationTestBase {
  @Test
  public void labelInterner_noDuplicatesAndNoWeakInterned() throws Exception {
    assumeFalse(AnalysisMock.get().isThisBazel());

    LabelInterner labelInterner = Label.getLabelInterner();
    assertThat(labelInterner).isNotNull();

    // Create a structure where package hello depends on dep1 and dep2, package dep1 also depends on
    // dep 2.
    write(
        "hello/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'], deps = ['//dep1:bar', '//dep2:baz'] )");
    write(
        "hello/foo.cc",
        "#include \"dep1/bar.h\"",
        "#include \"dep2/baz.h\"",
        "int main() {",
        "  g();",
        "  return f();",
        "}");
    write(
        "dep1/BUILD",
        "cc_library(name = 'bar', hdrs = ['bar.h'], srcs = ['bar.cc'], deps = ['//dep2:baz'])");
    write(
        "dep1/bar.cc",
        "#include \"bar.h\"",
        "#include \"dep2/baz.h\"",
        "int f() {",
        "  g();",
        "  return 0;",
        "}");
    write("dep1/bar.h", "int f();");
    write("dep2/BUILD", "cc_library(name = 'baz', hdrs = ['baz.h'], srcs = ['baz.cc'])");
    write("dep2/baz.cc", "#include \"baz.h\"", "void g() { }");
    write("dep2/baz.h", "void g();");
    buildTarget("//hello:foo");

    List<Label> allPackageTargetsLabelInstances = new ArrayList<>();
    List<Label> allRuleDepLabelInstances = new ArrayList<>();
    skyframeExecutor().getEvaluator().getInMemoryGraph().getValues().values().stream()
        .filter(PackageValue.class::isInstance)
        .flatMap(v -> ((PackageValue) v).getPackage().getTargets().values().stream())
        .forEach(
            t -> {
              allPackageTargetsLabelInstances.add(t.getLabel());
              if (t instanceof Rule) {
                allRuleDepLabelInstances.addAll(((Rule) t).getLabels());
              }
            });

    // All labels (either `Package#targets` labels or Rule direct dependent labels) should refer to
    // the same Label instance stored in `labelInterner`. So use a strong Interner to track
    // canonical instance of label instances. Equal labels are expected to refer to the same
    // instance.
    Interner<Label> strongInterner = BlazeInterners.newStrongInterner();
    for (Label l : Iterables.concat(allPackageTargetsLabelInstances, allRuleDepLabelInstances)) {
      Label canonical = strongInterner.intern(l);
      assertThat(canonical).isSameInstanceAs(l);
    }

    // Set LabelInterner's GlobalPool to null. Weak intern all `Package#targets` labels should
    // return a different instance since they are not present in weak interner.
    // TODO(b/250641010): Some of the labels in `allRuleDepLabelInstances` are actually weak
    //  interned. Investigate why this is happening and how to resolve.
    LabelInterner.setGlobalPool(null);
    for (Label l : allPackageTargetsLabelInstances) {
      Label duplicate = Label.createUnvalidated(l.getPackageIdentifier(), l.getName());
      assertThat(duplicate).isNotSameInstanceAs(l);
      labelInterner.removeWeak(duplicate);
    }
  }
}
