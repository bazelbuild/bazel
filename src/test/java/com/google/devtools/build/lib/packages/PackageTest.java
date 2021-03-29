// Copyright 2021 The Bazel Authors. All rights reserved.
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
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder.DefaultPackageSettings;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Package}. */
@RunWith(JUnit4.class)
public class PackageTest {

  @Test
  public void testBuildPartialPopulatesImplicitTestSuiteValueIdempotently() throws Exception {
    Package.Builder pkgBuilder =
        new Package.Builder(
            DefaultPackageSettings.INSTANCE,
            PackageIdentifier.createInMainRepo("test_pkg"),
            "workspace",
            /*noImplicitFileExport=*/ true,
            ImmutableMap.of());
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    pkgBuilder.setFilename(
        RootedPath.toRootedPath(
            Root.fromPath(fileSystem.getPath("/someRoot")), PathFragment.create("test_pkg")));

    RuleClass fauxTestClass =
        new RuleClass.Builder("faux_test", RuleClassType.TEST, /*starlark=*/ false)
            .addAttribute(
                Attribute.attr("tags", Type.STRING_LIST).nonconfigurable("tags aren't").build())
            .addAttribute(Attribute.attr("size", Type.STRING).build())
            .addAttribute(Attribute.attr("timeout", Type.STRING).build())
            .addAttribute(Attribute.attr("flaky", Type.BOOLEAN).build())
            .addAttribute(Attribute.attr("shard_count", Type.INTEGER).build())
            .addAttribute(Attribute.attr("local", Type.BOOLEAN).build())
            .setConfiguredTargetFunction(mock(StarlarkCallable.class))
            .build();

    Label testLabel = Label.parseAbsoluteUnchecked("//test_pkg:my_test");
    Rule rule =
        pkgBuilder.createRule(
            testLabel,
            fauxTestClass,
            Location.BUILTIN,
            ImmutableList.of(),
            new AttributeContainer.Mutable(fauxTestClass.getAttributeCount()));
    rule.populateOutputFiles(new StoredEventHandler(), pkgBuilder);
    pkgBuilder.addRule(rule);

    pkgBuilder.buildPartial();
    assertThat(pkgBuilder.getTestSuiteImplicitTestsRef()).containsExactly(testLabel);

    // Multiple calls are valid - make sure they're safe.
    pkgBuilder.buildPartial();
    assertThat(pkgBuilder.getTestSuiteImplicitTestsRef()).containsExactly(testLabel);
  }
}
