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
package com.google.devtools.build.lib.query2.common;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.runtime.QuiescingExecutorsImpl;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.common.options.Options;
import java.util.UUID;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class QueryPreloadingTest extends QueryPreloadingTestCase {
  @Test
  public void testLabelVisitorDetectsMissingPackages() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    scratch.file(
        "pkg/BUILD",
        """
        sh_library(name = 'x', deps = ['//nopkg:y', 'z'])
        sh_library(name = 'z')
        """);

    assertLabelsVisitedWithErrors(
        ImmutableSet.of("//pkg:x", "//pkg:z"), ImmutableSet.of("//pkg:x"));
    assertContainsEvent("no such package 'nopkg'");
  }

  /**
   * Tests that Blaze is resilient to changing symlinks between builds. This test is a more
   * "integrated" version of FilesystemValueCheckerTest#testDirtySymlink.
   */
  @Test
  public void testChangingSymlink() throws Exception {
    Path path = scratch.file("foo/BUILD", "sh_library(name = 'foo')");
    Path sym1 = scratch.resolve(rootDirectory + "/sym1/BUILD");
    Path sym2 = scratch.resolve(rootDirectory + "/sym2/BUILD");
    Path symlink = scratch.resolve(rootDirectory + "/bar/BUILD");
    FileSystemUtils.ensureSymbolicLink(symlink, sym1);
    FileSystemUtils.ensureSymbolicLink(sym1, path);
    FileSystemUtils.ensureSymbolicLink(sym2, path);
    scratch.file("unrelated/BUILD", "sh_library(name = 'unrelated')");
    assertLabelsVisited(ImmutableSet.of("//bar:foo"), ImmutableSet.of("//bar:foo"), !KEEP_GOING);
    assertThat(sym1.delete()).isTrue();
    FileSystemUtils.ensureSymbolicLink(sym1, sym2);
    syncPackages();
    assertLabelsVisited(
        ImmutableSet.of("//unrelated:unrelated"),
        ImmutableSet.of("//unrelated:unrelated"),
        !KEEP_GOING);
    assertThat(sym1.delete()).isTrue();
    FileSystemUtils.ensureSymbolicLink(sym1, path);
    assertThat(symlink.delete()).isTrue();
    scratch.file("bar/BUILD", "sh_library(name = 'bar')");
    syncPackages();
    assertLabelsVisited(ImmutableSet.of("//bar:bar"), ImmutableSet.of("//bar:bar"), !KEEP_GOING);
  }

  @Test
  public void testFailFastLoading() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    Path buildFile =
        scratch.file(
            "pkg/BUILD",
            """
            sh_library(name = 'x', deps = ['z', 'z'])
            sh_library(name = 'z')
            """);

    // We expect an error on "//pkg:x". However, we can still finish the evaluation and also return
    // "//pkg:z" even without keep_going.
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
    assertContainsEvent("Label '//pkg:z' is duplicated in the 'deps' attribute of rule 'x'");
    assertLabelsVisitedWithErrors(
        ImmutableSet.of("//pkg:x", "//pkg:z"), ImmutableSet.of("//pkg:x"));

    // Also make sure reloading works if the package has changed, but the names
    // of the targets have not.
    scratch.overwriteFile(
        "pkg/BUILD",
        """
        sh_library(name = 'x', deps = ['z'])
        sh_library(name = 'z')
        """);
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
  }

  @Test
  public void testNewFailure() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    Path buildFile =
        scratch.file(
            "pkg/BUILD",
            """
            sh_library(name = 'x', deps = ['z'])
            sh_library(name = 'z')
            """);
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);

    scratch.overwriteFile(
        "pkg/BUILD",
        """
        sh_library(name = 'x', deps = ['z', 'z'])
        sh_library(name = 'z')
        """);
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();
    // We expect an error on "//pkg:x". However, we can still finish the evaluation and also return
    // "//pkg:z" even without keep_going.
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
    // Also check keep-going.
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        KEEP_GOING);
  }

  @Test
  public void testNewTransitiveFailure() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    Path buildFile =
        scratch.file(
            "pkg/BUILD",
            """
            sh_library(name = 'x', deps = ['z'])
            sh_library(name = 'z')
            """);
    scratch.file(
        "pkg2/BUILD",
        """
        sh_library(name = 'q', deps=['F','F'])
        sh_library(name = 'F')
        """);
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);

    scratch.overwriteFile(
        "pkg/BUILD",
        """
        sh_library(name = 'x', deps = ['z'])
        sh_library(name = 'z', deps = [ '//pkg2:q'])
        """);
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q", "//pkg2:F"),
        ImmutableSet.of("//pkg:x"),
        KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q", "//pkg2:F"),
        ImmutableSet.of("//pkg:x"),
        KEEP_GOING);
  }

  @Test
  public void testAddDepInNewPkg() throws Exception {
    Path buildFile =
        scratch.file(
            "pkg/BUILD",
            """
            sh_library(name = 'x', deps = ['z'])
            sh_library(name = 'z')
            """);
    scratch.file("pkg2/BUILD", "sh_library(name = 'q')");

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);

    scratch.overwriteFile(
        "pkg/BUILD",
        """
        sh_library(name = 'x', deps = ['z', '//pkg2:q'])
        sh_library(name = 'z')
        """);
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
  }

  // Regression test for: "IllegalArgumentException thrown during build."  This happened if "."
  // occurred in a label name segment.
  @Test
  public void testDotLabelName() throws Exception {
    scratch.file("pkg/BUILD", "exports_files(srcs = ['.', 'x/.'])");

    assertLabelsVisited(
        ImmutableSet.of("//pkg:.", "//pkg:x/."),
        ImmutableSet.of("//pkg:.", "//pkg:x/."),
        !KEEP_GOING);

    syncPackages();

    assertLabelsVisited(
        ImmutableSet.of("//pkg:.", "//pkg:x/."),
        ImmutableSet.of("//pkg:.", "//pkg:x/."),
        !KEEP_GOING);
  }

  @Test
  public void testLabelVisitorPlural() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    scratch.file(
        "pkg/BUILD",
        """
        sh_library(name = 'x', deps = ['//nopkg:y', 'z'])
        sh_library(name = 'z')
        sh_library(name = 'o', deps = ['//nopkg2:o'])
        """);

    assertLabelsVisitedWithErrors(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg:o"), ImmutableSet.of("//pkg:x", "//pkg:o"));
    assertContainsEvent("no such package 'nopkg'");
    assertContainsEvent("no such package 'nopkg2'");
  }

  // Indirectly tests that there are dependencies between packages and their subpackages.
  @Test
  public void testSubpackageBoundaryAdd() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        sh_library(name = 'x', deps = ['//foo:y/z'])
        sh_library(name = 'y/z')
        """);

    assertLabelsVisited(
        ImmutableSet.of("//foo:x", "//foo:y/z"), ImmutableSet.of("//foo:x"), !KEEP_GOING);

    scratch.file("foo/y/BUILD", "sh_library(name = 'z')");
    syncPackages(
        ModifiedFileSet.builder()
            .modify(PathFragment.create("foo/y"))
            .modify(PathFragment.create("foo/y/BUILD"))
            .build());

    reporter.removeHandler(failFastHandler); // expect errors
    assertLabelsVisitedWithErrors(ImmutableSet.of("//foo:x"), ImmutableSet.of("//foo:x"));
    assertContainsEvent("Label '//foo:y/z' crosses boundary of subpackage 'foo/y'");
  }

  // Indirectly tests that there are dependencies between packages and their subpackages.
  @Test
  public void testSubpackageBoundaryDelete() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "foo/BUILD",
        """
        sh_library(name = 'x', deps = ['//foo:y/z'])
        sh_library(name = 'y/z')
        """);
    scratch.file("foo/y/BUILD", "sh_library(name = 'z')");
    assertLabelsVisitedWithErrors(ImmutableSet.of("//foo:x"), ImmutableSet.of("//foo:x"));
    assertContainsEvent("Label '//foo:y/z' crosses boundary of subpackage 'foo/y'");

    scratch.deleteFile("foo/y/BUILD");
    syncPackages(ModifiedFileSet.builder().modify(PathFragment.create("foo/y/BUILD")).build());

    reporter.addHandler(failFastHandler); // don't expect errors
    assertLabelsVisited(
        ImmutableSet.of("//foo:x", "//foo:y/z"), ImmutableSet.of("//foo:x"), !KEEP_GOING);
  }

  @Test
  public void testInterruptPending() throws Exception {
    scratch.file("x/BUILD");
    Thread.currentThread().interrupt();

    assertThrows(
        InterruptedException.class,
        () ->
            assertLabelsVisitedWithErrors(ImmutableSet.of("//x:x"), ImmutableSet.of("//x:BUILD")));
  }

  // Regression test for "crash when // encountered in package name".
  @Test
  public void testDoubleSlashInPackageName() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file("foo/BUILD", "sh_library(name='x', deps=['//foo//y'])");
    assertLabelsVisitedWithErrors(ImmutableSet.of("//foo:x"), ImmutableSet.of("//foo:x"));
    assertContainsEvent(
        "//foo:x: invalid label '//foo//y' in element 0 of attribute "
            + "'deps' of 'sh_library': invalid package name 'foo//y': "
            + "package names may not contain '//' path separators");
  }

  // Regression test for "Bazel hangs on input of illegal rule".
  @Test
  public void testCrashInLoadPackageIsReportedEffectively() throws Exception {
    reporter.removeHandler(failFastHandler);
    // Inject a NullPointerException into loadPackage().  This is triggered by
    // any ERROR event.
    reporter.addHandler(
        event -> {
          if (EventKind.ERRORS.contains(event.getKind())) {
            throw new NullPointerException("oops");
          }
        });

    // Visitation of //x reaches package "bad" by many paths.  The first time,
    // loadPackage() crashes (because of the injected NPE).  Previously,
    // on a subsequent visitation, the visitor would get livelocked due the
    // stale PendingEntry stuck in the PackageCache.  With the fix, the NPE is
    // thrown.
    scratch.file("bad/BUILD", "this is a bad build file");
    scratch.file(
        "foo/BUILD",
        """
        sh_library(name='x',
                   deps=['//bad:a', '//bad:b', '//bad:c',
                         '//bad:d', '//bad:e', '//bad:f'])
        """);

    try {
      // Used to get stuck.
      assertLabelsVisitedWithErrors(ImmutableSet.of("//foo:x"), ImmutableSet.of("//foo:x"));
      fail(); // unreachable
    } catch (NullPointerException npe) {
      // This is expected for legacy blaze.
    } catch (RuntimeException re) {
      // This is expected for Skyframe blaze.
      assertThat(re).hasCauseThat().isInstanceOf(NullPointerException.class);
    }
  }

  // Regression test for: "Need better context for missing build file error due to
  // use in visibility rule".
  @Test
  public void testErrorMessageContainsTarget() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    scratch.file(
        "foo/BUILD",
        """
        package_group(name = 'pkgs', includes = ['//not/a/package:pkgs'])
        sh_library(name = 'foo', visibility = [':pkgs'])
        """);

    assertLabelsVisitedWithErrors(
        ImmutableSet.of("//foo:foo", "//foo:pkgs"), ImmutableSet.of("//foo:foo"));
    assertContainsEvent(
        "in target '//foo:pkgs', no such label '//not/a/package:pkgs': no "
            + "such package 'not/a/package'");
  }

  @Test
  public void testKeepGoing() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "parent/BUILD",
        """
        sh_library(name = 'parent', deps = ['//child:child'])
        x = 1//0
        """); // dynamic error
    scratch.file(
        "child/BUILD",
        """
        sh_library(name = 'child')
        x = 1//0
        """); // dynamic error
    assertLabelsVisited(
        ImmutableSet.of("//parent:parent", "//child:child"),
        ImmutableSet.of("//parent:parent"),
        KEEP_GOING);
  }

  @Test
  public void testNewBuildFileConflict() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file("pkg/BUILD", "sh_library(name = 'x', deps = ['//pkg2:q/sub'])");
    scratch.file("pkg2/BUILD", "sh_library(name = 'q/sub')");

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg2:q/sub"), ImmutableSet.of("//pkg:x"), !KEEP_GOING);

    scratch.file("pkg2/q/BUILD");
    syncPackages();

    assertLabelsVisitedWithErrors(ImmutableSet.of("//pkg:x"), ImmutableSet.of("//pkg:x"));
    assertContainsEvent("Label '//pkg2:q/sub' crosses boundary of subpackage 'pkg2/q'");
    assertContainsEvent("no such target '//pkg2:q/sub'");
    // Check stability (not redundant).
    assertLabelsVisitedWithErrors(ImmutableSet.of("//pkg:x"), ImmutableSet.of("//pkg:x"));
    assertContainsEvent("Label '//pkg2:q/sub' crosses boundary of subpackage 'pkg2/q'");
  }

  @Test
  public void testWithNoSubincludes() throws Exception {
    PackageOptions packageOptions = Options.getDefaults(PackageOptions.class);
    packageOptions.defaultVisibility = RuleVisibility.PRIVATE;
    packageOptions.showLoadingProgress = true;
    packageOptions.globbingThreads = 7;
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
            packageOptions,
            Options.getDefaults(BuildLanguageOptions.class),
            UUID.randomUUID(),
            ImmutableMap.of(),
            QuiescingExecutorsImpl.forTesting(),
            new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.of());
    this.visitor = getSkyframeExecutor().getQueryTransitivePackagePreloader();
    scratch.file(
        "pkg/BUILD",
        """
        sh_library(name = 'x', deps = ['z'])
        sh_library(name = 'z')
        """);
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !KEEP_GOING);

    scratch.file(
        "hassub/BUILD",
        """
        load('//sub:sub.bzl', 'fct')
        fct()
        """);
    scratch.file("sub/BUILD", "exports_files(['sub'])");
    scratch.file("sub/sub.bzl", "def fct(): native.sh_library(name='zzz')");

    assertLabelsVisited(
        ImmutableSet.of("//hassub:zzz"),
        ImmutableSet.of("//hassub:zzz"),
        !KEEP_GOING);
  }

}
