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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.util.Collection;
import java.util.Set;
import java.util.UUID;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SkyframeLabelVisitorTest extends SkyframeLabelVisitorTestCase {
  @Test
  public void testLabelVisitorDetectsMissingPackages() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    scratch.file(
        "pkg/BUILD", "sh_library(name = 'x', deps = ['//nopkg:y', 'z'])", "sh_library(name = 'z')");

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
    assertLabelsVisited(
        ImmutableSet.of("//bar:foo"), ImmutableSet.of("//bar:foo"), !EXPECT_ERROR, !KEEP_GOING);
    assertThat(sym1.delete()).isTrue();
    FileSystemUtils.ensureSymbolicLink(sym1, sym2);
    syncPackages();
    assertLabelsVisited(
        ImmutableSet.of("//unrelated:unrelated"),
        ImmutableSet.of("//unrelated:unrelated"),
        !EXPECT_ERROR,
        !KEEP_GOING);
    assertThat(sym1.delete()).isTrue();
    FileSystemUtils.ensureSymbolicLink(sym1, path);
    assertThat(symlink.delete()).isTrue();
    symlink = scratch.file("bar/BUILD", "sh_library(name = 'bar')");
    syncPackages();
    assertLabelsVisited(
        ImmutableSet.of("//bar:bar"), ImmutableSet.of("//bar:bar"), !EXPECT_ERROR, !KEEP_GOING);
  }

  @Test
  public void testFailFastLoading() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    Path buildFile =
        scratch.file(
            "pkg/BUILD", "sh_library(name = 'x', deps = ['z', 'z'])", "sh_library(name = 'z')");

    // In the first case below, we will hit see an error on "//pkg:x", and therefore
    // not traverse into "//pkg:z" due to fail-fast.
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x"), ImmutableSet.of("//pkg:x"), EXPECT_ERROR, !KEEP_GOING);
    assertContainsEvent("Label '//pkg:z' is duplicated in the 'deps' attribute of rule 'x'");
    assertLabelsVisitedWithErrors(
        ImmutableSet.of("//pkg:x", "//pkg:z"), ImmutableSet.of("//pkg:x"));

    // Also make sure reloading works if the package has changed, but the names
    // of the targets have not.
    scratch.overwriteFile(
        "pkg/BUILD", "sh_library(name = 'x', deps = ['z'])", "sh_library(name = 'z')");
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);
  }

  @Test
  public void testNewFailure() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    Path buildFile =
        scratch.file("pkg/BUILD", "sh_library(name = 'x', deps = ['z'])", "sh_library(name = 'z')");
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);

    scratch.overwriteFile(
        "pkg/BUILD", "sh_library(name = 'x', deps = ['z', 'z'])", "sh_library(name = 'z')");
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x"), ImmutableSet.of("//pkg:x"), EXPECT_ERROR, !KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x"), ImmutableSet.of("//pkg:x"), EXPECT_ERROR, !KEEP_GOING);
  }

  @Test
  public void testNewTransitiveFailure() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    Path buildFile =
        scratch.file("pkg/BUILD", "sh_library(name = 'x', deps = ['z'])", "sh_library(name = 'z')");
    scratch.file("pkg2/BUILD", "sh_library(name = 'q', deps=['F','F'])", "sh_library(name = 'F')");
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);

    scratch.overwriteFile(
        "pkg/BUILD",
        "sh_library(name = 'x', deps = ['z'])",
        "sh_library(name = 'z', deps = [ '//pkg2:q'])");
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q", "//pkg2:F"),
        ImmutableSet.of("//pkg:x"),
        EXPECT_ERROR,
        KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q", "//pkg2:F"),
        ImmutableSet.of("//pkg:x"),
        EXPECT_ERROR,
        KEEP_GOING);
  }

  @Test
  public void testAddDepInNewPkg() throws Exception {
    Path buildFile =
        scratch.file("pkg/BUILD", "sh_library(name = 'x', deps = ['z'])", "sh_library(name = 'z')");
    scratch.file("pkg2/BUILD", "sh_library(name = 'q')");

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);

    scratch.overwriteFile(
        "pkg/BUILD", "sh_library(name = 'x', deps = ['z', '//pkg2:q'])", "sh_library(name = 'z')");
    buildFile.setLastModifiedTime(buildFile.getLastModifiedTime() + 1);
    syncPackages();

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);
    // Check stability (not redundant).
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg2:q"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
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
        !EXPECT_ERROR,
        !KEEP_GOING);

    syncPackages();

    assertLabelsVisited(
        ImmutableSet.of("//pkg:.", "//pkg:x/."),
        ImmutableSet.of("//pkg:.", "//pkg:x/."),
        !EXPECT_ERROR,
        !KEEP_GOING);
  }

  @Test
  public void testLabelVisitorPlural() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors

    scratch.file(
        "pkg/BUILD",
        "sh_library(name = 'x', deps = ['//nopkg:y', 'z'])",
        "sh_library(name = 'z')",
        "sh_library(name = 'o', deps = ['//nopkg2:o'])");

    assertLabelsVisitedWithErrors(
        ImmutableSet.of("//pkg:x", "//pkg:z", "//pkg:o"), ImmutableSet.of("//pkg:x", "//pkg:o"));
    assertContainsEvent("no such package 'nopkg'");
    assertContainsEvent("no such package 'nopkg2'");
  }

  // Indirectly tests that there are dependencies between packages and their subpackages.
  @Test
  public void testSubpackageBoundaryAdd() throws Exception {
    scratch.file(
        "x/BUILD", "sh_library(name = 'x', deps = ['//x:y/z'])", "sh_library(name = 'y/z')");

    assertLabelsVisited(
        ImmutableSet.of("//x:x", "//x:y/z"), ImmutableSet.of("//x:x"), !EXPECT_ERROR, !KEEP_GOING);

    scratch.file("x/y/BUILD", "sh_library(name = 'z')");
    syncPackages(
        ModifiedFileSet.builder()
            .modify(PathFragment.create("x/y"))
            .modify(PathFragment.create("x/y/BUILD"))
            .build());

    reporter.removeHandler(failFastHandler); // expect errors
    assertLabelsVisitedWithErrors(ImmutableSet.of("//x:x"), ImmutableSet.of("//x:x"));
    assertContainsEvent("Label '//x:y/z' crosses boundary of subpackage 'x/y'");
  }

  // Indirectly tests that there are dependencies between packages and their subpackages.
  @Test
  public void testSubpackageBoundaryDelete() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "x/BUILD", "sh_library(name = 'x', deps = ['//x:y/z'])", "sh_library(name = 'y/z')");
    scratch.file("x/y/BUILD", "sh_library(name = 'z')");
    assertLabelsVisitedWithErrors(ImmutableSet.of("//x:x"), ImmutableSet.of("//x:x"));
    assertContainsEvent("Label '//x:y/z' crosses boundary of subpackage 'x/y'");

    scratch.deleteFile("x/y/BUILD");
    syncPackages(ModifiedFileSet.builder().modify(PathFragment.create("x/y/BUILD")).build());

    reporter.addHandler(failFastHandler); // don't expect errors
    assertLabelsVisited(
        ImmutableSet.of("//x:x", "//x:y/z"), ImmutableSet.of("//x:x"), !EXPECT_ERROR, !KEEP_GOING);
  }

  @Test
  public void testInterruptPending() throws Exception {
    scratch.file("x/BUILD");
    Thread.currentThread().interrupt();

    try {
      assertLabelsVisitedWithErrors(ImmutableSet.of("//x:x"), ImmutableSet.of("//x:BUILD"));
      fail();
    } catch (InterruptedException e) {
      // Expected
    }
  }

  // Regression test for "crash when // encountered in package name".
  @Test
  public void testDoubleSlashInPackageName() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file("x/BUILD", "sh_library(name='x', deps=['//x//y'])");
    assertLabelsVisitedWithErrors(ImmutableSet.of("//x:x"), ImmutableSet.of("//x"));
    assertContainsEvent(
        "//x:x: invalid label '//x//y' in element 0 of attribute "
            + "'deps' in 'sh_library' rule: invalid package name 'x//y': "
            + "package names may not contain '//' path separators");
  }

  // Regression test for "Bazel hangs on input of illegal rule".
  @Test
  public void testCrashInLoadPackageIsReportedEffectively() throws Exception {
    reporter.removeHandler(failFastHandler);
    // Inject a NullPointerException into loadPackage().  This is triggered by
    // any ERROR event.
    reporter.addHandler(
        new EventHandler() {
          @Override
          public void handle(Event event) {
            if (EventKind.ERRORS.contains(event.getKind())) {
              throw new NullPointerException("oops");
            }
          }
        });

    // Visitation of //x reaches package "bad" by many paths.  The first time,
    // loadPackage() crashes (because of the injected NPE).  Previously,
    // on a subsequent visitation, the visitor would get livelocked due the
    // stale PendingEntry stuck in the PackageCache.  With the fix, the NPE is
    // thrown.
    scratch.file("bad/BUILD", "this is a bad build file");
    scratch.file(
        "x/BUILD",
        "sh_library(name='x', ",
        "           deps=['//bad:a', '//bad:b', '//bad:c',",
        "                 '//bad:d', '//bad:e', '//bad:f'])");

    try {
      // Used to get stuck.
      assertLabelsVisitedWithErrors(ImmutableSet.of("//x:x"), ImmutableSet.of("//x"));
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
        "a/BUILD",
        "package_group(name = 'pkgs', includes = ['//not/a/package:pkgs'])",
        "sh_library(name = 'foo', visibility = [':pkgs'])");

    assertLabelsVisitedWithErrors(
        ImmutableSet.of("//a:foo", "//a:pkgs"), ImmutableSet.of("//a:foo"));
    assertContainsEvent(
        "in target '//a:pkgs', no such label '//not/a/package:pkgs': no "
            + "such package 'not/a/package'");
  }

  @Test
  public void testKeepGoing() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "parent/BUILD",
        "sh_library(name = 'parent', deps = ['//child:child'])",
        "invalidbuildsyntax");
    scratch.file("child/BUILD", "sh_library(name = 'child')", "invalidbuildsyntax");
    assertLabelsVisited(
        ImmutableSet.of("//parent:parent", "//child:child"),
        ImmutableSet.of("//parent:parent"),
        EXPECT_ERROR,
        KEEP_GOING);
  }

  /**
   * In the case of Skyframe we print a warning inside SkyframeLabelVisitor because the existing
   * interfaces forces us to do the keep_going + show warning logic there.
   */
  @Test
  public void testNewBuildFileConflict() throws Exception {
    Collection<Event> warnings = assertNewBuildFileConflict();
    assertThat(warnings).hasSize(1);
    assertThat(warnings.iterator().next().toString())
        .contains("errors encountered while loading target '//pkg:x'");
  }

  @Test
  public void testWithNoSubincludes() throws Exception {
    PackageCacheOptions packageCacheOptions = Options.getDefaults(PackageCacheOptions.class);
    packageCacheOptions.defaultVisibility = ConstantRuleVisibility.PRIVATE;
    packageCacheOptions.showLoadingProgress = true;
    packageCacheOptions.globbingThreads = 7;
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(rootDirectory),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
            packageCacheOptions,
            Options.getDefaults(SkylarkSemanticsOptions.class),
            loadingMock.getDefaultsPackageContent(),
            UUID.randomUUID(),
            ImmutableMap.<String, String>of(),
            ImmutableMap.<String, String>of(),
            new TimestampGranularityMonitor(BlazeClock.instance()));
    this.visitor = getSkyframeExecutor().pkgLoader();
    scratch.file("pkg/BUILD", "sh_library(name = 'x', deps = ['z'])", "sh_library(name = 'z')");
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);
    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg:z"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);

    scratch.file("hassub/BUILD", "load('//sub:sub.bzl', 'fct')", "fct()");
    scratch.file("sub/BUILD", "exports_files(['sub'])");
    scratch.file("sub/sub.bzl", "def fct(): native.sh_library(name='zzz')");

    assertLabelsVisited(
        ImmutableSet.of("//hassub:zzz"),
        ImmutableSet.of("//hassub:zzz"),
        !EXPECT_ERROR,
        !KEEP_GOING);
  }

  // Regression test for: "ClassCastException in SkyframeLabelVisitor.sync()"
  @Test
  public void testRootCauseOnInconsistentFilesystem() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("foo/BUILD", "sh_library(name = 'foo', deps = ['//bar:baz/fizz'])");
    Path barBuildFile = scratch.file("bar/BUILD", "sh_library(name = 'bar/baz')");
    Path bazDir = barBuildFile.getParentDirectory().getRelative("baz");
    scratch.file("bar/baz/BUILD");
    FileStatus inconsistentParentFileStatus =
        new FileStatus() {
          @Override
          public boolean isFile() {
            return true;
          }

          @Override
          public boolean isSpecialFile() {
            return false;
          }

          @Override
          public boolean isDirectory() {
            return false;
          }

          @Override
          public boolean isSymbolicLink() {
            return false;
          }

          @Override
          public long getSize() throws IOException {
            return 0;
          }

          @Override
          public long getLastModifiedTime() throws IOException {
            return 0;
          }

          @Override
          public long getLastChangeTime() throws IOException {
            return 0;
          }

          @Override
          public long getNodeId() throws IOException {
            return 0;
          }
        };
    fs.stubStat(bazDir, inconsistentParentFileStatus);
    Set<Label> labels = ImmutableSet.of(Label.parseAbsolute("//foo:foo"));
    getSkyframeExecutor()
        .getPackageManager()
        .newTransitiveLoader()
        .sync(reporter, labels, /*keepGoing=*/ true, /*parallelThreads=*/ 100);
    assertContainsEvent("Inconsistent filesystem operations");
  }
}
