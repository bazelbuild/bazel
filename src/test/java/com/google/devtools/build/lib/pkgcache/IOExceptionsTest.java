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
package com.google.devtools.build.lib.pkgcache;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

import javax.annotation.Nullable;

/**
 * Tests for recovering from IOExceptions thrown by the filesystem when reading BUILD files. Needs
 * its own test class because it uses a custom filesystem.
 */
@RunWith(JUnit4.class)
public class IOExceptionsTest extends PackageLoadingTestCase {

  private static final String FS_ROOT = "/fsg";

  protected TransitivePackageLoader visitor;

  private static final Function<Path, String> NULL_FUNCTION = new Function<Path, String>() {
    @Override
    @Nullable
    public String apply(Path path) {
      return null;
    }
  };

  private Function<Path, String> crashMessage = NULL_FUNCTION;

  @Before
  public final void initializeVisitor() throws Exception {
    setUpSkyframe(ConstantRuleVisibility.PRIVATE, ruleClassProvider.getDefaultsPackageContent());
    this.visitor = skyframeExecutor.pkgLoader();
  }

  protected void syncPackages() throws Exception {
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, rootDirectory);
  }

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(BlazeClock.instance(), new PathFragment(FS_ROOT)) {
      @Override
      public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
        String crash = crashMessage.apply(path);
        if (crash != null) {
          throw new IOException(crash);
        }
        return super.stat(path, followSymlinks);
      }
    };
  }

  @Test
  public void testBasicFailure() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    final Path buildPath = scratch.file("pkg/BUILD",
        "sh_library(name = 'x')");
    crashMessage = new Function<Path, String>() {
      @Override
      public String apply(Path path) {
        if (buildPath.equals(path)) {
          return "custom crash: " + buildPath;
        }
        return null;
      }
    };
    assertFalse(visitor.sync(reporter, ImmutableSet.<Target>of(),
        ImmutableSet.of(Label.parseAbsolute("//pkg:x")), /*keepGoing=*/false,
        /*parallelThreads=*/5, Integer.MAX_VALUE));
    assertContainsEvent("no such package 'pkg'");
    assertEquals(1, eventCollector.count());
    scratch.overwriteFile("pkg/BUILD",
        "# another comment to force reload",
        "sh_library(name = 'x')");
    crashMessage = NULL_FUNCTION;
    syncPackages();
    eventCollector.clear();
    reporter.addHandler(failFastHandler);
    assertTrue(visitor.sync(reporter, ImmutableSet.<Target>of(),
        ImmutableSet.of(Label.parseAbsolute("//pkg:x")), /*keepGoing=*/false,
        /*parallelThreads=*/5, Integer.MAX_VALUE));
    assertNoEvents();
  }


  @Test
  public void testNestedFailure() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file("top/BUILD",
        "sh_library(name = 'top', deps = ['//pkg:x'])");
    final Path buildPath = scratch.file("pkg/BUILD",
        "sh_library(name = 'x')");
    crashMessage = new Function<Path, String>() {
      @Override
      public String apply(Path path) {
        if (buildPath.equals(path)) {
          return "custom crash: " + buildPath;
        }
        return null;
      }
    };
    assertFalse(visitor.sync(reporter, ImmutableSet.<Target>of(),
        ImmutableSet.of(Label.parseAbsolute("//top:top")), /*keepGoing=*/false,
        /*parallelThreads=*/5, Integer.MAX_VALUE));
    assertContainsEvent("no such package 'pkg'");
    // The traditional label visitor does not propagate the original IOException message.
    // assertContainsEvent("custom crash");
    // This fails in Skyframe because the top node has already printed out the error but
    // SkyframeLabelVisitor prints it out again.
    // assertEquals(1, eventCollector.count());
    scratch.overwriteFile("pkg/BUILD",
        "# another comment to force reload",
        "sh_library(name = 'x')");
    crashMessage = NULL_FUNCTION;
    syncPackages();
    eventCollector.clear();
    reporter.addHandler(failFastHandler);
    assertTrue(visitor.sync(reporter, ImmutableSet.<Target>of(),
        ImmutableSet.of(Label.parseAbsolute("//top:top")), /*keepGoing=*/false,
        /*parallelThreads=*/5, Integer.MAX_VALUE));
    assertNoEvents();
  }

  @Test
  public void testOneLevelUpFailure() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    final Path buildPath = scratch.file("top/BUILD",
        "sh_library(name = 'x')");
    buildPath.getParentDirectory().getRelative("pkg").createDirectory();
    crashMessage = new Function<Path, String>() {
      @Override
      public String apply(Path path) {
        if (buildPath.equals(path)) {
          return "custom crash: " + buildPath;
        }
        return null;
      }
    };
    assertFalse(visitor.sync(reporter, ImmutableSet.<Target>of(),
        ImmutableSet.of(Label.parseAbsolute("//top/pkg:x")), /*keepGoing=*/false,
        /*parallelThreads=*/5, Integer.MAX_VALUE));
    assertContainsEvent("no such package 'top/pkg'");
    assertEquals(1, eventCollector.count());
  }
}
