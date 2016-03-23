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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Tests for {@link ContainingPackageLookupFunction}.
 */
@RunWith(JUnit4.class)
public class ContainingPackageLookupFunctionTest extends FoundationTestCase {

  private AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;

  @Before
  public final void setUp() throws Exception  {
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)));
    deletedPackages = new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(pkgLocator, false);

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(SkyFunctions.PACKAGE_LOOKUP, new PackageLookupFunction(deletedPackages));
    skyFunctions.put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction());
    skyFunctions.put(SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES,
        new BlacklistedPackagePrefixesFunction());
    skyFunctions.put(SkyFunctions.FILE_STATE, new FileStateFunction(
        new AtomicReference<TimestampGranularityMonitor>(), externalFilesHelper));
    skyFunctions.put(SkyFunctions.FILE, new FileFunction(pkgLocator));
    RecordingDifferencer differencer = new RecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.set(differencer,
        PathFragment.EMPTY_FRAGMENT);
  }

  private ContainingPackageLookupValue lookupContainingPackage(String packageName)
      throws InterruptedException {
    SkyKey key =
        ContainingPackageLookupValue.key(PackageIdentifier.createInMainRepo(packageName));
    return driver
        .<ContainingPackageLookupValue>evaluate(
            ImmutableList.of(key),
            false,
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            NullEventHandler.INSTANCE)
        .get(key);
  }

  @Test
  public void testNoContainingPackage() throws Exception {
    ContainingPackageLookupValue value = lookupContainingPackage("a/b");
    assertFalse(value.hasContainingPackage());
  }

  @Test
  public void testContainingPackageIsParent() throws Exception {
    scratch.file("a/BUILD");
    ContainingPackageLookupValue value = lookupContainingPackage("a/b");
    assertTrue(value.hasContainingPackage());
    assertEquals(PackageIdentifier.createInMainRepo("a"), value.getContainingPackageName());
    assertEquals(rootDirectory, value.getContainingPackageRoot());
  }

  @Test
  public void testContainingPackageIsSelf() throws Exception {
    scratch.file("a/b/BUILD");
    ContainingPackageLookupValue value = lookupContainingPackage("a/b");
    assertTrue(value.hasContainingPackage());
    assertEquals(PackageIdentifier.createInMainRepo("a/b"), value.getContainingPackageName());
    assertEquals(rootDirectory, value.getContainingPackageRoot());
  }

  @Test
  public void testEqualsAndHashCodeContract() throws Exception {
    ContainingPackageLookupValue valueA1 = ContainingPackageLookupValue.NONE;
    ContainingPackageLookupValue valueA2 = ContainingPackageLookupValue.NONE;
    ContainingPackageLookupValue valueB1 =
        ContainingPackageLookupValue.withContainingPackage(
            PackageIdentifier.createInMainRepo("b"), rootDirectory);
    ContainingPackageLookupValue valueB2 =
        ContainingPackageLookupValue.withContainingPackage(
            PackageIdentifier.createInMainRepo("b"), rootDirectory);
    PackageIdentifier cFrag = PackageIdentifier.createInMainRepo("c");
    ContainingPackageLookupValue valueC1 =
        ContainingPackageLookupValue.withContainingPackage(cFrag, rootDirectory);
    ContainingPackageLookupValue valueC2 =
        ContainingPackageLookupValue.withContainingPackage(cFrag, rootDirectory);
    ContainingPackageLookupValue valueCOther =
        ContainingPackageLookupValue.withContainingPackage(
            cFrag, rootDirectory.getRelative("other_root"));
    new EqualsTester()
        .addEqualityGroup(valueA1, valueA2)
        .addEqualityGroup(valueB1, valueB2)
        .addEqualityGroup(valueC1, valueC2)
        .addEqualityGroup(valueCOther)
        .testEquals();
  }
}
