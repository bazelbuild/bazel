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
package com.google.devtools.build.lib.packages;

import static org.junit.Assert.assertFalse;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.util.PackageFactoryApparatus;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.concurrent.SynchronousQueue;

/**
 * Checks against a class initialization deadlock. "query sometimes hangs".
 *
 * <p>This requires static initialization of PackageGroup and PackageSpecification
 * to occur in a multithreaded context, and therefore must be in its own class.
 */
@RunWith(JUnit4.class)
public class PackageGroupStaticInitializationTest {
  private Scratch scratch = new Scratch("/workspace");
  private EventCollectionApparatus events = new EventCollectionApparatus();
  private PackageFactoryApparatus packages = new PackageFactoryApparatus(events.reporter());

  @Test
  public void testNoDeadlockOnPackageGroupCreation() throws Exception {
    scratch.file("fruits/BUILD", "package_group(name = 'mango', packages = ['//...'])");

    final SynchronousQueue<PackageSpecification> groupQueue = new SynchronousQueue<>();
    Thread producingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                try {
                  groupQueue.put(PackageSpecification.fromString(
                      Label.parseAbsoluteUnchecked("//context"), "//fruits/..."));
                } catch (Exception e) {
                  // Can't throw from Runnable, but this will cause the test to timeout
                  // when the consumer can't take the object.
                  e.printStackTrace();
                }
              }
            });

    Thread consumingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                try {
                  getPackageGroup("fruits", "mango");
                  groupQueue.take();
                } catch (Exception e) {
                  // Can't throw from Runnable, but this will cause the test to timeout
                  // when the producer can't put the object.
                  e.printStackTrace();
                }
              }
            });

    consumingThread.start();
    producingThread.start();
    producingThread.join(3000);
    consumingThread.join(3000);
    assertFalse(producingThread.isAlive());
    assertFalse(consumingThread.isAlive());
  }

  private Package getPackage(String packageName) throws Exception {
    PathFragment buildFileFragment = new PathFragment(packageName).getRelative("BUILD");
    Path buildFile = scratch.resolve(buildFileFragment.getPathString());
    return packages.createPackage(packageName, buildFile);
  }

  private PackageGroup getPackageGroup(String pkg, String name) throws Exception {
    return (PackageGroup) getPackage(pkg).getTarget(name);
  }
}
