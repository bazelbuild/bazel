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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.util.PackageFactoryApparatus;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.concurrent.SynchronousQueue;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
  private Root root;

  @Before
  public void setUp() throws Exception {
    root = Root.fromPath(scratch.dir(""));
  }

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
                  RepositoryName defaultRepoName =
                      Label.parseAbsoluteUnchecked("//context")
                          .getPackageIdentifier()
                          .getRepository();
                  groupQueue.put(PackageSpecification.fromString(defaultRepoName, "//fruits/..."));
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
    assertThat(producingThread.isAlive()).isFalse();
    assertThat(consumingThread.isAlive()).isFalse();
  }

  private Package getPackage(String packageName) throws Exception {
    PathFragment buildFileFragment = PathFragment.create(packageName).getRelative("BUILD");
    Path buildFile = scratch.resolve(buildFileFragment.getPathString());
    return packages.createPackage(packageName, RootedPath.toRootedPath(root, buildFile));
  }

  private PackageGroup getPackageGroup(String pkg, String name) throws Exception {
    return (PackageGroup) getPackage(pkg).getTarget(name);
  }
}
