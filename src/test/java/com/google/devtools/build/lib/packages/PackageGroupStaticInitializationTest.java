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
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import java.util.concurrent.SynchronousQueue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Checks against a class initialization deadlock. "query sometimes hangs".
 *
 * <p>This requires static initialization of PackageGroup and PackageSpecification to occur in a
 * multithreaded context, and therefore must be in its own class.
 */
@RunWith(JUnit4.class)
public class PackageGroupStaticInitializationTest extends PackageLoadingTestCase {

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
                  getTarget("//fruits:mango");
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
}
