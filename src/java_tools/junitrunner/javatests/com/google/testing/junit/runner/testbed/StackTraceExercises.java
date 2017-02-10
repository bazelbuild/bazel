// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.testbed;

import junit.framework.TestCase;

/**
 * This is a testbed for testing stack trace functionality.
 * Failures in this test should not cause continuous builds to go red.
 */
public class StackTraceExercises extends TestCase {

  /**
   * Succeeds fast but leaves behind a devious shutdown hook designed to wreak havoc.
   */
  public void testSneakyShutdownHook() throws Exception {
    Runtime.getRuntime().addShutdownHook(new Thread() {
      public void run() {
        handleHook();
      }});
   }

   private static void handleHook() {
     try {
       System.out.println("Entered shutdown hook");
       System.out.flush();
       Fifo.waitUntilDataAvailable();
       Thread.sleep(15000);
     } catch (Exception e) {
       throw new Error(e);
      }
  }

  /**
   * A test which invokes System.exit(0). Bad test!
   */
  public void testNotSoFastBuddy() {
    System.out.println("Hey, not so fast there");
    System.exit(0);
  }
}
