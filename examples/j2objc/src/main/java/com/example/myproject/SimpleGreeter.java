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

package com.example.myproject;

import com.google.j2objc.annotations.ObjectiveCName;

/**
 * A simple Java class that uses a few features of J2ObjC.
 */
public class SimpleGreeter {
  private final Object obj;

  public SimpleGreeter(Object obj) {
    this.obj = obj;
  }

  /**
   * A simple method that says Hello to the object you pass in.
   */
  public String hello() {
    return "Hello, " + obj + "!";
  }

  /**
   * A method renamed with ObjectiveCName.
   */
  @ObjectiveCName("greetings")
  public String hello2() {
    return "Greetings, " + obj + "!";
  }

  /**
   * Returns a String description of this SimpleGreeter. Note that this method is renamed
   * by the J2ObjC transpilation.
   */
  public String toString() {
    return "This is a SimpleGreeter for " + obj;
  }
}
