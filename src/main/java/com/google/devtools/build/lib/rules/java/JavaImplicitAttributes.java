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

package com.google.devtools.build.lib.rules.java;

/** Implicit attribute names that Java rules use, such as the JDK target name. */
public class JavaImplicitAttributes {
  /** Label of the default target JDK. */
  public static final String JDK_LABEL = "//tools/jdk:jdk";

  /** Label of the default host JDK. */
  public static final String HOST_JDK_LABEL = "//tools/jdk:host_jdk";
}
