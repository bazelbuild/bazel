/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.testing.junit;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Indicates that a specific test class or test method requires a minimum and/or maximum JDK version
 * to execute.
 *
 * <p>A test will be ignored (passed vacuously) if the actual JDK version under test is out of the
 * expected JDK version range. (inclusive). It is up to the implementer to specify the source of
 * truth of the JDK version under investigation, e.g. the Java runtime environment, source code
 * language level, class file major version, etc.
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.TYPE, ElementType.METHOD})
public @interface JdkSuppress {
  /** The minimum JDK version to execute (inclusive) */
  int minJdkVersion() default JdkVersion.V1_8;
  /** The maximum JDK version to execute (inclusive) */
  int maxJdkVersion() default Integer.MAX_VALUE;
}
