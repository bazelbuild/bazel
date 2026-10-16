// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for options classes for which an implementation should be generated.
 *
 * <p>Classes annotated with this should conform to the following constraints:
 *
 * <ul>
 *   <li>Methods annotated with <code>@Option</code> should be public, their name should start with
 *       <code>get</code> and the character after should be in upper case.
 * </ul>
 *
 * The generated options class will have the <code>Impl</code> suffix added to its name and will
 * contain a field, a getter and a setter method for each method annotated with <code>@Option</code>
 * .
 */
@SkybridgeInterface
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface OptionsClass {}
