// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Interface for a {@link Fragment} object to declare which {@link FragmentOptions} it needs for
 * construction.
 *
 * <p>Blaze instantiates {@link Fragment} with a {@link BuildOptions} that only contains the {@link
 * FragmentOptions} specified here.
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface RequiresOptions {
  /** The options required by the annotated fragment. By default, fragments require no options. */
  Class<? extends FragmentOptions>[] options() default {};

  /** Whether the annotated fragment requires access to starlark options. */
  boolean starlark() default false;
}
