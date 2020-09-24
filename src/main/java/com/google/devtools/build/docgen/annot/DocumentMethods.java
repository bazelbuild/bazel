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
package com.google.devtools.build.docgen.annot;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * A annotation applied to a class that indicates to docgen that the class's {@link
 * net.starlark.java.annot.StarlarkMethod}-annotated methods should be included in docgen's output
 * as standalone functions.
 *
 * <p>It is not necessary to apply this annotation to a class already annotated with {@link
 * net.starlark.java.annot.StarlarkBuiltin}; docgen will automatically document such classes as
 * built-in data types with Starlark methods defined by the annotated Java methods.
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface DocumentMethods {}
