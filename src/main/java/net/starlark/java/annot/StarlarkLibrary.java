// Copyright 2026 The Bazel Authors. All rights reserved.
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

package net.starlark.java.annot;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * An annotation for classes (and interfaces) that contain Starlark top-level methods (annotated
 * with {@link StarlarkMethod}).
 *
 * <p>Typically, classes that get passed to {@link Starlark#addMethods} should have this annotation.
 * {@link MethodLibrary} is the canonical example.
 *
 * <p>Classes that represent Starlark data types (implementations of {@link StarlarkValue}) and
 * which contain {@code @StarlarkMethod}-annotated methods should instead be annotated with {@link
 * StarlarkBuiltin}. No class should be annotated with, or inherit from ancestors that are annotated
 * with, both {@code @StarlarkLibrary} and {@code @StarlarkBuiltin}. (However, due to limitations in
 * the annotation processor, we only enforce that a class annotated with one cannot inherit from an
 * ancestor annotated with the other.)
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface StarlarkLibrary {}
