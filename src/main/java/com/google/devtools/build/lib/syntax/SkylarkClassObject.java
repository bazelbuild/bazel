// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

/**
 * A marker interface for a {@link ClassObject} whose {@link #getValue} always returns a
 * Skylark-friendly value, with no defensive conversion required.
 *
 * <p>An value is Skylark-friendly if its class (or a supertype) implements
 * {@link com.google.devtools.build.lib.skylarkinterface.SkylarkValue},
 * is annotated with {@link com.google.devtools.build.lib.skylarkinterface.SkylarkModule}, or is a
 * Skylark primitive like {@link String}.
 *
 * <p>Ideally, this class should not be needed, and all {@link ClassObject}s should only expose
 * Skylark-friendly values.
 */
public interface SkylarkClassObject extends ClassObject {
}
