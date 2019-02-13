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

package com.google.devtools.build.lib.skylarkinterface;

/**
 * A container of application-specific context which may be passed to Starlark callable functions.
 *
 * <p>An application using the Starlark interpreter can provide a custom context implementation to a
 * Starlark environment using {@link Environment#setStarlarkContext}. The context object will then
 * be passed to any {@link SkylarkCallable} method with {@link SkylarkCallable#useContext} set to
 * true.
 *
 * <p>Note that the interpreter passes a StarlarkContext to a callable method which requests it, and
 * does no casting to an application-specific context object. The application's implementation of
 * the callable method must cast a context object to the application-specific context itself.
 */
public interface StarlarkContext {}
