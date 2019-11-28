// Copyright 2019 The Bazel Authors. All rights reserved.
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
 * A StarlarkIterable value may be iterated by Starlark language constructs such as {@code for}
 * loops, list and dict comprehensions, and {@code f(*args)}.
 *
 * <p>Functionally this interface is equivalent to {@code java.lang.Iterable}, but it additionally
 * affirms that the iterability of a Java class should be exposed to Starlark programs.
 */
public interface StarlarkIterable<T> extends StarlarkValue, Iterable<T> {}
