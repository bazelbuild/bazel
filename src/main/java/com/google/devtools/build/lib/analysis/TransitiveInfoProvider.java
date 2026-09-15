// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

/**
 * Contains rolled-up data about the transitive closure of a configured target.
 *
 * For more information about how analysis works, see
 * {@link com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory}.
 * TransitiveInfoProviders need to be serializable, and for that reason they must conform to
 * the following restrictions:
 *
 * <ul>
 * <li>The provider interface must directly extend {@code TransitiveInfoProvider}.
 * <li>Every method must return immutable data.</li>
 * <li>Every method must return the same object if called multiple times with the same
 * arguments.</li>
 * <li>Overloading a method name multiple times is forbidden.</li>
 * <li>The return type of a method must satisfy one of the following conditions:
 * <ul>
 *  <li>It must be from the set of {String, Integer, int, Boolean, bool, Label, PathFragment,
 * Artifact}, OR</li>
 *  <li>it must be an ImmutableList/List/Collection/Iterable of T, where T is either
 * one of the types above with a default serializer or T implements ValueSerializer, OR</li>
 *  <li>it must be serializable (TBD)</li>
 * </ul>
 * <li>If the method takes arguments, it must declare a custom serializer (TBD).</li>
 * </ul>
 *
 * <p>Some typical uses of this interface are:
 * <ul>
 * <li>The set of Python source files in the transitive closure of this rule
 * <li>The set of declared C++ header files in the transitive closure
 * <li>The files that need to be built when the target is mentioned on the command line
 * </ul>
 *
 * <p>Note that if implemented naively, this would result in the memory requirements
 * being O(n^2): in a long dependency chain, if every target adds one single artifact, storing the
 * transitive closures of every rule would take 1+2+3+...+n-1+n = O(n^2) memory.
 *
 * <p>In order to avoid this, we introduce the concept of nested sets, {@link
 * com.google.devtools.build.lib.collect.nestedset.NestedSet}. A nested set is an immutable
 * data structure that can contain direct members and other nested sets (recursively). Nested sets
 * are iterable and can be flattened into ordered sets, where the order depends on which
 * implementation of NestedSet you pick.
 *
 * @see TransitiveInfoCollection
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 */
public interface TransitiveInfoProvider {

}
