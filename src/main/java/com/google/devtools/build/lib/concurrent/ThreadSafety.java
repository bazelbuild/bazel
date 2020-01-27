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
package com.google.devtools.build.lib.concurrent;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Define some standard attributes for documenting thread safety properties.
 *<p>
 * The names used here are adapted from those used in Joshua Bloch's book
 * "Effective Java", which are also described at
 * <http://www.ibm.com/developerworks/java/library/j-jtp09263/>.
 *<p>
 * These attributes are just documentation.  They don't have any run-time
 * effect.  The main aim is mainly just to standardize the terminology.
 * (However, if this catches on, I can also imagine in the future having
 * a presubmit check that checks that all new classes have thread safety
 * annotations :)
 *<p>
 * See ThreadSafetyTest for examples of how these attributes should be used.
 */
public class ThreadSafety {
  /**
   * The Immutable attribute indicates that instances of the class are
   * immutable, or at least appear that way are far as their external API
   * is concerned.  Immutable classes are usually also ThreadSafe,
   * but can be ThreadHostile if they perform unsynchronized access to
   * mutable static data.  (We deviate from Bloch's nomenclature by
   * not assuming that Immutable implies ThreadSafe; developers should
   * explicitly annotate classes as both Immutable and ThreadSafe when
   * appropriate.)
   */
  @Documented
  @Target(value = {ElementType.TYPE})
  @Retention(RetentionPolicy.RUNTIME)
  public @interface Immutable {}

  /**
   * The ThreadSafe attribute marks a class or method which can safely be used
   * from multiple threads without any need for external synchronization.
   *
   * When applied to a class, this attribute indicates that instances
   * of the class can safely be used concurrently from multiple threads
   * without any need for external synchronization, i.e. that all non-static methods
   * are thread-safe (except any private methods that are explicitly
   * annotated with a different thread safety annotation).  In addition it
   * also indicates that all non-static nested classes are thread-safe (except any private
   * nested classes that are explicitly annotated with a different thread
   * safety annotation). Note that no guarantees are made about static class methods or static
   * nested classes - they should be annotated separately.
   *
   * When applied to a method, this attribute indicates that the
   * method can safely be called concurrently from multiple threads.
   * The implementation must synchronize any accesses to mutable data.
   */
  @Documented
  @Target(value = {ElementType.CONSTRUCTOR, ElementType.METHOD, ElementType.TYPE})
  @Retention(RetentionPolicy.SOURCE)
  public @interface ThreadSafe {}

  /**
   * The ThreadCompatible attribute marks a class or method that
   * is thread-safe provided that only one thread attempts to
   * access each object at a time.
   *
   * The implementation of such a class must synchronize accesses
   * to mutable static data, but can assume that each instance will
   * only be accessed from one thread at a time.
   *
   * The client must obtain an appropriate lock before calling ThreadCompatible
   * methods, or must otherwise ensure that only one thread calls such methods.
   * Unless otherwise specified, an appropriate lock means synchronizing on the
   * instance.
   *
   * A ThreadCompatible class may contain private methods or private nested
   * classes that are not ThreadCompatible provided that they are explicitly
   * annotated with a different thread safety annotation.
   */
  @Documented
  @Target(value = {ElementType.METHOD, ElementType.TYPE})
  @Retention(RetentionPolicy.SOURCE)
  public @interface ThreadCompatible {}

  /**
   * The ThreadHostile attribute marks a class or method that
   * can't safely be used by multiple threads, for example because
   * it performs unsynchronized access to mutable static objects.
   */
  @Documented
  @Target(value = {ElementType.CONSTRUCTOR, ElementType.METHOD, ElementType.TYPE})
  @Retention(RetentionPolicy.SOURCE)
  public @interface ThreadHostile {}

  /**
   * The ConditionallyThreadSafe attribute marks a class that contains
   * some methods (or nested classes) which are ThreadSafe but others which are
   * only ThreadCompatible or ThreadHostile.
   *
   * The methods (and nested classes) of a ConditionallyThreadSafe class should
   * each have their thread safety marked.
   */
  @Documented
  @Target(value = {ElementType.METHOD, ElementType.TYPE})
  @Retention(RetentionPolicy.SOURCE)
  public @interface ConditionallyThreadSafe {}

  /**
   * The ConditionallyThreadCompatible attribute marks a class that contains
   * some methods (or nested classes) which are ThreadCompatible but others
   * which are ThreadHostile.
   *
   * The methods (and nested classes) of a ConditionallyThreadCompatible class
   * should each have their thread safety marked.
   */
  @Documented
  @Target(value = {ElementType.METHOD, ElementType.TYPE})
  @Retention(RetentionPolicy.SOURCE)
  public @interface ConditionallyThreadCompatible {}

}
