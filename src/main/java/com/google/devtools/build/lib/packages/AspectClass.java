// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

/**
 *  A class of aspects.
 *  <p>Aspects might be defined natively, in Java ({@link NativeAspectClass})
 *  or in Skylark ({@link SkylarkAspectClass}).
 *
 *  Bazel propagates aspects through a multistage process. The general pipeline is as follows:
 *
 *  <pre>
 *  {@link AspectClass}
 *   |
 *   V
 *  {@code AspectDescriptor} <- {@link AspectParameters}
 *   \
 *   V
 *  {@link Aspect} <- {@link AspectDefinition} (might require loading Skylark files)
 *   |
 *   V
 *  {@code ConfiguredAspect}  <- {@code ConfiguredTarget}
 *  </pre>
 *
 *  <ul>
 *    <li>{@link AspectClass} is a moniker for "user" definition of the aspect, be it
 *    a native aspect or a Skylark aspect.  It contains either a reference to
 *    the native class implementing the aspect or the location of the Skylark definition
 *    of the aspect in the source tree, i.e. label of .bzl file + symbol name.
 *    </li>
 *    <li>{@link AspectParameters} is a (key,value) pair list that can be used to
 *    parameterize aspect classes</li>
 *    <li>{@link com.google.devtools.build.lib.analysis.AspectDescriptor} is a pair
 *    of {@code AspectClass} and {@link AspectParameters}. It uniquely identifies
 *    the aspect and can be used in SkyKeys.
 *    </li>
 *    <li>{@link AspectDefinition} is a class encapsulating the aspect definition (what
 *    attributes aspoect has, and along which dependencies does it propagate.
 *    </li>
 *    <li>{@link Aspect} is a fully instantiated instance of an Aspect after it is loaded.
 *    Getting an {@code Aspect} from {@code AspectDescriptor} for Skylark aspects
 *    requires adding a Skyframe dependency.
 *    </li>
 *    <li>{@link com.google.devtools.build.lib.analysis.ConfiguredAspect} represents a result
 *    of application of an {@link Aspect} to a given
 *    {@link com.google.devtools.build.lib.analysis.ConfiguredTarget}.
 *    </li>
 *  </ul>
 *
 *  {@link com.google.devtools.build.lib.analysis.AspectDescriptor}, or in general, a tuple
 *  of ({@link AspectClass}, {@link AspectParameters}) is an identifier that should be
 *  used in SkyKeys or in other contexts that need equality for aspects.
 *  See also {@link com.google.devtools.build.lib.skyframe.AspectFunction} for details
 *  on Skyframe treatment of Aspects.
 */
public interface AspectClass {

  /**
   * Returns an aspect name.
   */
  String getName();
}
