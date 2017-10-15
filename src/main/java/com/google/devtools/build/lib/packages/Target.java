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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

import java.util.Set;
import javax.annotation.Nullable;

/**
 *  A node in the build dependency graph, identified by a Label.
 *
 * This SkylarkModule does not contain any documentation since Skylark's Target type refers to
 * TransitiveInfoCollection.class, which contains the appropriate documentation.
 */
@SkylarkModule(name = "target", doc = "", documented = false)
public interface Target {

  /**
   *  Returns the label of this target.  (e.g. "//foo:bar")
   */
  @SkylarkCallable(name = "label", doc = "")
  Label getLabel();

  /**
   *  Returns the name of this rule (relative to its owning package).
   */
  @SkylarkCallable(name = "name", doc = "")
  String getName();

  /**
   *  Returns the Package to which this rule belongs.
   */
  Package getPackage();

  /**
   * Returns a string describing this kind of target: e.g. "cc_library rule",
   * "source file", "generated file".
   */
  String getTargetKind();

  /**
   * Returns the rule associated with this target, if any.
   *
   * If this is a Rule, returns itself; it this is an OutputFile, returns its
   * generating rule; if this is an input file, returns null.
   */
  @Nullable
  Rule getAssociatedRule();

  /**
   * Returns the license associated with this target.
   */
  License getLicense();

  /**
   * Returns the place where the target was defined.
   */
  Location getLocation();

  /**
   * Returns the set of distribution types associated with this target.
   */
  Set<DistributionType> getDistributions();

  /**
   * Returns the visibility of this target.
   */
  RuleVisibility getVisibility();

  /**
   * Returns whether this target type can be configured (e.g. accepts non-null configurations).
   */
  boolean isConfigurable();
}
