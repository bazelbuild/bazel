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

import java.util.List;

/**
 * A RuleVisibility specifies which other rules can depend on a specified rule.
 * Note that the actual method that performs this check is declared in
 * RuleConfiguredTargetVisibility.
 *
 * <p>The conversion to ConfiguredTargetVisibility is handled in an ugly
 * if-ladder, because I want to avoid this package depending on build.lib.view.
 *
 * All implementations of this interface are immutable.
 */
public interface RuleVisibility {
  /**
   * Returns the list of labels that need to be loaded so that the visibility
   * decision can be made during analysis time. E.g. for package group
   * visibility, this is the list of package groups referenced. Does not include
   * labels that have special meanings in the visibility declaration, e.g.
   * "//visibility:*" or "//*:__pkg__".
   */
  List<Label> getDependencyLabels();

  /**
   * Returns the list of labels used during the declaration of this visibility.
   * These do not necessarily represent loadable labels: for example, for public
   * or private visibilities, the special labels "//visibility:*" will be
   * returned, and so will be the special "//*:__pkg__" labels indicating a
   * single package.
   */
  List<Label> getDeclaredLabels();
}

