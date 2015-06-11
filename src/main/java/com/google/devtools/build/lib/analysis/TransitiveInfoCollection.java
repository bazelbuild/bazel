// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkModule;

import javax.annotation.Nullable;

/**
 * Objects that implement this interface bundle multiple {@link TransitiveInfoProvider} interfaces.
 *
 * <p>This interface (together with {@link TransitiveInfoProvider} is the cornerstone of the data
 * model of the analysis phase.
 *
 * <p>The computation a configured target does is allowed to depend on the following things:
 * <ul>
 * <li>The associated Target (which will usually be a Rule)
 * <li>Its own configuration (the configured target does not have access to other configurations,
 * e.g. the host configuration)
 * <li>The transitive info providers and labels of its direct dependencies.
 * </ul>
 *
 * <p>And these are the only inputs. Notably, a configured target is not supposed to access
 * other configured targets, the transitive info collections of configured targets it does not
 * directly depend on, the actions created by anyone else or the contents of any input file. We
 * strive to make it impossible for configured targets to do these things.
 *
 * <p>A configured target is expected to produce the following data during its analysis:
 * <ul>
 * <li>A number of Artifacts and Actions generating them
 * <li>A set of {@link TransitiveInfoProvider}s that it passes on to the targets directly dependent
 * on it
 * </ul>
 *
 * <p>The information that can be passed on to dependent targets by way of
 * {@link TransitiveInfoProvider} is subject to constraints (which are detailed in the
 * documentation of that class).
 *
 * <p>Configured targets are currently allowed to create artifacts at any exec path. It would be
 * better if they could be constrained to a subtree based on the label of the configured target,
 * but this is currently not feasible because multiple rules violate this constraint and the
 * output format is part of its interface.
 *
 * <p>In principle, multiple configured targets should not create actions with conflicting
 * outputs. There are still a few exceptions to this rule that are slated to be eventually
 * removed, we have provisions to handle this case (Action instances that share at least one
 * output file are required to be exactly the same), but this does put some pressure on the design
 * and we are eventually planning to eliminate this option.
 *
 * <p>These restrictions together make it possible to:
 * <ul>
 * <li>Correctly cache the analysis phase; by tightly constraining what a configured target is
 * allowed to access and what it is not, we can know when it needs to invalidate a particular
 * one and when it can reuse an already existing one.
 * <li>Serialize / deserialize individual configured targets at will, making it possible for
 * example to swap out part of the analysis state if there is memory pressure or to move them in
 * persistent storage so that the state can be reconstructed at a different time or in a
 * different process. The stretch goal is to eventually facilitate cross-user caching of this
 * information.
 * </ul>
 *
 * <p>Implementations of build rules should <b>not</b> hold on to references to the
 * {@link TransitiveInfoCollection}s representing their direct prerequisites in order to reduce
 * their memory footprint (otherwise, the referenced object could refer one of its direct
 * dependencies in turn, thereby making the size of the objects reachable from a single instance
 * unbounded).
 *
 * @see TransitiveInfoProvider
 */
@SkylarkModule(name = "Target", doc =
      "A BUILD target. It is essentially a <code>struct</code> with the following fields:"
    + "<ul>"
    + "<li><h3 id=\"modules.Target.label\">label</h3><code><a class=\"anchor\" "
    + "href=\"#modules.Label\">Label</a> Target.label</code><br>The identifier of the target.</li>"
    + "<li><h3 id=\"modules.Target.files\">files</h3><code><a class=\"anchor\" "
    + "href=\"#modules.set\">set</a> Target.files </code><br>The (transitive) set of <a "
    + "class=\"anchor\" href=\"#modules.File\">File</a>s produced by this target.</li>"
    + "<li><h3 id=\"modules.Target.extraproviders\">Extra providers</h3>For rule targets all "
    + "additional providers provided by this target are accessible as <code>struct</code> fields. "
    + "These extra providers are defined in the <code>struct</code> returned by the rule "
    + "implementation function.</li>"
    + "</ul>")
public interface TransitiveInfoCollection extends Iterable<TransitiveInfoProvider> {

  /**
   * Returns the transitive information provider requested, or null if the provider is not found.
   * The provider has to be a TransitiveInfoProvider Java class.
   */
  @Nullable <P extends TransitiveInfoProvider> P getProvider(Class<P> provider);

  /**
   * Returns the label associated with this prerequisite.
   */
  Label getLabel();

  /**
   * <p>Returns the {@link BuildConfiguration} for which this transitive info collection is defined.
   * Configuration is defined for all configured targets with exception of {@link
   * InputFileConfiguredTarget} and {@link PackageGroupConfiguredTarget} for which it is always
   * <b>null</b>.</p>
   */
  @Nullable BuildConfiguration getConfiguration();

  /**
   * Returns the transitive information requested or null, if the information is not found.
   * The transitive information has to have been added using the Skylark framework.
   */
  @Nullable Object get(String providerKey);

  /**
   * Returns an unmodifiable iterator over the transitive info providers in the collections.
   */
  @Override
  UnmodifiableIterator<TransitiveInfoProvider> iterator();
}
