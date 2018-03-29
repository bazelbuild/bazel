// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link AndroidResourcesTest} */
@RunWith(JUnit4.class)
public class AndroidResourcesTest extends ResourceTestBase {
  private static final PathFragment DEFAULT_RESOURCE_ROOT = PathFragment.create(RESOURCE_ROOT);
  private static final ImmutableList<PathFragment> RESOURCES_ROOTS =
      ImmutableList.of(DEFAULT_RESOURCE_ROOT);

  @Before
  @Test
  public void testGetResourceRootsNoResources() throws Exception {
    assertThat(getResourceRoots()).isEmpty();
  }

  @Test
  public void testGetResourceRootsInvalidResourceDirectory() throws Exception {
    try {
      getResourceRoots("is-this-drawable-or-values/foo.xml");
      assertWithMessage("Expected exception not thrown!").fail();
    } catch (RuleErrorException e) {
      // expected
    }

    errorConsumer.assertAttributeError(
        "resource_files", "is not in the expected resource directory structure");
  }

  @Test
  public void testGetResourceRootsMultipleRoots() throws Exception {
    try {
      getResourceRoots("subdir/values/foo.xml", "otherdir/values/bar.xml");
      assertWithMessage("Expected exception not thrown!").fail();
    } catch (RuleErrorException e) {
      // expected
    }

    errorConsumer.assertAttributeError(
        "resource_files", "All resources must share a common directory");
  }

  @Test
  public void testGetResourceRoots() throws Exception {
    assertThat(getResourceRoots("values-hdpi/foo.xml", "values-mdpi/bar.xml"))
        .isEqualTo(RESOURCES_ROOTS);
  }

  @Test
  public void testGetResourceRootsCommonSubdirectory() throws Exception {
    assertThat(getResourceRoots("subdir/values-hdpi/foo.xml", "subdir/values-mdpi/bar.xml"))
        .containsExactly(DEFAULT_RESOURCE_ROOT.getRelative("subdir"));
  }

  private ImmutableList<PathFragment> getResourceRoots(String... pathResourceStrings)
      throws Exception {
    return getResourceRoots(getResources(pathResourceStrings));
  }

  private ImmutableList<PathFragment> getResourceRoots(ImmutableList<Artifact> artifacts)
      throws Exception {
    return AndroidResources.getResourceRoots(errorConsumer, artifacts, "resource_files");
  }

  @Test
  public void testFilterEmpty() throws Exception {
    assertFilter(ImmutableList.<Artifact>of(), ImmutableList.<Artifact>of());
  }

  @Test
  public void testFilterNoop() throws Exception {
    ImmutableList<Artifact> resources = getResources("values-en/foo.xml", "values-es/bar.xml");
    assertFilter(resources, resources);
  }

  @Test
  public void testFilterToEmpty() throws Exception {
    assertFilter(
        getResources("values-en/foo.xml", "values-es/bar.xml"), ImmutableList.<Artifact>of());
  }

  @Test
  public void testPartiallyFilter() throws Exception {
    Artifact keptResource = getResource("values-en/foo.xml");
    assertFilter(
        ImmutableList.of(keptResource, getResource("values-es/bar.xml")),
        ImmutableList.of(keptResource));
  }

  @Test
  public void testFilterIsDependency() throws Exception {
    Artifact keptResource = getResource("values-en/foo.xml");
    assertFilter(
        ImmutableList.of(keptResource, getResource("drawable/bar.png")),
        ImmutableList.of(keptResource),
        /* isDependency = */ true);
  }

  private void assertFilter(
      ImmutableList<Artifact> unfilteredResources, ImmutableList<Artifact> filteredResources)
      throws Exception {
    assertFilter(unfilteredResources, filteredResources, /* isDependency = */ false);
  }

  private void assertFilter(
      ImmutableList<Artifact> unfilteredResources,
      ImmutableList<Artifact> filteredResources,
      boolean isDependency)
      throws Exception {
    ImmutableList<PathFragment> unfilteredResourcesRoots = getResourceRoots(unfilteredResources);
    AndroidResources unfiltered =
        new AndroidResources(unfilteredResources, unfilteredResourcesRoots);

    ImmutableList.Builder<Artifact> filteredDepsBuilder = ImmutableList.builder();

    ResourceFilter fakeFilter =
        ResourceFilter.of(ImmutableSet.copyOf(filteredResources), filteredDepsBuilder::add);

    Optional<AndroidResources> filtered = unfiltered.maybeFilter(fakeFilter, isDependency);

    if (filteredResources.equals(unfilteredResources)) {
      // We expect filtering to have been a no-op
      assertThat(filtered.isPresent()).isFalse();
    } else {
      // The resources and their roots should be filtered
      assertThat(filtered.get().getResources())
          .containsExactlyElementsIn(filteredResources)
          .inOrder();
      assertThat(filtered.get().getResourceRoots())
          .containsExactlyElementsIn(getResourceRoots(filteredResources))
          .inOrder();
    }

    if (!isDependency) {
      // The filter should not record any filtered dependencies
      assertThat(filteredDepsBuilder.build()).isEmpty();
    } else {
      // The filtered dependencies should be exactly the list of filtered resources
      assertThat(unfilteredResources)
          .containsExactlyElementsIn(
              Iterables.concat(filteredDepsBuilder.build(), filteredResources));
    }
  }
}
