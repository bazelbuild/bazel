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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Represents aspects that should be applied to a target as part of {@link Dependency}.
 *
 * <p>{@link Dependency} encapsulates all information that is needed to analyze an edge between an
 * AspectValue or a ConfiguredTargetValue and their direct dependencies, and {@link
 * AspectCollection} represents an aspect-related part of this information.
 *
 * <p>Analysis arrives to a particular node in target graph with an ordered list of aspects that
 * need to be applied. Some of those aspects should visible to the node in question; some of them
 * are not directly visible, but are visible to other aspects, as specified by {@link
 * com.google.devtools.build.lib.packages.AspectDefinition#getRequiredProvidersForAspects()}.
 *
 * <p>As an example, of all these things in interplay, consider android_binary rule depending on
 * java_proto_library rule depending on proto_library rule; consider further that we analyze the
 * android_binary with some ide_info aspect:
 *
 * <pre>
 *    proto_library(name = "pl") + ide_info_aspect
 *       ^
 *       | [java_proto_aspect]
 *    java_proto_library(name = "jpl") + ide_info_aspect
 *       ^
 *       | [DexArchiveAspect]
 *    android_binary(name = "ab") + ide_info_aspect
 * </pre>
 *
 * ide_info_aspect is interested in java_proto_aspect, but not in DexArchiveAspect.
 *
 * <p>Let's look is the {@link AspectCollection} for a Dependency representing a jpl->pl edge for
 * ide_info_aspect application to target <code>jpl</code>:
 *
 * <ul>
 *   <li>the full list of aspects is [java_proto_aspect, DexArchiveAspect, ide_info_aspect] in this
 *       order (the order is determined by the order in which aspects originate on {@code
 *       ab->...->pl} path).
 *   <li>however, DexArchiveAspect is not visible to either ide_info_aspect or java_proto_aspect, so
 *       the reduced list(and a result of {@link #getAllAspects()}) will be [java_proto_aspect,
 *       ide_info_aspect]
 *   <li>both java_proto_aspect and ide_info_aspect will be visible to <code>jpl + ide_info_aspect
 *       </code> node: the former because java_proto_library originates java_proto_aspect, and the
 *       aspect applied to the node sees the same dependencies; and the latter because the aspect
 *       sees itself on all targets it propagates to. So {@link #getVisibleAspects()} will return
 *       both of them.
 *   <li>Since ide_info_aspect declared its interest in java_proto_aspect and the latter comes
 *       before it in the order, {@link AspectDeps} for ide_info_aspect will contain
 *       java_proto_aspect (so the application of ide_info_aspect to <code>pl</code> target will see
 *       java_proto_aspect as well).
 * </ul>
 *
 * More details on members of {@link AspectCollection} follow, as well as more examples of aspect
 * visibility rules.
 *
 * <p>{@link AspectDeps} is a class that represents an aspect and all aspects that are directly
 * visible to it.
 *
 * <p>{@link #getVisibleAspects()} returns aspects that should be visible to the node in question.
 *
 * <p>{@link #getAllAspects()} return all aspects that should be applied to the target, in
 * topological order.
 *
 * <p>In the following scenario, consider rule r<sub>i</sub> sending an aspect a<sub>i</sub> to its
 * dependency:
 *
 * <pre>
 *      [r0]
 *       ^
 *  (a1) |
 *      [r1]
 *  (a2) |
 *      [r2]
 *  (a3) |
 *      [r3]
 * </pre>
 *
 * When a3 is propagated to target r0, the analysis arrives there with a path [a1, a2, a3]. Since we
 * analyse the propagation of aspect a3, the only visible aspect is a3.
 *
 * <p>Let's first assume that aspect a3 wants to see aspects a1 and a2, but aspects a1 and a2 are
 * not interested in each other (according to their {@link
 * com.google.devtools.build.lib.packages.AspectDefinition#getRequiredProvidersForAspects()}).
 *
 * <p>Since a3 is interested in all aspects, the result of {@link #getAllAspects()} will be [a1, a2,
 * a3], and {@link AspectCollection} will be:
 *
 * <ul>
 *   <li>a3 -> [a1, a2], a3 is visible
 *   <li>a2 -> []
 *   <li>a1 -> []
 * </ul>
 *
 * <p>Now what happens if a3 is interested in a2 but not a1, and a2 is interested in a1? Again, all
 * aspects are transitively interesting to a visible a3, so {@link #getAllAspects()} will be [a1,
 * a2, a3], but {@link AspectCollection} will now be:
 *
 * <ul>
 *   <li>a3 -> [a2], a3 is visible
 *   <li>a2 -> [a1]
 *   <li>a1 -> []
 * </ul>
 *
 * <p>As a final example, what happens if a3 is interested in a1, and a1 is interested in a2, but a3
 * is not interested in a2? Now the result of {@link #getAllAspects()} will be [a1, a3]. a1 is
 * interested in a2, but a2 comes later in the path than a1, so a1 does not see it (a1 only started
 * propagating on r1 -> r0 edge, and there is now a2 originating on that path). And {@link
 * AspectCollection} will now be:
 *
 * <ul>
 *   <li>a3 -> [a1], a3 is visible
 *   <li>a1 -> []
 * </ul>
 *
 * Note that is does not matter if a2 is interested in a1 or not - since no one after it in the path
 * is interested in it, a2 is filtered out.
 */
@Immutable
public final class AspectCollection {
  /** all aspects in the path; transitively visible to {@link #visibleAspects} */
  private final ImmutableSet<AspectDescriptor> aspectPath;

  /** aspects that should be visible to a dependency */
  private final ImmutableSet<AspectDeps> visibleAspects;

  public static final AspectCollection EMPTY = new AspectCollection(
      ImmutableSet.<AspectDescriptor>of(), ImmutableSet.<AspectDeps>of());


  private AspectCollection(
      ImmutableSet<AspectDescriptor> allAspects,
      ImmutableSet<AspectDeps> visibleAspects) {
    this.aspectPath = allAspects;
    this.visibleAspects = visibleAspects;
  }

  public Iterable<AspectDescriptor> getAllAspects() {
    return aspectPath;
  }

  public ImmutableSet<AspectDeps> getVisibleAspects() {
    return visibleAspects;
  }

  public boolean isEmpty() {
    return aspectPath.isEmpty();
  }

  @Override
  public int hashCode() {
    return aspectPath.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof AspectCollection)) {
      return false;
    }
    AspectCollection that = (AspectCollection) obj;
    return Objects.equal(aspectPath, that.aspectPath);
  }


  /**
   * Represents an aspect with all the aspects it depends on
   * (within an {@link AspectCollection}.
   *
   * We preserve the order of aspects to correspond to the order in the
   * original {@link AspectCollection#aspectPath}, although that is not
   * strictly needed semantically.
   */
  @Immutable
  public static final class AspectDeps {
    private final AspectDescriptor aspect;
    private final ImmutableList<AspectDeps> dependentAspects;

    private AspectDeps(AspectDescriptor aspect,
        ImmutableList<AspectDeps> dependentAspects) {
      this.aspect = aspect;
      this.dependentAspects = dependentAspects;
    }

    public AspectDescriptor getAspect() {
      return aspect;
    }

    public ImmutableList<AspectDeps> getDependentAspects() {
      return dependentAspects;
    }
  }

  public static AspectCollection createForTests(AspectDescriptor... descriptors) {
    return createForTests(ImmutableSet.copyOf(descriptors));
  }

  public static AspectCollection createForTests(ImmutableSet<AspectDescriptor> descriptors) {
    ImmutableSet.Builder<AspectDeps> depsBuilder = ImmutableSet.builder();
    for (AspectDescriptor descriptor : descriptors) {
      depsBuilder.add(new AspectDeps(descriptor, ImmutableList.<AspectDeps>of()));
    }
    return new AspectCollection(descriptors, depsBuilder.build());
  }

  /**
   *  Creates an {@link AspectCollection} from an ordered list of aspects and
   *  a set of visible aspects.
   *
   *  The order of aspects is reverse to the order in which they originated, with
   *  the earliest originating occurring last in the list.
   */
  public static AspectCollection create(
      Iterable<Aspect> aspectPath,
      Set<AspectDescriptor> visibleAspects) throws AspectCycleOnPathException {
    LinkedHashMap<AspectDescriptor, Aspect> aspectMap = deduplicateAspects(aspectPath);

    LinkedHashMap<AspectDescriptor, ArrayList<AspectDescriptor>> deps =
        new LinkedHashMap<>();

    // Calculate all needed aspects (either visible from outside or visible to
    // other needed aspects). Already discovered needed aspects are in key set of deps.
    // 1) Start from the end of the path. The aspect only sees other aspects that are
    //    before it
    // 2) If the 'aspect' is visible from outside, it is needed.
    // 3) Otherwise, check whether 'aspect' is visible to any already needed aspects,
    //    if it is visible to a needed 'depAspect',
    //            add the 'aspect' to a list of aspects visible to 'depAspect'.
    //    if 'aspect' is needed, add it to 'deps'.
    // At the end of this algorithm, key set of 'deps' contains a subset of original
    // aspect list consisting only of needed aspects, in reverse (since we iterate
    // the original list in reverse).
    //
    // deps[aspect] contains all aspects that 'aspect' needs, in reverse order.
    for (Map.Entry<AspectDescriptor, Aspect> aspect :
        ImmutableList.copyOf(aspectMap.entrySet()).reverse()) {
      boolean needed = visibleAspects.contains(aspect.getKey());
      for (AspectDescriptor depAspectDescriptor : deps.keySet()) {
        if (depAspectDescriptor.equals(aspect.getKey())) {
          continue;
        }
        Aspect depAspect = aspectMap.get(depAspectDescriptor);
        if (depAspect.getDefinition().getRequiredProvidersForAspects()
            .isSatisfiedBy(aspect.getValue().getDefinition().getAdvertisedProviders())) {
          deps.get(depAspectDescriptor).add(aspect.getKey());
          needed = true;
        }
      }
      if (needed && !deps.containsKey(aspect.getKey())) {
        deps.put(aspect.getKey(), new ArrayList<AspectDescriptor>());
      }
    }

    // Record only the needed aspects from all aspects, in correct order.
    ImmutableList<AspectDescriptor> neededAspects = ImmutableList.copyOf(deps.keySet()).reverse();

    // Calculate visible aspect paths.
    HashMap<AspectDescriptor, AspectDeps> aspectPaths = new HashMap<>();
    ImmutableSet.Builder<AspectDeps> visibleAspectPaths = ImmutableSet.builder();
    for (AspectDescriptor visibleAspect : visibleAspects) {
      visibleAspectPaths.add(buildAspectDeps(visibleAspect, aspectPaths, deps));
    }
    return new AspectCollection(ImmutableSet.copyOf(neededAspects),
        visibleAspectPaths.build());
  }

  /**
   * Deduplicate aspects in path.
   *
   * @throws AspectCycleOnPathException if an aspect occurs twice on the path and
   *         the second occurrence sees a different set of aspects.
   */
  private static LinkedHashMap<AspectDescriptor, Aspect> deduplicateAspects(
      Iterable<Aspect> aspectPath) throws AspectCycleOnPathException {

    LinkedHashMap<AspectDescriptor, Aspect> aspectMap = new LinkedHashMap<>();
    ArrayList<Aspect> seenAspects = new ArrayList<>();
    for (Aspect aspect : aspectPath) {
      if (!aspectMap.containsKey(aspect.getDescriptor())) {
        aspectMap.put(aspect.getDescriptor(), aspect);
        seenAspects.add(aspect);
      } else {
        validateDuplicateAspect(aspect, seenAspects);
      }
    }
    return aspectMap;
  }

  /**
   * Detect inconsistent duplicate occurrence of an aspect on the path.
   * There is a previous occurrence of {@code aspect} in {@code seenAspects}.
   *
   * If in between that previous occurrence and the newly discovered occurrence
   * there is an aspect that is visible to {@code aspect}, then the second occurrence
   * is inconsistent - the set of aspects it sees is different from the first one.
   */
  private static void validateDuplicateAspect(Aspect aspect, ArrayList<Aspect> seenAspects)
      throws AspectCycleOnPathException {
    for (int i = seenAspects.size() - 1; i >= 0; i--) {
      Aspect seenAspect = seenAspects.get(i);
      if (aspect.getDescriptor().equals(seenAspect.getDescriptor())) {
        // This is a previous occurrence of the same aspect.
        return;
      }

      if (aspect.getDefinition().getRequiredProvidersForAspects()
          .isSatisfiedBy(seenAspect.getDefinition().getAdvertisedProviders())) {
        throw new AspectCycleOnPathException(aspect.getDescriptor(), seenAspect.getDescriptor());
      }
    }
  }

  private static AspectDeps buildAspectDeps(AspectDescriptor descriptor,
      HashMap<AspectDescriptor, AspectDeps> aspectPaths,
      LinkedHashMap<AspectDescriptor, ArrayList<AspectDescriptor>> deps) {
    if (aspectPaths.containsKey(descriptor)) {
      return aspectPaths.get(descriptor);
    }

    ImmutableList.Builder<AspectDeps> aspectPathBuilder = ImmutableList.builder();
    ArrayList<AspectDescriptor> depList = deps.get(descriptor);

    // deps[aspect] contains all aspects visible to 'aspect' in reverse order.
    for (int i = depList.size() - 1; i >= 0; i--) {
      aspectPathBuilder.add(buildAspectDeps(depList.get(i), aspectPaths, deps));
    }
    AspectDeps aspectPath = new AspectDeps(descriptor, aspectPathBuilder.build());
    aspectPaths.put(descriptor, aspectPath);
    return aspectPath;
  }

  /**
   * Signals an inconsistency on aspect path: an aspect occurs twice on the path and the second
   * occurrence sees a different set of aspects.
   *
   * <p>{@link #getAspect()} is the aspect occurring twice, and {@link #getPreviousAspect()} is the
   * aspect that the second occurrence sees but the first does not.
   */
  public static class AspectCycleOnPathException extends Exception {

    private final AspectDescriptor aspect;
    private final AspectDescriptor previousAspect;

    public AspectCycleOnPathException(AspectDescriptor aspect, AspectDescriptor previousAspect) {
      super(String.format("Aspect %s is applied twice, both before and after aspect %s",
          aspect.getDescription(), previousAspect.getDescription()
      ));
      this.aspect = aspect;
      this.previousAspect = previousAspect;
    }

    public AspectDescriptor getAspect() {
      return aspect;
    }

    public AspectDescriptor getPreviousAspect() {
      return previousAspect;
    }
  }
}
