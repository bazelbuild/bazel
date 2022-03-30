package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * KnownTargetsDependencyResolver is a DependencyResolver which resolves statically over a known
 * set of targets. It can be useful when performing queries over a known pre-resolved universe of
 * targets.
 */
public class KnownTargetsDependencyResolver extends DependencyResolver {

  private final Map<Label, Target> knownTargets;

  public KnownTargetsDependencyResolver(Map<Label, Target> knownTargets) {
    this.knownTargets = knownTargets;
  }

  @Override
  protected Map<Label, Target> getTargets(
      OrderedSetMultimap<DependencyKind, Label> labelMap,
      TargetAndConfiguration fromNode,
      NestedSetBuilder<Cause> rootCauses) {
    return labelMap.values().stream()
        .distinct()
        .filter(Objects::nonNull)
        .filter(knownTargets::containsKey)
        .collect(Collectors.toMap(Function.identity(), knownTargets::get));
  }
}
