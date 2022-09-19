// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.genquery;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.LabelVisitationUtils;
import com.google.devtools.build.lib.packages.LabelVisitationUtils.LabelProcessor;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.TargetLoadingUtil;
import com.google.devtools.build.lib.skyframe.TargetLoadingUtil.TargetAndErrorIfAny;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;

/**
 * Factory for {@link GenQueryPackageProvider} which directly relies on {@link
 * com.google.devtools.build.lib.skyframe.PackageValue} Skyframe data to collect required
 * information.
 */
public class GenQueryDirectPackageProviderFactory implements GenQueryPackageProviderFactory {

  public static final SkyFunctionName GENQUERY_SCOPE =
      SkyFunctionName.createHermetic("GENQUERY_SCOPE");

  /**
   * It can be common, due to macro expansion, that several genquery rules share the same value for
   * their scope attribute. By doing scope traversal as its own Skyframe node, a set of genquery
   * rules sharing the same scope will require only one scope traversal to occur.
   */
  @AutoCodec
  static class Key extends AbstractSkyKey<ImmutableList<Label>> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(ImmutableList<Label> arg) {
      super(ImmutableList.sortedCopyOf(arg));
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(ImmutableList<Label> arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return GENQUERY_SCOPE;
    }
  }

  private static class Value implements SkyValue {
    private final GenQueryPackageProvider genQueryPackageProvider;

    private Value(GenQueryPackageProvider genQueryPackageProvider) {
      this.genQueryPackageProvider = genQueryPackageProvider;
    }
  }

  private static class Function implements SkyFunction {
    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      @SuppressWarnings("unchecked")
      ImmutableList<Label> scope = (ImmutableList<Label>) skyKey.argument();

      GenQueryPackageProvider provider;
      try {
        provider = constructPackageMapImpl(env, scope);
      } catch (BrokenQueryScopeException e) {
        throw new BrokenQueryScopeSkyFunctionException(e, Transience.PERSISTENT);
      }
      if (provider == null) {
        return null;
      }
      return new Value(provider);
    }
  }

  private static class BrokenQueryScopeSkyFunctionException extends SkyFunctionException {
    protected BrokenQueryScopeSkyFunctionException(
        BrokenQueryScopeException cause, Transience transience) {
      super(cause, transience);
    }
  }

  public static final SkyFunction FUNCTION = new Function();

  /**
   * This factory's strategy relies on Skyframe "state" to prevent redundant work from being done
   * across Skyframe restarts.
   *
   * <p>The {@code collectedPackages} and {@code collectedTargets} fields are populated by {@link
   * #constructPackageMap}'s target dependency traversal, until {@code collectedTargets} contains
   * the transitive closure of the specified {@code scope} and {@code collectedPackages} contains
   * (at least; see [0] below) all the packages for the targets in {@code collectedTargets}.
   *
   * <p>([0] In the future, {@code collectedPackages} might also contain packages needed to evaluate
   * "buildfiles" functions; see b/123795023.)
   *
   * <p>The {@code labelsToVisitNextRestart} field contains labels of targets belonging to
   * previously unloaded packages, the "frontier" of the last Skyframe evaluation attempt's
   * traversal.
   */
  private static class ScopeTraversal implements SkyKeyComputeState {
    private final LinkedHashMap<PackageIdentifier, Package> collectedPackages =
        new LinkedHashMap<>();
    private final LinkedHashMap<Label, Target> collectedTargets = new LinkedHashMap<>();
    private final LinkedHashSet<Label> labelsToVisitNextRestart = new LinkedHashSet<>();

    private ScopeTraversal(Collection<Label> initialScope) {
      labelsToVisitNextRestart.addAll(initialScope);
    }
  }

  @Nullable
  @Override
  public GenQueryPackageProvider constructPackageMap(Environment env, ImmutableList<Label> scope)
      throws InterruptedException, BrokenQueryScopeException {
    SkyValue value = env.getValueOrThrow(Key.create(scope), BrokenQueryScopeException.class);
    if (value == null) {
      return null;
    }
    return ((Value) value).genQueryPackageProvider;
  }

  @Nullable
  private static GenQueryPackageProvider constructPackageMapImpl(
      Environment env, ImmutableList<Label> scope)
      throws InterruptedException, BrokenQueryScopeException {
    ScopeTraversal traversal = env.getState(() -> new ScopeTraversal(scope));

    LinkedHashSet<Label> labelsToVisit = new LinkedHashSet<>(traversal.labelsToVisitNextRestart);
    traversal.labelsToVisitNextRestart.clear();

    // Constructing these here minimizes garbage creation. They're used in dep traversals below.
    var attrDepConsumer =
        new LabelProcessor() {
          LinkedHashSet<Label> nextLabelsToVisitRef = null;

          boolean attrDepNeedsRestart = false;
          boolean attrDepUnvisited = false;
          boolean hasAspects = false;
          HashMultimap<Attribute, Label> transitions = null;

          @Override
          public void process(Target from, @Nullable Attribute attribute, Label to) {
            if (hasAspects
                && !attrDepNeedsRestart
                && traversal.labelsToVisitNextRestart.contains(to)) {
              attrDepNeedsRestart = true;
              return;
            }
            if (!traversal.collectedTargets.containsKey(to)) {
              attrDepUnvisited = true;
              nextLabelsToVisitRef.add(to);
              return;
            }

            if (hasAspects
                && !attrDepNeedsRestart
                && !attrDepUnvisited
                && attribute != null
                && DependencyFilter.NO_NODEP_ATTRIBUTES.test((Rule) from, attribute)) {
              transitions.put(attribute, to);
            }
          }
        };

    var aspectDepConsumer =
        new BiConsumer<Attribute, Label>() {
          LinkedHashSet<Label> nextLabelsToVisitRef = null;

          @Override
          public void accept(Attribute aspectAttribute, Label aspectLabel) {
            if (!traversal.collectedTargets.containsKey(aspectLabel)) {
              nextLabelsToVisitRef.add(aspectLabel);
            }
          }
        };

    while (!labelsToVisit.isEmpty()) {
      LinkedHashSet<Label> nextLabelsToVisit = new LinkedHashSet<>();
      attrDepConsumer.nextLabelsToVisitRef = nextLabelsToVisit;
      aspectDepConsumer.nextLabelsToVisitRef = nextLabelsToVisit;
      for (Label label : labelsToVisit) {

        // If this is the first time label is visited, then collectedTargets will not contain an
        // entry for it. The else branch will do one of three things:
        // 1) discover that there is a problem with the label's package. If so, this throws
        //    BrokenQueryScopeException to stop this genquery evaluation.
        // 2) discover that needed package information has not been computed by Skyframe. If so,
        //    this records that label must be visited after the next Skyframe restart by adding it
        //    to labelsToVisitNextRestart; at that time that package information will have been
        //    computed.
        // 3) use the package information already computed by Skyframe to collect the label's target
        //    and package.
        //
        // Labels may be visited a second time. This happens if at least one of the label's target
        // is a rule with aspects and its dependency attributes' labels hadn't been visited when the
        // label was first visited. Note that this is the typical case for such rules! This code
        // ensures that all of a rule's dependency attributes' labels are visited at least once
        // before its label is visited a second time.
        //
        // If a rule's dependency attributes' labels have all already been visited (which may
        // occur the first time a label is visited, but is guaranteed to occur if it's visited a
        // second time) then:
        // 1) if all those dependency attributes' labels' targets have been collected, then this
        //    code will enqueue the rule's aspect dependencies' labels for visitation.
        // 2) otherwise, at least one of those dependency attributes' labels must have been added to
        //    labelsToVisitNextRestart, so the rule's aspect dependencies can't be computed during
        //    this Skyframe restart, so the rule's label also must be visited after the next
        //    Skyframe restart.

        Target target = traversal.collectedTargets.get(label);
        if (target == null) {
          TargetAndErrorIfAny targetAndErrorIfAny;
          try {
            targetAndErrorIfAny = TargetLoadingUtil.loadTarget(env, label);
          } catch (NoSuchTargetException | NoSuchPackageException e) {
            throw new BrokenQueryScopeException(
                "errors were encountered while computing transitive closure of the scope.", e);
          }

          if (targetAndErrorIfAny == null) {
            traversal.labelsToVisitNextRestart.add(label);
            continue;
          }
          if (!targetAndErrorIfAny.isPackageLoadedSuccessfully()) {
            throw new BrokenQueryScopeException(
                "errors were encountered while computing transitive closure of the scope.");
          }

          target = targetAndErrorIfAny.getTarget();
          traversal.collectedTargets.put(label, target);
          traversal.collectedPackages.put(label.getPackageIdentifier(), target.getPackage());
        }

        attrDepConsumer.attrDepNeedsRestart = false;
        attrDepConsumer.attrDepUnvisited = false;
        attrDepConsumer.hasAspects = target instanceof Rule && ((Rule) target).hasAspects();
        attrDepConsumer.transitions = attrDepConsumer.hasAspects ? HashMultimap.create() : null;
        LabelVisitationUtils.visitTarget(
            target, DependencyFilter.NO_NODEP_ATTRIBUTES_EXCEPT_VISIBILITY, attrDepConsumer);

        if (!attrDepConsumer.hasAspects) {
          continue;
        }

        if (attrDepConsumer.attrDepNeedsRestart) {
          traversal.labelsToVisitNextRestart.add(label);
          continue;
        } else if (attrDepConsumer.attrDepUnvisited) {
          // This schedules label to be visited a second time during this Skyframe restart. Because
          // the loop above scheduled its unvisited attribute deps for visitation, and
          // nextLabelsToVisit preserves insertion order, when label is visited a second time,
          // attributeDepUnvisited will be false, and its aspect deps will be computable.
          nextLabelsToVisit.add(label);
          continue;
        }

        Rule rule = (Rule) target;
        for (Attribute attribute : attrDepConsumer.transitions.keySet()) {
          for (Aspect aspect : attribute.getAspects(rule)) {
            if (hasDepThatSatisfies(
                rule,
                aspect,
                attrDepConsumer.transitions.get(attribute),
                traversal.collectedTargets)) {
              AspectDefinition.forEachLabelDepFromAllAttributesOfAspect(
                  rule, aspect, DependencyFilter.ALL_DEPS, aspectDepConsumer);
            }
          }
        }
      }
      labelsToVisit = nextLabelsToVisit;
    }
    if (env.valuesMissing()) {
      return null;
    }

    return new GenQueryPackageProvider(
        ImmutableMap.copyOf(traversal.collectedPackages),
        ImmutableMap.copyOf(traversal.collectedTargets));
  }

  private static boolean hasDepThatSatisfies(
      Rule fromRule, Aspect aspect, Iterable<Label> toLabels, Map<Label, Target> targets) {
    for (Label toLabel : toLabels) {
      Target toTarget =
          Preconditions.checkNotNull(
              targets.get(toLabel),
              "%s dep %s should have been visited but was not",
              fromRule.getLabel(),
              toLabel);
      AdvertisedProviderSet advertisedProviderSet =
          toTarget instanceof Rule
              ? ((Rule) toTarget).getRuleClassObject().getAdvertisedProviders()
              : null;
      if (advertisedProviderSet != null
          && AspectDefinition.satisfies(aspect, advertisedProviderSet)) {
        return true;
      }
    }
    return false;
  }
}
