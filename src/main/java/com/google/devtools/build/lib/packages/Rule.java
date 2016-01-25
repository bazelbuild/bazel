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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.BinaryPredicate;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * An instance of a build rule in the build language.  A rule has a name, a
 * package to which it belongs, a class such as <code>cc_library</code>, and
 * set of typed attributes.  The set of attribute names and types is a property
 * of the rule's class.  The use of the term "class" here has nothing to do
 * with Java classes.  All rules are implemented by the same Java classes, Rule
 * and RuleClass.
 *
 * <p>Here is a typical rule as it appears in a BUILD file:
 * <pre>
 * cc_library(name = 'foo',
 *            defines = ['-Dkey=value'],
 *            srcs = ['foo.cc'],
 *            deps = ['bar'])
 * </pre>
 */
public final class Rule implements Target, DependencyFilter.AttributeInfoProvider {

  /** Label predicate that allows every label. */
  public static final Predicate<Label> ALL_LABELS = Predicates.alwaysTrue();

  private final Label label;

  private final Package pkg;

  private final RuleClass ruleClass;

  private final AttributeContainer attributes;
  private final RawAttributeMapper attributeMap;

  private RuleVisibility visibility;

  private boolean containsErrors;

  private final Location location;

  private final String workspaceName;

  // Initialized in the call to populateOutputFiles.
  private List<OutputFile> outputFiles;
  private ListMultimap<String, OutputFile> outputFileMap;

  Rule(Package pkg, Label label, RuleClass ruleClass, Location location,
      AttributeContainer attributeContainer) {
    this.pkg = Preconditions.checkNotNull(pkg);
    this.label = label;
    this.ruleClass = Preconditions.checkNotNull(ruleClass);
    this.location = Preconditions.checkNotNull(location);
    this.attributes = attributeContainer;
    this.attributeMap = new RawAttributeMapper(pkg, ruleClass, label, attributes);
    this.containsErrors = false;
    this.workspaceName = pkg.getWorkspaceName();
  }

  void setVisibility(RuleVisibility visibility) {
    this.visibility = visibility;
  }

  void setAttributeValue(Attribute attribute, Object value, boolean explicit) {
    attributes.setAttributeValue(attribute, value, explicit);
  }

  void setAttributeValueByName(String attrName, Object value) {
    attributes.setAttributeValueByName(attrName, value);
  }

  void setAttributeLocation(int attrIndex, Location location) {
    attributes.setAttributeLocation(attrIndex, location);
  }

  void setContainsErrors() {
    this.containsErrors = true;
  }

  /**
   * Returns the name of the workspace that this rule is in.
   */
  public String getWorkspaceName() {
    return workspaceName;
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public String getName() {
    return label.getName();
  }

  @Override
  public Package getPackage() {
    return pkg;
  }

  public RuleClass getRuleClassObject() {
    return ruleClass;
  }

  @Override
  public String getTargetKind() {
    return ruleClass.getTargetKind();
  }

  /**
   * Returns the class of this rule. (e.g. "cc_library")
   */
  public String getRuleClass() {
    return ruleClass.getName();
  }

  /**
   * Returns the build features that apply to this rule.
   */
  public ImmutableSet<String> getFeatures() {
    return pkg.getFeatures();
  }

  /**
   * Returns true iff the outputs of this rule should be created beneath the
   * bin directory, false if beneath genfiles.  For most rule
   * classes, this is a constant, but for genrule, it is a property of the
   * individual rule instance, derived from the 'output_to_bindir' attribute.
   */
  public boolean hasBinaryOutput() {
    return ruleClass.getName().equals("genrule") // this is unfortunate...
        ? NonconfigurableAttributeMapper.of(this).get("output_to_bindir", Type.BOOLEAN)
        : ruleClass.hasBinaryOutput();
  }

  /**
   * Returns true iff there were errors while constructing this rule, such as
   * attributes with missing values or values of the wrong type.
   */
  public boolean containsErrors() {
    return containsErrors;
  }

  /**
   * Returns an (unmodifiable, unordered) collection containing all the
   * Attribute definitions for this kind of rule.  (Note, this doesn't include
   * the <i>values</i> of the attributes, merely the schema.  Call
   * get[Type]Attr() methods to access the actual values.)
   */
  public Collection<Attribute> getAttributes() {
    return ruleClass.getAttributes();
  }

  /**
   * Returns true if this rule has any attributes that are configurable.
   *
   * <p>Note this is *not* the same as having attribute *types* that are configurable. For example,
   * "deps" is configurable, in that one can write a rule that sets "deps" to a configuration
   * dictionary. But if *this* rule's instance of "deps" doesn't do that, its instance
   * of "deps" is not considered configurable.
   *
   * <p>In other words, this method signals which rules might have their attribute values
   * influenced by the configuration.
   */
  public boolean hasConfigurableAttributes() {
    for (Attribute attribute : getAttributes()) {
      if (attributeMap.isConfigurable(attribute.getName(), attribute.getType())) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns true if the given attribute is configurable.
   */
  public boolean isConfigurableAttribute(String attributeName) {
    return attributeMap.isConfigurable(attributeName, attributeMap.getAttributeType(attributeName));
  }

  /**
   * Returns the attribute definition whose name is {@code attrName}, or null
   * if not found.  (Use get[X]Attr for the actual value.)
   *
   * @deprecated use {@link AbstractAttributeMapper#getAttributeDefinition} instead
   */
  @Deprecated
  public Attribute getAttributeDefinition(String attrName) {
    return attributeMap.getAttributeDefinition(attrName);
  }

  /**
   * Returns an (unmodifiable, ordered) collection containing all the declared output files of this
   * rule.
   *
   * <p>All implicit output files (declared in the {@link RuleClass}) are
   * listed first, followed by any explicit files (declared via the 'outs' attribute). Additionally
   * both implicit and explicit outputs will retain the relative order in which they were declared.
   *
   * <p>This ordering is useful because it is propagated through to the list of targets returned by
   * getOuts() and allows targets to access their implicit outputs easily via
   * {@code getOuts().get(N)} (providing that N is less than the number of implicit outputs).
   *
   * <p>The fact that the relative order of the explicit outputs is also retained is less obviously
   * useful but is still well defined.
   */
  public Collection<OutputFile> getOutputFiles() {
    return outputFiles;
  }

  /**
   * Returns an (unmodifiable, ordered) map containing the list of output files for every
   * output type attribute.
   */
  public ListMultimap<String, OutputFile> getOutputFileMap() {
    return outputFileMap;
  }

  @Override
  public Location getLocation() {
    return location;
  }

  @Override
  public Rule getAssociatedRule() {
    return this;
  }

  /**
   * Returns this rule's raw attribute info, suitable for being fed into an
   * {@link AttributeMap} for user-level attribute access. Don't use this method
   * for direct attribute access.
   */
  public AttributeContainer getAttributeContainer() {
    return attributes;
  }

  /********************************************************************
   * Attribute accessor functions.
   *
   * The below provide access to attribute definitions and other generic
   * metadata.
   *
   * For access to attribute *values* (e.g. "What's the value of attribute
   * X for Rule Y?"), go through {@link RuleContext#attributes}. If no
   * RuleContext is available, create a localized {@link AbstractAttributeMapper}
   * instance instead.
   ********************************************************************/

  /**
   * Returns the default value for the attribute {@code attrName}, which may be
   * of any type, but must exist (an exception is thrown otherwise).
   */
  public Object getAttrDefaultValue(String attrName) {
    Object defaultValue = ruleClass.getAttributeByName(attrName).getDefaultValue(this);
    // Computed defaults not expected here.
    Preconditions.checkState(!(defaultValue instanceof Attribute.ComputedDefault));
    return defaultValue;
  }

  /**
   * Returns true iff the rule class has an attribute with the given name and type.
   */
  public boolean isAttrDefined(String attrName, Type<?> type) {
    return ruleClass.hasAttr(attrName, type);
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    return attributes.isAttributeValueExplicitlySpecified(attribute);
  }

  /**
   * Returns true iff the value of the specified attribute is explicitly set in the BUILD file (as
   * opposed to its default value). This also returns true if the value from the BUILD file is the
   * same as the default value. In addition, this method return false if the rule has no attribute
   * with the given name.
   */
  public boolean isAttributeValueExplicitlySpecified(String attrName) {
    return attributeMap.isAttributeValueExplicitlySpecified(attrName);
  }

  /**
   * Returns the location of the attribute definition for this rule, if known;
   * or the location of the whole rule otherwise. "attrName" need not be a
   * valid attribute name for this rule.
   * 
   * <p>This method ignores whether the present rule was created by a macro or not.
   */
  public Location getAttributeLocationWithoutMacro(String attrName) {
    return getAttributeLocation(attrName, false /* useBuildLocation */);
  }

  /**
   * Returns the location of the attribute definition for this rule, if known;
   * or the location of the whole rule otherwise. "attrName" need not be a
   * valid attribute name for this rule.
   *
   * <p>If this rule was created by a macro, this method returns the
   * location of the macro invocation in the BUILD file instead.
   */
  public Location getAttributeLocation(String attrName) {
    return getAttributeLocation(attrName, true /* useBuildLocation */);
  }

  private Location getAttributeLocation(String attrName, boolean useBuildLocation) {
    /*
     * If the rule was created by a macro, we have to deal with two locations: one in the BUILD
     * file where the macro is invoked and one in the bzl file where the rule is created.
     * For error reporting, we are usually more interested in the former one.
     * Different methods in this class refer to different locations, though:
     * - getLocation() points to the location of the macro invocation in the BUILD file (thanks to
     *   RuleFactory).
     * - attributes.getAttributeLocation() points to the location in the bzl file.
     */
    if (wasCreatedByMacro() && useBuildLocation) {
      return getLocation();
    }

    Location attrLocation = null;
    if (!attrName.equals("name")) {
      attrLocation = attributes.getAttributeLocation(attrName);
    }
    return attrLocation != null ? attrLocation : getLocation();
  }

  /**
   * Returns whether this rule was created by a macro.
   */
  public boolean wasCreatedByMacro() {
    return hasStringAttribute("generator_name") || hasStringAttribute("generator_function");
  }

  private boolean hasStringAttribute(String attrName) {
    Object value = attributes.getAttr(attrName);
    if (value != null && value instanceof String) {
      return !((String) value).isEmpty();
    }
    return false;
  }

  /**
   * Returns a new List instance containing all direct dependencies (all types).
   */
  public Collection<Label> getLabels() {
    return getLabels(DependencyFilter.ALL_DEPS);
  }

  /**
   * Returns a new Collection containing all Labels that match a given Predicate,
   * not including outputs.
   *
   * @param predicate A binary predicate that determines if a label should be
   *     included in the result. The predicate is evaluated with this rule and
   *     the attribute that contains the label. The label will be contained in the
   *     result iff (the predicate returned {@code true} and the labels are not outputs)
   */
  public Collection<Label> getLabels(BinaryPredicate<? super Rule, Attribute> predicate) {
    return ImmutableSortedSet.copyOf(getTransitions(predicate).values());
  }

  /**
   * Returns a new Multimap containing all attributes that match a given Predicate and
   * corresponding labels, not including outputs.
   *
   * @param predicate A binary predicate that determines if a label should be
   *     included in the result. The predicate is evaluated with this rule and
   *     the attribute that contains the label. The label will be contained in the
   *     result iff (the predicate returned {@code true} and the labels are not outputs)
   */
  public Multimap<Attribute, Label> getTransitions(
      final BinaryPredicate<? super Rule, Attribute> predicate) {
    final Multimap<Attribute, Label> transitions = HashMultimap.create();
    // TODO(bazel-team): move this to AttributeMap, too. Just like visitLabels, which labels should
    // be visited may depend on the calling context. We shouldn't implicitly decide this for
    // the caller.
    AggregatingAttributeMapper.of(this).visitLabels(new AttributeMap.AcceptsLabelAttribute() {
      @Override
      public void acceptLabelAttribute(Label label, Attribute attribute) {
        if (predicate.apply(Rule.this, attribute)) {
          transitions.put(attribute, label);
        }
      }
    });
    return transitions;
  }

  /**
   * Check if this rule is valid according to the validityPredicate of its RuleClass.
   */
  void checkValidityPredicate(EventHandler eventHandler) {
    PredicateWithMessage<Rule> predicate = getRuleClassObject().getValidityPredicate();
    if (!predicate.apply(this)) {
      reportError(predicate.getErrorReason(this), eventHandler);
    }
  }

  /**
   * Collects the output files (both implicit and explicit). All the implicit output files are added
   * first, followed by any explicit files. Additionally both implicit and explicit output files
   * will retain the relative order in which they were declared.
   */
  void populateOutputFiles(EventHandler eventHandler, Package.Builder pkgBuilder)
      throws LabelSyntaxException, InterruptedException {
    Preconditions.checkState(outputFiles == null);
    // Order is important here: implicit before explicit
    outputFiles = Lists.newArrayList();
    outputFileMap = LinkedListMultimap.create();
    populateImplicitOutputFiles(eventHandler, pkgBuilder);
    populateExplicitOutputFiles(eventHandler);
    outputFiles = ImmutableList.copyOf(outputFiles);
    outputFileMap = ImmutableListMultimap.copyOf(outputFileMap);
  }

  // Explicit output files are user-specified attributes of type OUTPUT.
  private void populateExplicitOutputFiles(EventHandler eventHandler) throws LabelSyntaxException {
    NonconfigurableAttributeMapper nonConfigurableAttributes =
        NonconfigurableAttributeMapper.of(this);
    for (Attribute attribute : ruleClass.getAttributes()) {
      String name = attribute.getName();
      Type<?> type = attribute.getType();
      if (type == BuildType.OUTPUT) {
        Label outputLabel = nonConfigurableAttributes.get(name, BuildType.OUTPUT);
        if (outputLabel != null) {
          addLabelOutput(attribute, outputLabel, eventHandler);
        }
      } else if (type == BuildType.OUTPUT_LIST) {
        for (Label label : nonConfigurableAttributes.get(name, BuildType.OUTPUT_LIST)) {
          addLabelOutput(attribute, label, eventHandler);
        }
      }
    }
  }

  /**
   * Implicit output files come from rule-specific patterns, and are a function
   * of the rule's "name", "srcs", and other attributes.
   */
  private void populateImplicitOutputFiles(EventHandler eventHandler, Package.Builder pkgBuilder)
      throws InterruptedException {
    try {
      for (String out : ruleClass.getImplicitOutputsFunction().getImplicitOutputs(attributeMap)) {
        try {
          addOutputFile(pkgBuilder.createLabel(out), eventHandler);
        } catch (LabelSyntaxException e) {
          reportError("illegal output file name '" + out + "' in rule "
                      + getLabel(), eventHandler);
        }
      }
    } catch (EvalException e) {
      reportError(e.print(), eventHandler);
    }
  }

  private void addLabelOutput(Attribute attribute, Label label, EventHandler eventHandler)
      throws LabelSyntaxException {
    if (!label.getPackageIdentifier().equals(pkg.getPackageIdentifier())) {
      throw new IllegalStateException("Label for attribute " + attribute
          + " should refer to '" + pkg.getName()
          + "' but instead refers to '" + label.getPackageFragment()
          + "' (label '" + label.getName() + "')");
    }
    if (label.getName().equals(".")) {
      throw new LabelSyntaxException("output file name can't be equal '.'");
    }
    OutputFile outputFile = addOutputFile(label, eventHandler);
    outputFileMap.put(attribute.getName(), outputFile);
  }

  private OutputFile addOutputFile(Label label, EventHandler eventHandler) {
    if (label.getName().equals(getName())) {
      // TODO(bazel-team): for now (23 Apr 2008) this is just a warning.  After
      // June 1st we should make it an error.
      reportWarning("target '" + getName() + "' is both a rule and a file; please choose "
                    + "another name for the rule", eventHandler);
    }
    OutputFile outputFile = new OutputFile(pkg, label, this);
    outputFiles.add(outputFile);
    return outputFile;
  }

  void reportError(String message, EventHandler eventHandler) {
    eventHandler.handle(Event.error(location, message));
    this.containsErrors = true;
  }

  void reportWarning(String message, EventHandler eventHandler) {
    eventHandler.handle(Event.warn(location, message));
  }

  /**
   * Returns a string of the form "cc_binary rule //foo:foo"
   *
   * @return a string of the form "cc_binary rule //foo:foo"
   */
  @Override
  public String toString() {
    return getRuleClass() + " rule " + getLabel();
  }

 /**
   * Returns the effective visibility of this Rule. Visibility is computed from
   * these sources in this order of preference:
   *   - 'visibility' attribute
   *   - 'default_visibility;' attribute of package() declaration
   *   - public.
   */
  @Override
  public RuleVisibility getVisibility() {
    if (visibility != null) {
      return visibility;
    }

    if (getRuleClassObject().isPublicByDefault()) {
      return ConstantRuleVisibility.PUBLIC;
    }

    return pkg.getDefaultVisibility();
  }

  public boolean isVisibilitySpecified() {
    return visibility != null;
  }

  @Override
  public boolean isConfigurable() {
    return true;
  }

  @Override
  @SuppressWarnings("unchecked")
  public Set<DistributionType> getDistributions() {
    if (isAttrDefined("distribs", BuildType.DISTRIBUTIONS)
        && isAttributeValueExplicitlySpecified("distribs")) {
      return NonconfigurableAttributeMapper.of(this).get("distribs", BuildType.DISTRIBUTIONS);
    } else {
      return getPackage().getDefaultDistribs();
    }
  }

  @Override
  public License getLicense() {
    if (isAttrDefined("licenses", BuildType.LICENSE)
        && isAttributeValueExplicitlySpecified("licenses")) {
      return NonconfigurableAttributeMapper.of(this).get("licenses", BuildType.LICENSE);
    } else {
      return getPackage().getDefaultLicense();
    }
  }

  /**
   * Returns the license of the output of the binary created by this rule, or
   * null if it is not specified.
   */
  public License getToolOutputLicense(AttributeMap attributes) {
    if (isAttrDefined("output_licenses", BuildType.LICENSE)
        && attributes.isAttributeValueExplicitlySpecified("output_licenses")) {
      return attributes.get("output_licenses", BuildType.LICENSE);
    } else {
      return null;
    }
  }

  /**
   * Returns the globs that were expanded to create an attribute value, or
   * null if unknown or not applicable.
   */
  public static GlobList<?> getGlobInfo(Object attributeValue) {
    if (attributeValue instanceof GlobList<?>) {
      return (GlobList<?>) attributeValue;
    } else {
      return null;
    }
  }

  private void checkForNullLabel(Label labelToCheck, Object context) {
    if (labelToCheck == null) {
      throw new IllegalStateException(String.format(
          "null label in rule %s, %s", getLabel().toString(), context));
    }
  }

  // Consistency check: check if this label contains any weird labels (i.e.
  // null-valued, with a packageFragment that is null...). The bug that prompted
  // the introduction of this code is #2210848 (NullPointerException in
  // Package.checkForConflicts() ).
  void checkForNullLabels() {
    AggregatingAttributeMapper.of(this).visitLabels(
        new AttributeMap.AcceptsLabelAttribute() {
          @Override
          public void acceptLabelAttribute(Label labelToCheck, Attribute attribute) {
            checkForNullLabel(labelToCheck, attribute);
          }
        });
    for (OutputFile outputFile : getOutputFiles()) {
      checkForNullLabel(outputFile.getLabel(), "output file");
    }
  }

  /**
   * Returns the Set of all tags exhibited by this target.  May be empty.
   */
  public Set<String> getRuleTags() {
    Set<String> ruleTags = new LinkedHashSet<>();
    for (Attribute attribute : getRuleClassObject().getAttributes()) {
      if (attribute.isTaggable()) {
        Type<?> attrType = attribute.getType();
        String name = attribute.getName();
        // This enforces the expectation that taggable attributes are non-configurable.
        Object value = NonconfigurableAttributeMapper.of(this).get(name, attrType);
        Set<String> tags = attrType.toTagSet(value, name);
        ruleTags.addAll(tags);
      }
    }
    return ruleTags;
  }

  /**
   * Computes labels of additional dependencies that can be provided by aspects that this rule
   * can require from its direct dependencies.
   */
  public Collection<? extends Label> getAspectLabelsSuperset(DependencyFilter predicate) {
    LinkedHashMultimap<Attribute, Label> labels = LinkedHashMultimap.create();
    for (Attribute attribute : this.getAttributes()) {
      for (Aspect candidateClass : attribute.getAspects(this)) {
        AspectDefinition.addAllAttributesOfAspect(Rule.this, labels, candidateClass, predicate);
      }
    }
    return labels.values();
  }
}
