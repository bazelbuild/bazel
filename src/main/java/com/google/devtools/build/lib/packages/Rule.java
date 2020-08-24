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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.util.BinaryPredicate;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An instance of a build rule in the build language. A rule has a name, a package to which it
 * belongs, a class such as <code>cc_library</code>, and set of typed attributes. The set of
 * attribute names and types is a property of the rule's class. The use of the term "class" here has
 * nothing to do with Java classes. All rules are implemented by the same Java classes, Rule and
 * RuleClass.
 *
 * <p>Here is a typical rule as it appears in a BUILD file:
 *
 * <pre>
 * cc_library(name = 'foo',
 *            defines = ['-Dkey=value'],
 *            srcs = ['foo.cc'],
 *            deps = ['bar'])
 * </pre>
 */
// Non-final only for mocking in tests. Do not subclass!
public class Rule implements Target, DependencyFilter.AttributeInfoProvider {

  /** Label predicate that allows every label. */
  public static final Predicate<Label> ALL_LABELS = Predicates.alwaysTrue();

  private static final String GENERATOR_FUNCTION = "generator_function";
  private static final String GENERATOR_LOCATION = "generator_location";

  private final Label label;

  private final Package pkg;

  private final RuleClass ruleClass;

  private final AttributeContainer attributes;

  private RuleVisibility visibility;

  private boolean containsErrors;

  private final Location location;

  private final CallStack callstack;

  private final ImplicitOutputsFunction implicitOutputsFunction;

  // Initialized in the call to populateOutputFiles.
  private List<OutputFile> outputFiles;
  private ListMultimap<String, OutputFile> outputFileMap;

  Rule(
      Package pkg,
      Label label,
      RuleClass ruleClass,
      Location location,
      CallStack callstack,
      AttributeContainer attributeContainer) {
    this(
        pkg,
        label,
        ruleClass,
        location,
        callstack,
        attributeContainer,
        ruleClass.getDefaultImplicitOutputsFunction());
  }

  Rule(
      Package pkg,
      Label label,
      RuleClass ruleClass,
      Location location,
      CallStack callstack,
      AttributeContainer attributeContainer,
      ImplicitOutputsFunction implicitOutputsFunction) {
    this.pkg = Preconditions.checkNotNull(pkg);
    this.label = label;
    this.ruleClass = Preconditions.checkNotNull(ruleClass);
    this.location = Preconditions.checkNotNull(location);
    this.callstack = Preconditions.checkNotNull(callstack);
    this.attributes = attributeContainer;
    this.implicitOutputsFunction = implicitOutputsFunction;
    this.containsErrors = false;
  }

  void setVisibility(RuleVisibility visibility) {
    this.visibility = visibility;
  }

  void setAttributeValue(Attribute attribute, Object value, boolean explicit) {
    attributes.setAttributeValue(attribute, value, explicit);
  }

  void setContainsErrors() {
    this.containsErrors = true;
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

  /** Returns true if this rule is an analysis test (set by analysis_test = true). */
  public boolean isAnalysisTest() {
    return ruleClass.isAnalysisTest();
  }

  /**
   * Returns true if this rule has at least one attribute with an analysis test transition. (A
   * starlark-defined transition using analysis_test_transition()).
   */
  public boolean hasAnalysisTestTransition() {
    return ruleClass.hasAnalysisTestTransition();
  }

  public boolean isBuildSetting() {
    return ruleClass.getBuildSetting() != null;
  }

  /**
   * Returns true iff there were errors while constructing this rule, such as
   * attributes with missing values or values of the wrong type.
   */
  public boolean containsErrors() {
    return containsErrors;
  }

  public boolean hasAspects() {
    return ruleClass.hasAspects();
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
   * Returns true if the given attribute is configurable.
   */
  public boolean isConfigurableAttribute(String attributeName) {
    Attribute attribute = ruleClass.getAttributeByNameMaybe(attributeName);
    return attribute != null
        ? AbstractAttributeMapper.isConfigurable(this, attributeName, attribute.getType())
        : false;
  }

  /**
   * Returns the attribute definition whose name is {@code attrName}, or null
   * if not found.  (Use get[X]Attr for the actual value.)
   *
   * @deprecated use {@link AbstractAttributeMapper#getAttributeDefinition} instead
   */
  @Deprecated
  public Attribute getAttributeDefinition(String attrName) {
    return ruleClass.getAttributeByNameMaybe(attrName);
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

  /** Returns the stack of function calls active when this rule was instantiated. */
  public CallStack getCallStack() {
    return callstack;
  }

  public ImplicitOutputsFunction getImplicitOutputsFunction() {
    return implicitOutputsFunction;
  }

  @Override
  public Rule getAssociatedRule() {
    return this;
  }

  /**
   * Returns this rule's raw attribute info, suitable for being fed into an {@link AttributeMap} for
   * user-level attribute access. Don't use this method for direct attribute access.
   */
  AttributeContainer getAttributeContainer() {
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
   *
   * <p>Note: RuleContext also has isAttrDefined(), which takes Aspects into account. Whenever
   * possible, use RuleContext.isAttrDefined() instead of this method.
   */
  public boolean isAttrDefined(String attrName, Type<?> type) {
    return ruleClass.hasAttr(attrName, type);
  }

  @Nullable
  private Object getAttrWithIndex(int attrIndex) {
    Object value = attributes.getAttributeValue(attrIndex);
    if (value != null) {
      return value;
    }
    Attribute attr = ruleClass.getAttribute(attrIndex);
    if (attr.hasComputedDefault()) {
      // Don't even try to compute it.
      // Correctness of this relies on the fact that at Rule creation time
      // we did not skip populating attributes with a computed default.
      return null;
    }
    switch (attr.getName()) {
      case GENERATOR_FUNCTION:
        return callstack.size() > 1 ? callstack.getFrame(1).name : "";
      case GENERATOR_LOCATION:
        return callstack.size() > 1 ? relativeLocation(callstack.getFrame(0).location) : "";
      default:
        return attr.getDefaultValue(null);
    }
  }

  @Nullable
  private String relativeLocation(Location location) {
    // Determining the workspace root only works reliably if both location and label point to files
    // in the same package.
    // It would be preferable to construct the path from the label itself, but this doesn't work for
    // rules created from function calls in a subincluded file, even if both files share a path
    // prefix (for example, when //a/package:BUILD subincludes //a/package/with/a/subpackage:BUILD).
    // We can revert to that approach once subincludes aren't supported anymore.
    //
    // TODO(b/151165647): this logic has always been wrong:
    // it spuriously matches occurrences of the package name earlier in the path.
    String absolutePath = location.toString();
    int pos = absolutePath.indexOf(getLabel().getPackageName());
    return (pos < 0) ? null : absolutePath.substring(pos);
  }

  /**
   * Returns the value of the given attribute for this rule. Returns null for invalid attributes and
   * default value if attribute was not set.
   *
   * @param attrName the name of the attribute to lookup.
   */
  @Nullable
  public Object getAttr(String attrName) {
    Integer attrIndex = ruleClass.getAttributeIndex(attrName);
    return attrIndex == null ? null : getAttrWithIndex(attrIndex);
  }

  /**
   * Returns the value of the given attribute if it has the right type.
   *
   * @throws IllegalArgumentException if the attribute does not have the expected type.
   */
  @Nullable
  public <T> Object getAttr(String attrName, Type<T> type) {
    Integer index = ruleClass.getAttributeIndex(attrName);
    if (index == null) {
      throw new IllegalArgumentException(
          "No such attribute " + attrName + " in " + ruleClass + " rule " + getLabel());
    }
    Attribute attr = ruleClass.getAttribute(index);
    if (attr.getType() != type) {
      throw new IllegalArgumentException(
          "Attribute "
              + attrName
              + " is of type "
              + attr.getType()
              + " and not of type "
              + type
              + " in "
              + ruleClass
              + " rule "
              + getLabel());
    }
    return getAttrWithIndex(index);
  }

  /**
   * Returns a {@link BuildType.SelectorList} for the given attribute if the attribute is
   * configurable for this rule, null otherwise.
   */
  @Nullable
  @SuppressWarnings("unchecked")
  public <T> BuildType.SelectorList<T> getSelectorList(String attributeName, Type<T> type) {
    Integer index = ruleClass.getAttributeIndex(attributeName);
    if (index == null) {
      return null;
    }
    Object attrValue = attributes.getAttributeValue(index);
    if (!(attrValue instanceof BuildType.SelectorList)) {
      return null;
    }
    if (((BuildType.SelectorList<?>) attrValue).getOriginalType() != type) {
      throw new IllegalArgumentException(
          "Attribute "
              + attributeName
              + " is not of type "
              + type
              + " in "
              + ruleClass
              + " rule "
              + getLabel());
    }
    return (BuildType.SelectorList<T>) attrValue;
  }

  /**
   * See {@link #isAttributeValueExplicitlySpecified(String)}
   */
  @Override
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    return isAttributeValueExplicitlySpecified(attribute.getName());
  }

  /**
   * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. This
   * returns true also if the value explicitly specified in the BUILD file is the same as the
   * attribute's default value. In addition, this method return false if the rule has no attribute
   * with the given name.
   */
  public boolean isAttributeValueExplicitlySpecified(String attrName) {
    if (attrName.equals(GENERATOR_FUNCTION) || attrName.equals(GENERATOR_LOCATION)) {
      return wasCreatedByMacro();
    }
    return attributes.isAttributeValueExplicitlySpecified(attrName);
  }

  /**
   * Returns whether this rule was created by a macro.
   */
  public boolean wasCreatedByMacro() {
    return hasStringAttribute("generator_name") || hasStringAttribute(GENERATOR_FUNCTION);
  }

  /** Returns the macro that generated this rule, or an empty string. */
  public String getGeneratorFunction() {
    Object value = getAttr(GENERATOR_FUNCTION);
    if (value instanceof String) {
      return (String) value;
    }
    return "";
  }

  private boolean hasStringAttribute(String attrName) {
    Object value = getAttr(attrName);
    if (value instanceof String) {
      return !((String) value).isEmpty();
    }
    return false;
  }

  /** Returns a new List instance containing all direct dependencies (all types). */
  public Collection<Label> getLabels() {
    final List<Label> labels = Lists.newArrayList();
    AggregatingAttributeMapper.of(this)
        .visitLabels()
        .stream()
        .map(AttributeMap.DepEdge::getLabel)
        .forEach(labels::add);
    return labels;
  }

  /**
   * Returns a new Collection containing all Labels that match a given Predicate, not including
   * outputs.
   *
   * @param predicate A binary predicate that determines if a label should be included in the
   *     result. The predicate is evaluated with this rule and the attribute that contains the
   *     label. The label will be contained in the result iff (the predicate returned {@code true}
   *     and the labels are not outputs)
   */
  public Collection<Label> getLabels(BinaryPredicate<? super Rule, Attribute> predicate) {
    return ImmutableSortedSet.copyOf(getTransitions(predicate).values());
  }

  /**
   * Returns a new Multimap containing all attributes that match a given Predicate and corresponding
   * labels, not including outputs.
   *
   * @param predicate A binary predicate that determines if a label should be included in the
   *     result. The predicate is evaluated with this rule and the attribute that contains the
   *     label. The label will be contained in the result iff (the predicate returned {@code true}
   *     and the labels are not outputs)
   */
  public Multimap<Attribute, Label> getTransitions(
      final BinaryPredicate<? super Rule, Attribute> predicate) {
    final Multimap<Attribute, Label> transitions = HashMultimap.create();
    // TODO(bazel-team): move this to AttributeMap, too. Just like visitLabels, which labels should
    // be visited may depend on the calling context. We shouldn't implicitly decide this for
    // the caller.
    AggregatingAttributeMapper.of(this)
        .visitLabels()
        .stream()
        .filter(depEdge -> predicate.apply(Rule.this, depEdge.getAttribute()))
        .forEach(depEdge -> transitions.put(depEdge.getAttribute(), depEdge.getLabel()));
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
    populateOutputFilesInternal(eventHandler, pkgBuilder, /*performChecks=*/ true);
  }

  void populateOutputFilesUnchecked(EventHandler eventHandler, Package.Builder pkgBuilder)
      throws InterruptedException {
    try {
      populateOutputFilesInternal(eventHandler, pkgBuilder, /*performChecks=*/ false);
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  private void populateOutputFilesInternal(
      EventHandler eventHandler, Package.Builder pkgBuilder, boolean performChecks)
      throws LabelSyntaxException, InterruptedException {
    Preconditions.checkState(outputFiles == null);
    // Order is important here: implicit before explicit
    ImmutableList.Builder<OutputFile> outputFilesBuilder = ImmutableList.builder();
    ImmutableListMultimap.Builder<String, OutputFile> outputFileMapBuilder =
        ImmutableListMultimap.builder();
    populateImplicitOutputFiles(eventHandler, pkgBuilder, outputFilesBuilder, performChecks);
    populateExplicitOutputFiles(
        eventHandler, outputFilesBuilder, outputFileMapBuilder, performChecks);
    outputFiles = outputFilesBuilder.build();
    outputFileMap = outputFileMapBuilder.build();
  }

  // Explicit output files are user-specified attributes of type OUTPUT.
  private void populateExplicitOutputFiles(
      EventHandler eventHandler,
      ImmutableList.Builder<OutputFile> outputFilesBuilder,
      ImmutableListMultimap.Builder<String, OutputFile> outputFileMapBuilder,
      boolean performChecks)
      throws LabelSyntaxException {
    NonconfigurableAttributeMapper nonConfigurableAttributes =
        NonconfigurableAttributeMapper.of(this);
    for (Attribute attribute : ruleClass.getAttributes()) {
      String name = attribute.getName();
      Type<?> type = attribute.getType();
      if (type == BuildType.OUTPUT) {
        Label outputLabel = nonConfigurableAttributes.get(name, BuildType.OUTPUT);
        if (outputLabel != null) {
          addLabelOutput(
              attribute,
              outputLabel,
              eventHandler,
              outputFilesBuilder,
              outputFileMapBuilder,
              performChecks);
        }
      } else if (type == BuildType.OUTPUT_LIST) {
        for (Label label : nonConfigurableAttributes.get(name, BuildType.OUTPUT_LIST)) {
          addLabelOutput(
              attribute,
              label,
              eventHandler,
              outputFilesBuilder,
              outputFileMapBuilder,
              performChecks);
        }
      }
    }
  }

  /**
   * Implicit output files come from rule-specific patterns, and are a function of the rule's
   * "name", "srcs", and other attributes.
   */
  private void populateImplicitOutputFiles(
      EventHandler eventHandler,
      Package.Builder pkgBuilder,
      ImmutableList.Builder<OutputFile> outputFilesBuilder,
      boolean performChecks)
      throws InterruptedException {
    try {
      RawAttributeMapper attributeMap = RawAttributeMapper.of(this);
      for (String out : implicitOutputsFunction.getImplicitOutputs(eventHandler, attributeMap)) {
        Label label;
        if (performChecks) {
          try {
            label = pkgBuilder.createLabel(out);
          } catch (LabelSyntaxException e) {
            reportError(
                "illegal output file name '"
                    + out
                    + "' in rule "
                    + getLabel()
                    + " due to: "
                    + e.getMessage(),
                eventHandler);
            continue;
          }
        } else {
          label = Label.createUnvalidated(pkgBuilder.getPackageIdentifier(), out);
        }
        addOutputFile(label, eventHandler, outputFilesBuilder);
      }
    } catch (EvalException e) {
      reportError(
          String.format("In rule %s: %s", getLabel(), e.getMessageWithStack()), eventHandler);
    }
  }

  private void addLabelOutput(
      Attribute attribute,
      Label label,
      EventHandler eventHandler,
      ImmutableList.Builder<OutputFile> outputFilesBuilder,
      ImmutableListMultimap.Builder<String, OutputFile> outputFileMapBuilder,
      boolean performChecks)
      throws LabelSyntaxException {
    if (performChecks) {
      if (!label.getPackageIdentifier().equals(pkg.getPackageIdentifier())) {
        throw new IllegalStateException("Label for attribute " + attribute
            + " should refer to '" + pkg.getName()
            + "' but instead refers to '" + label.getPackageFragment()
            + "' (label '" + label.getName() + "')");
      }
      if (label.getName().equals(".")) {
        throw new LabelSyntaxException("output file name can't be equal '.'");
      }
    }
    OutputFile outputFile = addOutputFile(label, eventHandler, outputFilesBuilder);
    outputFileMapBuilder.put(attribute.getName(), outputFile);
  }

  private OutputFile addOutputFile(
      Label label,
      EventHandler eventHandler,
      ImmutableList.Builder<OutputFile> outputFilesBuilder) {
    if (label.getName().equals(getName())) {
      // TODO(bazel-team): for now (23 Apr 2008) this is just a warning.  After
      // June 1st we should make it an error.
      reportWarning("target '" + getName() + "' is both a rule and a file; please choose "
                    + "another name for the rule", eventHandler);
    }
    OutputFile outputFile = new OutputFile(pkg, label, ruleClass.getOutputFileKind(), this);
    outputFilesBuilder.add(outputFile);
    return outputFile;
  }

  void reportError(String message, EventHandler eventHandler) {
    eventHandler.handle(Event.error(location, message));
    this.containsErrors = true;
  }

  private void reportWarning(String message, EventHandler eventHandler) {
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
    } else if (getRuleClassObject().ignoreLicenses()) {
      return License.NO_LICENSE;
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
    AggregatingAttributeMapper.of(this)
        .visitLabels()
        .forEach(depEdge -> checkForNullLabel(depEdge.getLabel(), depEdge.getAttribute()));
    getOutputFiles().forEach(outputFile -> checkForNullLabel(outputFile.getLabel(), "output file"));
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
    if (!hasAspects()) {
      return ImmutableList.of();
    }
    SetMultimap<Attribute, Label> labels = LinkedHashMultimap.create();
    for (Attribute attribute : this.getAttributes()) {
      for (Aspect candidateClass : attribute.getAspects(this)) {
        AspectDefinition.addAllAttributesOfAspect(Rule.this, labels, candidateClass, predicate);
      }
    }
    return labels.values();
  }

  /**
   * @return The repository name.
   */
  public RepositoryName getRepository() {
    return RepositoryName.createFromValidStrippedName(pkg.getWorkspaceName());
  }

  /** Returns the suffix of target kind for all rules. */
  public static String targetKindSuffix() {
    return " rule";
  }
}
