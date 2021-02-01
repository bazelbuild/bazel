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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.StarlarkImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.Package.ConfigSettingVisibilityPolicy;
import com.google.devtools.build.lib.util.BinaryPredicate;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.syntax.Location;

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

  private AttributeContainer attributes;

  private RuleVisibility visibility;

  private boolean containsErrors;

  private final Location location;

  private final CallStack callstack;

  private final ImplicitOutputsFunction implicitOutputsFunction;

  /**
   * A compact representation of a multimap from "output keys" to output files.
   *
   * <p>An output key is an identifier used to access the output in {@code ctx.outputs}, or the
   * empty string in the case of an output that's not exposed there. For explicit outputs, the
   * output key is the name of the attribute under which that output appears. For Starlark-defined
   * implicit outputs, the output key is determined by the dict returned from the Starlark function.
   * Native-defined implicit outputs are not named in this manner, and so are invisible to {@code
   * ctx.outputs} and use the empty string key. (It'd be pathological for the empty string to be
   * used as a key in the other two cases, but this class makes no attempt to prohibit that.)
   *
   * <p>Rather than naively store an ImmutableListMultimap, we save space by compressing it as an
   * ImmutableList, where each key is followed by all the values having that key. We distinguish
   * keys (Strings) from values (OutputFiles) by the fact that they have different types. The
   * accessor methods traverse the list and create a more user-friendly view.
   *
   * <p>To distinguish implicit outputs from explicit outputs, we store all the implicit outputs in
   * the list first, and record how many implicit output keys there are in a separate field.
   *
   * <p>The order of the implicit outputs is the same as returned by the implicit output function.
   * This allows a native rule implementation and native implicit outputs function to agree on the
   * index of a given kind of output. The order of explicit outputs preserves the attribute
   * iteration order and the order of values in a list attribute; the latter is important so that
   * {@code ctx.outputs.some_list} has a well-defined order.
   */
  // Both of these fields are initialized by populateOutputFiles().
  private ImmutableList<Object> flattenedOutputFileMap;

  private int numImplicitOutputKeys;

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
    Integer attrIndex = ruleClass.getAttributeIndex(attribute.getName());
    Preconditions.checkArgument(
        attrIndex != null, "attribute %s is not valid for this rule", attribute.getName());
    attributes.setAttributeValue(attrIndex, value, explicit);
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
    // TODO(murali): This method should be property of ruleclass not rule instance.
    // Further, this call to AbstractAttributeMapper.isConfigurable is delegated right back
    // to this instance!.
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
   * Constructs and returns an immutable list containing all the declared output files of this rule.
   *
   * <p>There are two kinds of outputs. Explicit outputs are declared in attributes of type OUTPUT
   * or OUTPUT_LABEL. Implicit outputs are determined by custom rule logic in an "implicit outputs
   * function" (either defined natively or in Starlark), and are named following a template pattern
   * based on the target's attributes.
   *
   * <p>All implicit output files (declared in the {@link RuleClass}) are listed first, followed by
   * any explicit files (declared via output attributes). Additionally, both implicit and explicit
   * outputs will retain the relative order in which they were declared.
   */
  public ImmutableList<OutputFile> getOutputFiles() {
    // Discard the String keys, taking only the OutputFile values.
    ImmutableList.Builder<OutputFile> result = ImmutableList.builder();
    for (Object o : flattenedOutputFileMap) {
      if (o instanceof OutputFile) {
        result.add((OutputFile) o);
      }
    }
    return result.build();
  }

  /**
   * Constructs and returns an immutable list of all the implicit output files of this rule, in the
   * order they were declared.
   */
  public ImmutableList<OutputFile> getImplicitOutputFiles() {
    ImmutableList.Builder<OutputFile> result = ImmutableList.builder();
    int seenKeys = 0;
    for (Object o : flattenedOutputFileMap) {
      if (o instanceof String) {
        if (++seenKeys > numImplicitOutputKeys) {
          break;
        }
      } else {
        result.add((OutputFile) o);
      }
    }
    return result.build();
  }

  /**
   * Constructs and returns an immutable multimap of the explicit outputs, from attribute name to
   * associated value.
   *
   * <p>Keys are listed in the same order as attributes. Order of attribute values (outputs in an
   * output list) is preserved.
   *
   * <p>Since this is a multimap, attributes that have no associated outputs are omitted from the
   * result.
   */
  public ImmutableListMultimap<String, OutputFile> getExplicitOutputFileMap() {
    ImmutableListMultimap.Builder<String, OutputFile> result = ImmutableListMultimap.builder();
    int seenKeys = 0;
    String key = null;
    for (Object o : flattenedOutputFileMap) {
      if (o instanceof String) {
        seenKeys++;
        key = (String) o;
      } else if (seenKeys > numImplicitOutputKeys) {
        result.put(key, (OutputFile) o);
      }
    }
    return result.build();
  }

  /**
   * Returns a map of the Starlark-defined implicit outputs, from dict key to output file.
   *
   * <p>If there is no implicit outputs function, or it is a native one, an empty map is returned.
   *
   * <p>This is not a multimap because Starlark-defined implicit output functions return exactly one
   * output per key.
   */
  public ImmutableMap<String, OutputFile> getStarlarkImplicitOutputFileMap() {
    if (!(implicitOutputsFunction instanceof StarlarkImplicitOutputsFunction)) {
      return ImmutableMap.of();
    }
    ImmutableMap.Builder<String, OutputFile> result = ImmutableMap.builder();
    int seenKeys = 0;
    String key = null;
    for (Object o : flattenedOutputFileMap) {
      if (o instanceof String) {
        if (++seenKeys > numImplicitOutputKeys) {
          break;
        }
        key = (String) o;
      } else {
        result.put(key, (OutputFile) o);
      }
    }
    return result.build();
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

  /**
   * Returns the value of the attribute with the given index. Returns null, if no such attribute
   * exists OR no value was set.
   */
  @Nullable
  private Object getAttrWithIndex(int attrIndex) {
    Object value = attributes.getAttributeValue(attrIndex);
    if (value != null) {
      return value;
    }
    Attribute attr = ruleClass.getAttribute(attrIndex);
    if (attr.hasComputedDefault()) {
      // Attributes with computed defaults are explicitly populated during rule creation.
      // However, computing those defaults could trigger reads of other attributes
      // which have not yet been populated. In such a case control comes here, and we return null.
      // NOTE: In this situation returning null does not result in a correctness issue, since
      // the value for the attribute is actually a function to compute the value.
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
    Integer attrIndex = ruleClass.getAttributeIndex(attrName);
    return attrIndex != null && attributes.isAttributeValueExplicitlySpecified(attrIndex);
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

  void freeze() {
    attributes = attributes.freeze();
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
   * Collects the output files (both implicit and explicit). Must be called before the output
   * accessors methods can be used, and must be called only once.
   */
  void populateOutputFiles(EventHandler eventHandler, Package.Builder pkgBuilder)
      throws LabelSyntaxException, InterruptedException {
    populateOutputFilesInternal(
        eventHandler, pkgBuilder.getPackageIdentifier(), /*checkLabels=*/ true);
  }

  void populateOutputFilesUnchecked(EventHandler eventHandler, Package.Builder pkgBuilder)
      throws InterruptedException {
    try {
      populateOutputFilesInternal(
          eventHandler, pkgBuilder.getPackageIdentifier(), /*checkLabels=*/ false);
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  @FunctionalInterface
  private interface ExplicitOutputHandler {
    public void accept(Attribute attribute, Label outputLabel) throws LabelSyntaxException;
  }

  @FunctionalInterface
  private interface ImplicitOutputHandler {
    public void accept(String outputKey, String outputName);
  }

  private void populateOutputFilesInternal(
      EventHandler eventHandler, PackageIdentifier pkgId, boolean checkLabels)
      throws LabelSyntaxException, InterruptedException {
    Preconditions.checkState(flattenedOutputFileMap == null);

    // We associate each output with its String key (or empty string if there's no key) as we go,
    // and compress it down to a flat list afterwards. We use ImmutableListMultimap because it's
    // more efficient than LinkedListMultimap and provides ordering guarantees among keys (whereas
    // ArrayListMultimap doesn't).
    ImmutableListMultimap.Builder<String, OutputFile> outputFileMap =
        ImmutableListMultimap.builder();
    // Detects collisions where the same output key is used for both an explicit and implicit entry.
    HashSet<String> implicitOutputKeys = new HashSet<>();

    // We need the implicits to appear before the explicits in the final data structure, so we
    // process them first. We check for duplicates while handling the explicits.
    //
    // Each of these cases has two subcases, so we factor their bodies out into lambdas.

    ImplicitOutputHandler implicitOutputHandler =
        // outputKey: associated dict key if Starlark-defined, empty string otherwise
        // outputName: package-relative path fragment
        (outputKey, outputName) -> {
          Label label;
          if (checkLabels) { // controls label syntax validation only
            try {
              label = Label.create(pkgId, outputName);
            } catch (LabelSyntaxException e) {
              reportError(
                  String.format(
                      "illegal output file name '%s' in rule %s due to: %s",
                      outputName, getLabel(), e.getMessage()),
                  eventHandler);
              return;
            }
          } else {
            label = Label.createUnvalidated(pkgId, outputName);
          }
          validateOutputLabel(label, eventHandler);

          OutputFile file = new OutputFile(label, this);
          outputFileMap.put(outputKey, file);
          implicitOutputKeys.add(outputKey);
        };

    // Populate the implicit outputs.
    try {
      RawAttributeMapper attributeMap = RawAttributeMapper.of(this);
      // TODO(bazel-team): Reconsider the ImplicitOutputsFunction abstraction. It doesn't seem to be
      // a good fit if it forces us to downcast in situations like this. It also causes
      // getImplicitOutputs() to declare that it throws EvalException (which then has to be
      // explicitly disclaimed by the subclass SafeImplicitOutputsFunction).
      if (implicitOutputsFunction instanceof StarlarkImplicitOutputsFunction) {
        for (Map.Entry<String, String> e :
            ((StarlarkImplicitOutputsFunction) implicitOutputsFunction)
                .calculateOutputs(eventHandler, attributeMap)
                .entrySet()) {
          implicitOutputHandler.accept(e.getKey(), e.getValue());
        }
      } else {
        for (String out : implicitOutputsFunction.getImplicitOutputs(eventHandler, attributeMap)) {
          implicitOutputHandler.accept(/*outputKey=*/ "", out);
        }
      }
    } catch (EvalException e) {
      reportError(
          String.format("In rule %s: %s", getLabel(), e.getMessageWithStack()), eventHandler);
    }

    ExplicitOutputHandler explicitOutputHandler =
        (attribute, outputLabel) -> {
          String attrName = attribute.getName();
          if (implicitOutputKeys.contains(attrName)) {
            reportError(
                String.format(
                    "Implicit output key '%s' collides with output attribute name", attrName),
                eventHandler);
          }
          if (checkLabels) {
            if (!outputLabel.getPackageIdentifier().equals(pkg.getPackageIdentifier())) {
              throw new IllegalStateException(
                  String.format(
                      "Label for attribute %s should refer to '%s' but instead refers to '%s'"
                          + " (label '%s')",
                      attribute,
                      pkg.getName(),
                      outputLabel.getPackageFragment(),
                      outputLabel.getName()));
            }
            if (outputLabel.getName().equals(".")) {
              throw new LabelSyntaxException("output file name can't be equal '.'");
            }
          }
          validateOutputLabel(outputLabel, eventHandler);

          OutputFile outputFile = new OutputFile(outputLabel, this);
          outputFileMap.put(attrName, outputFile);
        };

    // Populate the explicit outputs.
    NonconfigurableAttributeMapper nonConfigurableAttributes =
        NonconfigurableAttributeMapper.of(this);
    for (Attribute attribute : ruleClass.getAttributes()) {
      String name = attribute.getName();
      Type<?> type = attribute.getType();
      if (type == BuildType.OUTPUT) {
        Label label = nonConfigurableAttributes.get(name, BuildType.OUTPUT);
        if (label != null) {
          explicitOutputHandler.accept(attribute, label);
        }
      } else if (type == BuildType.OUTPUT_LIST) {
        for (Label label : nonConfigurableAttributes.get(name, BuildType.OUTPUT_LIST)) {
          explicitOutputHandler.accept(attribute, label);
        }
      }
    }

    // Flatten the result into the final list.
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    for (Map.Entry<String, Collection<OutputFile>> e : outputFileMap.build().asMap().entrySet()) {
      builder.add(e.getKey());
      for (OutputFile out : e.getValue()) {
        builder.add(out);
      }
    }
    flattenedOutputFileMap = builder.build();
    numImplicitOutputKeys = implicitOutputKeys.size();
  }

  private void validateOutputLabel(Label label, EventHandler eventHandler) {
    if (label.getName().equals(getName())) {
      // TODO(bazel-team): for now (23 Apr 2008) this is just a warning.  After
      // June 1st we should make it an error.
      reportWarning("target '" + getName() + "' is both a rule and a file; please choose "
                    + "another name for the rule", eventHandler);
    }
  }

  void reportError(String message, EventHandler eventHandler) {
    eventHandler.handle(Event.error(location, message));
    this.containsErrors = true;
  }

  private void reportWarning(String message, EventHandler eventHandler) {
    eventHandler.handle(Event.warn(location, message));
  }

  /** Returns a string of the form "cc_binary rule //foo:foo" */
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
    // Temporary logic to relax config_setting's visibility enforcement while depot migrations set
    // visibility settings properly (legacy code may have visibility settings that would break if
    // enforced). See https://github.com/bazelbuild/bazel/issues/12669. Ultimately this entire
    // conditional should be removed.
    if (ruleClass.getName().equals("config_setting")) {
      ConfigSettingVisibilityPolicy policy = getPackage().getConfigSettingVisibilityPolicy();
      if (visibility != null) {
        return visibility; // Use explicitly set visibility
      } else if (policy == ConfigSettingVisibilityPolicy.DEFAULT_PUBLIC) {
        return ConstantRuleVisibility.PUBLIC; // Default: //visibility:public.
      } else {
        return pkg.getDefaultVisibility(); // Default: same as all other rules.
      }
    }

    // All other rules.
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
    return getLabel().getPackageIdentifier().getRepository();
  }

  /** Returns the suffix of target kind for all rules. */
  public static String targetKindSuffix() {
    return " rule";
  }
}
