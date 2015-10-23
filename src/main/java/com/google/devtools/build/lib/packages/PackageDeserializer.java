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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.common.collect.Sets;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.License.LicenseParsingException;
import com.google.devtools.build.lib.packages.Package.Builder.GeneratedLabelConflict;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.StringDictUnaryEntry;
import com.google.devtools.build.lib.syntax.GlobCriteria;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * Functionality to deserialize loaded packages.
 */
public class PackageDeserializer {

  private static final Logger LOG = Logger.getLogger(PackageDeserializer.class.getName());

  /**
   * Provides the deserializer with tools it needs to build a package from its serialized form.
   */
  public interface PackageDeserializationEnvironment {

    /** Converts the serialized package's path string into a {@link Path} object. */
    Path getPath(String buildFilePath);

    /** Returns a {@link RuleClass} object for the serialized rule. */
    RuleClass getRuleClass(Build.Rule rulePb, Location ruleLocation);

    /** Description of what rule attributes of each rule should be deserialized. */
    AttributesToDeserialize attributesToDeserialize();
  }

  /**
   * A class that defines what attributes to keep after deserialization. Note that all attributes of
   * type label are kept in order to navigate between dependencies.
   *
   * <p>If {@code addSyntheticAttributeHash} is {@code true}, a synthetic attribute is added to each
   * Rule that contains a stable hash of the entire serialized rule for the sake of permitting
   * equality comparisons that respect the attributes that were dropped according to {@code
   * attributesToKeep}.
   */
  public static class AttributesToDeserialize {

    private final boolean addSyntheticAttributeHash;
    private final Predicate<String> shouldKeepAttributeWithName;

    public AttributesToDeserialize(boolean addSyntheticAttributeHash,
        Predicate<String> shouldKeepAttributeWithName) {
      this.addSyntheticAttributeHash = addSyntheticAttributeHash;
      this.shouldKeepAttributeWithName = shouldKeepAttributeWithName;
    }

    public boolean includeAttribute(String attr) { return shouldKeepAttributeWithName.apply(attr); }
  }

  public static final AttributesToDeserialize DESERIALIZE_ALL_ATTRS =
      new AttributesToDeserialize(false, Predicates.<String>alwaysTrue());

  // Workaround for Java serialization making it tough to pass in a deserialization environment
  // manually.
  // volatile is needed to ensure that the objects are published safely.
  // TODO(bazel-team): Subclass ObjectOutputStream to pass this through instead.
  public static volatile PackageDeserializationEnvironment defaultPackageDeserializationEnvironment;

  // Cache label deserialization across all instances- PackgeDeserializers need to be created on
  // demand due to initialiation constraints wrt the setting of static members.
  private static final Interner<Label> LABEL_INTERNER = Interners.newWeakInterner();

  /** Class encapsulating state for a single package deserialization. */
  private static class DeserializationContext {
    private final Package.Builder packageBuilder;

    private DeserializationContext(Package.Builder packageBuilder) {
      this.packageBuilder = packageBuilder;
    }
  }

  private final PackageDeserializationEnvironment packageDeserializationEnvironment;

  /**
   * Creates a {@link PackageDeserializer} using {@link #defaultPackageDeserializationEnvironment}.
   */
  public PackageDeserializer() {
    this.packageDeserializationEnvironment = defaultPackageDeserializationEnvironment;
  }

  public PackageDeserializer(PackageDeserializationEnvironment packageDeserializationEnvironment) {
    this.packageDeserializationEnvironment =
        Preconditions.checkNotNull(packageDeserializationEnvironment);
  }

  private static ParsedAttributeValue deserializeAttribute(
      Type<?> expectedType, Build.Attribute attrPb) throws PackageDeserializationException {
    Object value = deserializeAttributeValue(expectedType, attrPb);
    return new ParsedAttributeValue(
        attrPb.hasExplicitlySpecified() && attrPb.getExplicitlySpecified(), value
    );
  }

  private static void deserializeInputFile(DeserializationContext context,
      Build.SourceFile sourceFile)
      throws PackageDeserializationException {
    InputFile inputFile;
    try {
      inputFile = context.packageBuilder.createInputFile(
          deserializeLabel(sourceFile.getName()).getName(), EmptyLocation.INSTANCE);
    } catch (GeneratedLabelConflict e) {
      throw new PackageDeserializationException(e);
    }

    if (!sourceFile.getVisibilityLabelList().isEmpty() || sourceFile.hasLicense()) {
      context.packageBuilder.setVisibilityAndLicense(inputFile,
          PackageFactory.getVisibility(deserializeLabels(sourceFile.getVisibilityLabelList())),
          deserializeLicense(sourceFile.getLicense()));
    }
  }

  private static void deserializePackageGroup(DeserializationContext context,
      Build.PackageGroup packageGroupPb) throws PackageDeserializationException {
    List<String> specifications = new ArrayList<>();
    for (String containedPackage : packageGroupPb.getContainedPackageList()) {
      specifications.add("//" + containedPackage);
    }

    try {
      context.packageBuilder.addPackageGroup(
          deserializeLabel(packageGroupPb.getName()).getName(),
          specifications,
          deserializeLabels(packageGroupPb.getIncludedPackageGroupList()),
          NullEventHandler.INSTANCE,  // TODO(bazel-team): Handle errors properly
          EmptyLocation.INSTANCE);
    } catch (LabelSyntaxException | Package.NameConflictException e) {
      throw new PackageDeserializationException(e);
    }
  }

  private void deserializeRule(DeserializationContext context, Build.Rule rulePb)
      throws PackageDeserializationException, InterruptedException {
    Location ruleLocation = EmptyLocation.INSTANCE;
    RuleClass ruleClass = packageDeserializationEnvironment.getRuleClass(rulePb, ruleLocation);
    Map<String, ParsedAttributeValue> attributeValues = new HashMap<>();
    AttributesToDeserialize attrToDeserialize =
        packageDeserializationEnvironment.attributesToDeserialize();

    Hasher hasher = Hashing.md5().newHasher();
    for (Build.Attribute attrPb : rulePb.getAttributeList()) {
      Type<?> type = ruleClass.getAttributeByName(attrPb.getName()).getType();
      attributeValues.put(attrPb.getName(), deserializeAttribute(type, attrPb));
      if (attrToDeserialize.addSyntheticAttributeHash) {
        // TODO(bazel-team): This might give false positives because of explicit vs implicit.
        hasher.putBytes(attrPb.toByteArray());
      }
    }
    AttributeContainerWithoutLocation attributeContainer =
        new AttributeContainerWithoutLocation(ruleClass, hasher.hash().asBytes());

    Label ruleLabel = deserializeLabel(rulePb.getName());
    try {
      Rule rule = createRuleWithParsedAttributeValues(ruleClass,
          ruleLabel, context.packageBuilder, ruleLocation, attributeValues,
          NullEventHandler.INSTANCE, attributeContainer);
      context.packageBuilder.addRule(rule);

      // Remove the attribute after it is added to package in order to pass the validations
      // and be able to compute all the outputs.
      if (attrToDeserialize != DESERIALIZE_ALL_ATTRS) {
        for (String attrName : attributeValues.keySet()) {
          Attribute attribute = ruleClass.getAttributeByName(attrName);
          if (!(attrToDeserialize.shouldKeepAttributeWithName.apply(attrName)
              || BuildType.isLabelType(attribute.getType()))) {
            attributeContainer.clearIfNotLabel(attrName);
          }
        }
      }

      Preconditions.checkState(!rule.containsErrors());
    } catch (NameConflictException | LabelSyntaxException e) {
      throw new PackageDeserializationException(e);
    }

  }

  /** "Empty" location implementation, all methods should return non-null, but empty, values. */
  private static class EmptyLocation extends Location {
    private static final EmptyLocation INSTANCE = new EmptyLocation();

    private static final PathFragment DEV_NULL = new PathFragment("/dev/null");
    private static final LineAndColumn EMPTY_LINE_AND_COLUMN = new LineAndColumn(0, 0);

    private EmptyLocation() {
      super(0, 0);
    }

    @Override
    public PathFragment getPath() {
      return DEV_NULL;
    }

    @Override
    public LineAndColumn getStartLineAndColumn() {
      return EMPTY_LINE_AND_COLUMN;
    }

    @Override
    public LineAndColumn getEndLineAndColumn() {
      return EMPTY_LINE_AND_COLUMN;
    }

    @Override
    public int hashCode() {
      return 42;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof EmptyLocation;
    }
  }

  /**
   * Exception thrown when something goes wrong during package deserialization.
   */
  public static class PackageDeserializationException extends Exception {
    private PackageDeserializationException(String message) {
      super(message);
    }

    private PackageDeserializationException(String message, Exception reason) {
      super(message, reason);
    }

    private PackageDeserializationException(Exception reason) {
      super(reason);
    }
  }

  private static Label deserializeLabel(String labelName) throws PackageDeserializationException {
    try {
      return LABEL_INTERNER.intern(Label.parseAbsolute(labelName));
    } catch (LabelSyntaxException e) {
      throw new PackageDeserializationException(
          "Invalid label '" + labelName + "':" + e.getMessage(), e);
    }
  }

  private static List<Label> deserializeLabels(List<String> labelNames)
      throws PackageDeserializationException {
    ImmutableList.Builder<Label> result = ImmutableList.builder();
    for (String labelName : labelNames) {
      result.add(deserializeLabel(labelName));
    }

    return result.build();
  }

  private static License deserializeLicense(Build.License licensePb)
      throws PackageDeserializationException {
    List<String> licenseStrings = new ArrayList<>();
    licenseStrings.addAll(licensePb.getLicenseTypeList());
    for (String exception : licensePb.getExceptionList()) {
      licenseStrings.add("exception=" + exception);
    }

    try {
      return License.parseLicense(licenseStrings);
    } catch (LicenseParsingException e) {
      throw new PackageDeserializationException(e);
    }
  }

  private static Set<DistributionType> deserializeDistribs(List<String> distributions)
      throws PackageDeserializationException {
    try {
      return License.parseDistributions(distributions);
    } catch (LicenseParsingException e) {
      throw new PackageDeserializationException(e);
    }
  }

  private static TriState deserializeTriStateValue(String value)
      throws PackageDeserializationException {
    if (value.equals("yes")) {
      return TriState.YES;
    } else if (value.equals("no")) {
      return TriState.NO;
    } else if (value.equals("auto")) {
      return TriState.AUTO;
    } else {
      throw new PackageDeserializationException(
          String.format("Invalid tristate value: '%s'", value));
    }
  }

  private static List<FilesetEntry> deserializeFilesetEntries(
      List<Build.FilesetEntry> filesetPbs)
      throws PackageDeserializationException {
    ImmutableList.Builder<FilesetEntry> result = ImmutableList.builder();
    for (Build.FilesetEntry filesetPb : filesetPbs) {
      Label srcLabel = deserializeLabel(filesetPb.getSource());
      List<Label> files =
          filesetPb.getFilesPresent() ? deserializeLabels(filesetPb.getFileList()) : null;
      List<String> excludes =
          filesetPb.getExcludeList().isEmpty()
              ? null : ImmutableList.copyOf(filesetPb.getExcludeList());
      String destDir = filesetPb.getDestinationDirectory();
      FilesetEntry.SymlinkBehavior symlinkBehavior =
          pbToSymlinkBehavior(filesetPb.getSymlinkBehavior());
      String stripPrefix = filesetPb.hasStripPrefix() ? filesetPb.getStripPrefix() : null;

      result.add(
          new FilesetEntry(srcLabel, files, excludes, destDir, symlinkBehavior, stripPrefix));
    }

    return result.build();
  }

  /**
   * Deserialize a package from its representation as a protocol message. The inverse of
   * {@link PackageSerializer#serialize}.
   * @throws IOException
   * @throws InterruptedException
   */
  private void deserializeInternal(Build.Package packagePb, StoredEventHandler eventHandler,
      Package.Builder builder, InputStream in)
      throws PackageDeserializationException, IOException, InterruptedException {
    Path buildFile = packageDeserializationEnvironment.getPath(packagePb.getBuildFilePath());
    Preconditions.checkNotNull(buildFile);
    DeserializationContext context = new DeserializationContext(builder);
    builder.setFilename(buildFile);

    if (packagePb.hasDefaultVisibilitySet() && packagePb.getDefaultVisibilitySet()) {
      builder.setDefaultVisibility(
          PackageFactory.getVisibility(
              deserializeLabels(packagePb.getDefaultVisibilityLabelList())));
    }

    // It's important to do this after setting the default visibility, since that implicitly sets
    // this bit to true
    builder.setDefaultVisibilitySet(packagePb.getDefaultVisibilitySet());
    if (packagePb.hasDefaultTestonly()) {
      builder.setDefaultTestonly(packagePb.getDefaultTestonly());
    }
    if (packagePb.hasDefaultDeprecation()) {
      builder.setDefaultDeprecation(packagePb.getDefaultDeprecation());
    }

    builder.setDefaultCopts(packagePb.getDefaultCoptList());
    if (packagePb.hasDefaultHdrsCheck()) {
      builder.setDefaultHdrsCheck(packagePb.getDefaultHdrsCheck());
    }
    if (packagePb.hasDefaultLicense()) {
      builder.setDefaultLicense(deserializeLicense(packagePb.getDefaultLicense()));
    }
    builder.setDefaultDistribs(deserializeDistribs(packagePb.getDefaultDistribList()));

    for (String subinclude : packagePb.getSubincludeLabelList()) {
      Label label = deserializeLabel(subinclude);
      builder.addSubinclude(label, null);
    }

    ImmutableList.Builder<Label> skylarkFileDependencies = ImmutableList.builder();
    for (String skylarkFile : packagePb.getSkylarkLabelList()) {
      skylarkFileDependencies.add(deserializeLabel(skylarkFile));
    }
    builder.setSkylarkFileDependencies(skylarkFileDependencies.build());

    MakeEnvironment.Builder makeEnvBuilder = new MakeEnvironment.Builder();
    for (Build.MakeVar makeVar : packagePb.getMakeVariableList()) {
      for (Build.MakeVarBinding binding : makeVar.getBindingList()) {
        makeEnvBuilder.update(
            makeVar.getName(), binding.getValue(), binding.getPlatformSetRegexp());
      }
    }
    builder.setMakeEnv(makeEnvBuilder);

    for (Build.Event event : packagePb.getEventList()) {
      deserializeEvent(eventHandler, event);
    }

    if (packagePb.hasContainsErrors() && packagePb.getContainsErrors()) {
      builder.setContainsErrors();
    }

    builder.setWorkspaceName(packagePb.getWorkspaceName());

    deserializeTargets(in, context);
  }

  private void deserializeTargets(InputStream in, DeserializationContext context)
      throws IOException, PackageDeserializationException, InterruptedException {
    Build.TargetOrTerminator tot;
    while (!(tot = Build.TargetOrTerminator.parseDelimitedFrom(in)).getIsTerminator()) {
      Build.Target target = tot.getTarget();
      switch (target.getType()) {
        case SOURCE_FILE:
          deserializeInputFile(context, target.getSourceFile());
          break;
        case PACKAGE_GROUP:
          deserializePackageGroup(context, target.getPackageGroup());
          break;
        case RULE:
          deserializeRule(context, target.getRule());
          break;
        default:
          throw new IllegalStateException("Unexpected Target type: " + target.getType());
      }
    }
  }

  /**
   * Deserializes a {@link Package} from {@code in}. The inverse of
   * {@link PackageSerializer#serialize}.
   *
   * <p>Expects {@code in} to contain a single
   * {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.Package} message followed
   * by a series of
   * {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.TargetOrTerminator}
   * messages encoding the associated targets.
   *
   * @param in stream to read from
   * @return a new {@link Package} as read from {@code in}
   * @throws PackageDeserializationException on failures deserializing the input
   * @throws IOException on failures reading from {@code in}
   * @throws InterruptedException
   */
  public Package deserialize(InputStream in)
      throws PackageDeserializationException, IOException, InterruptedException {
    try {
      return deserializeInternal(in);
    } catch (PackageDeserializationException | RuntimeException e) {
      LOG.log(Level.WARNING, "Failed to deserialize Package object", e);
      throw e;
    }
  }

  private Package deserializeInternal(InputStream in)
      throws PackageDeserializationException, IOException, InterruptedException {
    // Read the initial Package message so we have the data to initialize the builder. We will read
    // the Targets in individually later.
    Build.Package packagePb = Build.Package.parseDelimitedFrom(in);
    Package.Builder builder;
    try {
      builder = new Package.Builder(
          PackageIdentifier
              .create(packagePb.getRepository(), new PathFragment(packagePb.getName())),
          null);
    } catch (LabelSyntaxException e) {
      throw new PackageDeserializationException(e);
    }
    StoredEventHandler eventHandler = new StoredEventHandler();
    deserializeInternal(packagePb, eventHandler, builder, in);
    builder.addEvents(eventHandler.getEvents());
    return builder.build();
  }

  private static void deserializeEvent(StoredEventHandler eventHandler, Build.Event event) {
    String message = event.getMessage();
    switch (event.getKind()) {
      case ERROR: eventHandler.handle(Event.error(message)); break;
      case WARNING: eventHandler.handle(Event.warn(message)); break;
      case INFO: eventHandler.handle(Event.info(message)); break;
      case PROGRESS: eventHandler.handle(Event.progress(message)); break;
      default: break;  // Ignore
    }
  }

  private static List<?> deserializeGlobs(List<?> matches,
      Build.Attribute attrPb) {
    if (attrPb.getGlobCriteriaCount() == 0) {
      return matches;
    }

    Builder<GlobCriteria> criteriaBuilder = ImmutableList.builder();
    for (Build.GlobCriteria criteriaPb : attrPb.getGlobCriteriaList()) {
      if (criteriaPb.hasGlob() && criteriaPb.getGlob()) {
        criteriaBuilder.add(GlobCriteria.fromGlobCall(
            ImmutableList.copyOf(criteriaPb.getIncludeList()),
            ImmutableList.copyOf(criteriaPb.getExcludeList())));
      } else {
        criteriaBuilder.add(
            GlobCriteria.fromList(ImmutableList.copyOf(criteriaPb.getIncludeList())));
      }
    }

    @SuppressWarnings({"unchecked", "rawtypes"}) GlobList<?> result =
        new GlobList(criteriaBuilder.build(), matches);
    return result;
  }

  // TODO(bazel-team): Verify that these put sane values in the attribute
  @VisibleForTesting
  static Object deserializeAttributeValue(Type<?> expectedType,
      Build.Attribute attrPb)
      throws PackageDeserializationException {
    switch (attrPb.getType()) {
      case INTEGER:
        return attrPb.hasIntValue() ? new Integer(attrPb.getIntValue()) : null;

      case STRING:
        if (!attrPb.hasStringValue()) {
          return null;
        } else if (expectedType == BuildType.NODEP_LABEL) {
          return deserializeLabel(attrPb.getStringValue());
        } else {
          return attrPb.getStringValue();
        }

      case LABEL:
      case OUTPUT:
        return attrPb.hasStringValue() ? deserializeLabel(attrPb.getStringValue()) : null;

      case STRING_LIST:
        if (expectedType == BuildType.NODEP_LABEL_LIST) {
          return deserializeGlobs(deserializeLabels(attrPb.getStringListValueList()), attrPb);
        } else {
          return deserializeGlobs(ImmutableList.copyOf(attrPb.getStringListValueList()), attrPb);
        }

      case LABEL_LIST:
      case OUTPUT_LIST:
        return deserializeGlobs(deserializeLabels(attrPb.getStringListValueList()), attrPb);

      case DISTRIBUTION_SET:
        return deserializeDistribs(attrPb.getStringListValueList());

      case LICENSE:
        return attrPb.hasLicense() ? deserializeLicense(attrPb.getLicense()) : null;

      case STRING_DICT: {
        // Building an immutable map will fail if the builder was given duplicate keys. These entry
        // lists may contain duplicate keys if the serialized map value was configured (e.g. via
        // the select function) and the different configuration values had keys in common. This is
        // because serialization flattens configurable map-valued attributes.
        //
        // As long as serialization does this flattening, to avoid failure during deserialization,
        // we dedupe entries in the list by their keys.
        // TODO(bazel-team): Serialize and deserialize configured values with fidelity (without
        // flattening them).
        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
        HashSet<String> keysSeenSoFar = Sets.newHashSet();
        for (Build.StringDictEntry entry : attrPb.getStringDictValueList()) {
          String key = entry.getKey();
          if (keysSeenSoFar.add(key)) {
            builder.put(key, entry.getValue());
          }
        }
        return builder.build();
      }

      case STRING_DICT_UNARY: {
        // See STRING_DICT case's comment about why this dedupes entries by their keys.
        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
        HashSet<String> keysSeenSoFar = Sets.newHashSet();
        for (StringDictUnaryEntry entry : attrPb.getStringDictUnaryValueList()) {
          String key = entry.getKey();
          if (keysSeenSoFar.add(key)) {
            builder.put(key, entry.getValue());
          }
        }
        return builder.build();
      }

      case FILESET_ENTRY_LIST:
        return deserializeFilesetEntries(attrPb.getFilesetListValueList());

      case LABEL_LIST_DICT: {
        // See STRING_DICT case's comment about why this dedupes entries by their keys.
        ImmutableMap.Builder<String, List<Label>> builder = ImmutableMap.builder();
        HashSet<String> keysSeenSoFar = Sets.newHashSet();
        for (Build.LabelListDictEntry entry : attrPb.getLabelListDictValueList()) {
          String key = entry.getKey();
          if (keysSeenSoFar.add(key)) {
            builder.put(key, deserializeLabels(entry.getValueList()));
          }
        }
        return builder.build();
      }

      case STRING_LIST_DICT: {
        // See STRING_DICT case's comment about why this dedupes entries by their keys.
        ImmutableMap.Builder<String, List<String>> builder = ImmutableMap.builder();
        HashSet<String> keysSeenSoFar = Sets.newHashSet();
        for (Build.StringListDictEntry entry : attrPb.getStringListDictValueList()) {
          String key = entry.getKey();
          if (keysSeenSoFar.add(key)) {
            builder.put(key, ImmutableList.copyOf(entry.getValueList()));
          }
        }
        return builder.build();
      }

      case BOOLEAN:
        return attrPb.hasBooleanValue() ? attrPb.getBooleanValue() : null;

      case TRISTATE:
        return attrPb.hasStringValue() ? deserializeTriStateValue(attrPb.getStringValue()) : null;

      case INTEGER_LIST:
        return ImmutableList.copyOf(attrPb.getIntListValueList());

      default:
          throw new PackageDeserializationException("Invalid discriminator: " + attrPb.getType());
    }
  }

  private static FilesetEntry.SymlinkBehavior pbToSymlinkBehavior(
      Build.FilesetEntry.SymlinkBehavior symlinkBehavior) {
    switch (symlinkBehavior) {
      case COPY:
        return FilesetEntry.SymlinkBehavior.COPY;
      case DEREFERENCE:
        return FilesetEntry.SymlinkBehavior.DEREFERENCE;
      default:
        throw new IllegalStateException();
    }
  }

  /**
   * An special {@code AttributeContainer} implementation that does not keep
   * the location and can contain a hashcode of the target attributes.
   */
  public static class AttributeContainerWithoutLocation extends AttributeContainer {

    @Nullable
    private final byte[] syntheticAttrHash;

    private AttributeContainerWithoutLocation(RuleClass ruleClass,
        @Nullable byte[] syntheticAttrHash) {
      super(ruleClass, null);
      this.syntheticAttrHash = syntheticAttrHash;
    }

    @Override
    public Location getAttributeLocation(String attrName) {
      return EmptyLocation.INSTANCE;
    }

    @Override
    void setAttributeLocation(int attrIndex, Location location) {
      throw new UnsupportedOperationException("Setting location not supported");
    }

    @Override
    void setAttributeLocation(Attribute attribute, Location location) {
      throw new UnsupportedOperationException("Setting location not supported");
    }

    @Nullable
    public byte[] getSyntheticAttrHash() {
      return syntheticAttrHash;
    }

    private void clearIfNotLabel(String attr) {
      setAttributeValueByName(attr, null);
    }
  }

  /**
   * Creates a rule with the attribute values that are already parsed.
   *
   * <p><b>WARNING:</b> This assumes that the attribute values here have the right type and
   * bypasses some sanity checks. If they are of the wrong type, everything will come down burning.
   */
  @SuppressWarnings("unchecked")
  private static Rule createRuleWithParsedAttributeValues(RuleClass ruleClass, Label label,
      Package.Builder pkgBuilder, Location ruleLocation,
      Map<String, ParsedAttributeValue> attributeValues, EventHandler eventHandler,
      AttributeContainer attributeContainer)
      throws LabelSyntaxException, InterruptedException {
    Rule rule = pkgBuilder.newRuleWithLabelAndAttrContainer(label, ruleClass, ruleLocation,
        attributeContainer);
    rule.checkValidityPredicate(eventHandler);

    for (Attribute attribute : rule.getRuleClassObject().getAttributes()) {
      ParsedAttributeValue value = attributeValues.get(attribute.getName());
      if (attribute.isMandatory()) {
        Preconditions.checkState(value != null);
      }

      if (value == null) {
        continue;
      }

      rule.setAttributeValue(attribute, value.value, value.explicitlySpecified);
      ruleClass.checkAllowedValues(rule, attribute, eventHandler);

      if (attribute.getName().equals("visibility")) {
        // TODO(bazel-team): Verify that this cast works
        rule.setVisibility(PackageFactory.getVisibility((List<Label>) value.value));
      }
    }

    rule.populateOutputFiles(eventHandler, pkgBuilder);
    Preconditions.checkState(!rule.containsErrors());
    return rule;
  }

  private static class ParsedAttributeValue {
    private final boolean explicitlySpecified;
    private final Object value;

    private ParsedAttributeValue(boolean explicitlySpecified, Object value) {
      this.explicitlySpecified = explicitlySpecified;
      this.value = value;
    }
  }
}
