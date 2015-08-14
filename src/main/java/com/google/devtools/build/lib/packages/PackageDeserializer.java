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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.License.LicenseParsingException;
import com.google.devtools.build.lib.packages.Package.Builder.GeneratedLabelConflict;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.RuleClass.ParsedAttributeValue;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.StringDictUnaryEntry;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.GlobCriteria;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

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
  }

  // Workaround for Java serialization making it tough to pass in a deserialization environment
  // manually.
  // volatile is needed to ensure that the objects are published safely.
  // TODO(bazel-team): Subclass ObjectOutputStream to pass this through instead.
  public static volatile PackageDeserializationEnvironment defaultPackageDeserializationEnvironment;

  /** Class encapsulating state for a single package deserialization. */
  private static class DeserializationContext {
    private final Package.Builder packageBuilder;
    private final PathFragment buildFilePath;

    public DeserializationContext(Path buildFilePath, Package.Builder packageBuilder) {
      this.buildFilePath = buildFilePath.asFragment();
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

  private static Location deserializeLocation(DeserializationContext context,
      Build.Location location) {
    return new ExplicitLocation(context.buildFilePath, location);
  }

  private static ParsedAttributeValue deserializeAttribute(DeserializationContext context,
      Type<?> expectedType, Build.Attribute attrPb) throws PackageDeserializationException {
    Object value = deserializeAttributeValue(expectedType, attrPb);
    return new ParsedAttributeValue(
        attrPb.hasExplicitlySpecified() && attrPb.getExplicitlySpecified(), value,
        deserializeLocation(context, attrPb.getParseableLocation()));
  }

  private void deserializeInputFile(DeserializationContext context, Build.SourceFile sourceFile)
      throws PackageDeserializationException {
    InputFile inputFile;
    try {
      inputFile = context.packageBuilder.createInputFile(
          deserializeLabel(sourceFile.getName()).getName(),
          deserializeLocation(context, sourceFile.getParseableLocation()));
    } catch (GeneratedLabelConflict e) {
      throw new PackageDeserializationException(e);
    }

    if (!sourceFile.getVisibilityLabelList().isEmpty() || sourceFile.hasLicense()) {
      context.packageBuilder.setVisibilityAndLicense(inputFile,
          PackageFactory.getVisibility(deserializeLabels(sourceFile.getVisibilityLabelList())),
          deserializeLicense(sourceFile.getLicense()));
    }
  }

  private void deserializePackageGroup(DeserializationContext context,
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
          deserializeLocation(context, packageGroupPb.getParseableLocation()));
    } catch (Label.SyntaxException | Package.NameConflictException e) {
      throw new PackageDeserializationException(e);
    }
  }

  private void deserializeRule(DeserializationContext context, Build.Rule rulePb)
      throws PackageDeserializationException {
    Location ruleLocation = deserializeLocation(context, rulePb.getParseableLocation());
    RuleClass ruleClass = packageDeserializationEnvironment.getRuleClass(rulePb, ruleLocation);
    Map<String, ParsedAttributeValue> attributeValues = new HashMap<>();
    for (Build.Attribute attrPb : rulePb.getAttributeList()) {
      Type<?> type = ruleClass.getAttributeByName(attrPb.getName()).getType();
      attributeValues.put(attrPb.getName(), deserializeAttribute(context, type, attrPb));
    }

    Label ruleLabel = deserializeLabel(rulePb.getName());
    try {
      Rule rule = ruleClass.createRuleWithParsedAttributeValues(
          ruleLabel, context.packageBuilder, ruleLocation, attributeValues,
          NullEventHandler.INSTANCE);
      context.packageBuilder.addRule(rule);

      Preconditions.checkState(!rule.containsErrors());
    } catch (NameConflictException | SyntaxException e) {
      throw new PackageDeserializationException(e);
    }
  }

  @Immutable
  private static final class ExplicitLocation extends Location {
    private final PathFragment path;
    private final int startLine;
    private final int startColumn;
    private final int endLine;
    private final int endColumn;

    private ExplicitLocation(PathFragment path, Build.Location location) {
      super(
          location.hasStartOffset() && location.hasEndOffset() ? location.getStartOffset() : 0,
          location.hasStartOffset() && location.hasEndOffset() ? location.getEndOffset() : 0);
      this.path = path;
      if (location.hasStartLine() && location.hasStartColumn()
          && location.hasEndLine() && location.hasEndColumn()) {
        this.startLine = location.getStartLine();
        this.startColumn = location.getStartColumn();
        this.endLine = location.getEndLine();
        this.endColumn = location.getEndColumn();
      } else {
        this.startLine = 0;
        this.startColumn = 0;
        this.endLine = 0;
        this.endColumn = 0;
      }
    }

    @Override
    public PathFragment getPath() {
      return path;
    }

    @Override
    public LineAndColumn getStartLineAndColumn() {
      return new LineAndColumn(startLine, startColumn);
    }

    @Override
    public LineAndColumn getEndLineAndColumn() {
      return new LineAndColumn(endLine, endColumn);
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          path.hashCode(), startLine, startColumn, endLine, endColumn, internalHashCode());
    }

    @Override
    public boolean equals(Object other) {
      if (other == null || !other.getClass().equals(getClass())) {
        return false;
      }
      ExplicitLocation that = (ExplicitLocation) other;
      return this.startLine == that.startLine
          && this.startColumn == that.startColumn
          && this.endLine == that.endLine
          && this.endColumn == that.endColumn
          && internalEquals(that)
          && Objects.equals(this.path, that.path);
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
      return Label.parseAbsolute(labelName);
    } catch (Label.SyntaxException e) {
      throw new PackageDeserializationException("Invalid label: " + e.getMessage(), e);
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
   * {@link PackageSerializer#serializePackage}.
   * @throws IOException
   */
  private void deserializeInternal(Build.Package packagePb, StoredEventHandler eventHandler,
      Package.Builder builder, InputStream in) throws PackageDeserializationException, IOException {
    Path buildFile = packageDeserializationEnvironment.getPath(packagePb.getBuildFilePath());
    Preconditions.checkNotNull(buildFile);
    DeserializationContext context = new DeserializationContext(buildFile, builder);
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
      deserializeEvent(context, eventHandler, event);
    }

    if (packagePb.hasContainsErrors() && packagePb.getContainsErrors()) {
      builder.setContainsErrors();
    }
    if (packagePb.hasContainsTemporaryErrors() && packagePb.getContainsTemporaryErrors()) {
      builder.setContainsTemporaryErrors();
    }

    deserializeTargets(in, context);
  }

  private void deserializeTargets(InputStream in, DeserializationContext context)
      throws IOException, PackageDeserializationException {
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
   * {@link PackageSerializer#serializePackage}.
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
   */
  public Package deserialize(InputStream in) throws PackageDeserializationException, IOException {
    try {
      return deserializeInternal(in);
    } catch (PackageDeserializationException | RuntimeException e) {
      LOG.log(Level.WARNING, "Failed to deserialize Package object", e);
      throw e;
    }
  }

  private Package deserializeInternal(InputStream in)
      throws PackageDeserializationException, IOException {
    // Read the initial Package message so we have the data to initialize the builder. We will read
    // the Targets in individually later.
    Build.Package packagePb = Build.Package.parseDelimitedFrom(in);
    Package.Builder builder;
    try {
      builder = new Package.Builder(
          new PackageIdentifier(packagePb.getRepository(), new PathFragment(packagePb.getName())));
    } catch (SyntaxException e) {
      throw new PackageDeserializationException(e);
    }
    StoredEventHandler eventHandler = new StoredEventHandler();
    deserializeInternal(packagePb, eventHandler, builder, in);
    builder.addEvents(eventHandler.getEvents());
    return builder.build();
  }

  private static void deserializeEvent(
      DeserializationContext context, StoredEventHandler eventHandler, Build.Event event) {
    Location location = null;
    if (event.hasLocation()) {
      location = deserializeLocation(context, event.getLocation());
    }

    String message = event.getMessage();
    switch (event.getKind()) {
      case ERROR: eventHandler.handle(Event.error(location, message)); break;
      case WARNING: eventHandler.handle(Event.warn(location, message)); break;
      case INFO: eventHandler.handle(Event.info(location, message)); break;
      case PROGRESS: eventHandler.handle(Event.progress(location, message)); break;
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
  private static Object deserializeAttributeValue(Type<?> expectedType,
      Build.Attribute attrPb)
      throws PackageDeserializationException {
    switch (attrPb.getType()) {
      case INTEGER:
        return attrPb.hasIntValue() ? new Integer(attrPb.getIntValue()) : null;

      case STRING:
        if (!attrPb.hasStringValue()) {
          return null;
        } else if (expectedType == Type.NODEP_LABEL) {
          return deserializeLabel(attrPb.getStringValue());
        } else {
          return attrPb.getStringValue();
        }

      case LABEL:
      case OUTPUT:
        return attrPb.hasStringValue() ? deserializeLabel(attrPb.getStringValue()) : null;

      case STRING_LIST:
        if (expectedType == Type.NODEP_LABEL_LIST) {
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
}
