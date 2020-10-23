// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.emptyToNull;

import com.android.SdkConstants;
import com.android.annotations.VisibleForTesting;
import com.android.ide.common.resources.configuration.FolderConfiguration;
import com.android.ide.common.resources.configuration.ResourceQualifier;
import com.android.resources.ResourceFolderType;
import com.android.resources.ResourceType;
import com.google.common.base.Ascii;
import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.xml.ResourcesAttribute;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.CheckReturnValue;
import javax.annotation.concurrent.Immutable;

/**
 * Represents a fully qualified name for an android resource.
 *
 * <p>Each resource name consists of the resource package, name, type, and qualifiers.
 */
// TODO(b/146498565): remove and/or replace.  For normal resources this can just be a ResourceName
// with Configuration, and the latter can come from aapt2 directly.  tools:* attributes should be a
// separate DataKey; as FullyQualifiedName they all have the same package, same singleton
// VirtualType, and same empty qualifiers.
@Immutable
public class FullyQualifiedName implements DataKey {
  public static final String DEFAULT_PACKAGE = "res-auto";
  private static final Joiner DASH_JOINER = Joiner.on('-');
  // To save on memory, always return one instance for each FullyQualifiedName.
  // Using a HashMap to deduplicate the instances -- the key retrieves a single instance.
  private static final ConcurrentMap<FullyQualifiedName, FullyQualifiedName> instanceCache =
      new ConcurrentHashMap<>();

  private static final AtomicInteger cacheHit = new AtomicInteger(0);
  private final String pkg;
  // TODO(b/146498565): use com.android.aapt.ConfigurationOuterClass.Configuration.
  private final ImmutableList<String> qualifiers;
  private final Type type;
  private final String name;

  private FullyQualifiedName(String pkg, ImmutableList<String> qualifiers, Type type, String name) {
    Preconditions.checkArgument(!pkg.isEmpty());
    this.pkg = pkg;
    this.qualifiers = qualifiers;
    this.type = type;
    this.name = name;
  }

  private static Type createTypeFrom(String rawType) {
    ResourceType resourceType = ResourceType.getEnum(rawType);
    VirtualType virtualType = VirtualType.getEnum(rawType);
    if (resourceType != null) {
      return new ResourceTypeWrapper(resourceType);
    } else if (virtualType != null) {
      return virtualType;
    }
    return null;
  }

  /**
   * Creates a new FullyQualifiedName with normalized qualifiers.
   *
   * @param rawPkg The resource package of the name. If unknown the default should be "res-auto"
   * @param qualifiers The resource qualifiers of the name, such as "en" or "xhdpi".
   * @param type The type of the name.
   * @param name The name of the name.
   * @return A new FullyQualifiedName.
   */
  public static FullyQualifiedName of(
      String rawPkg, List<String> qualifiers, Type type, String name) {
    checkNotNull(rawPkg);
    checkNotNull(qualifiers);
    checkNotNull(type);
    checkNotNull(name);
    ImmutableList<String> immutableQualifiers = ImmutableList.copyOf(qualifiers);
    String pkg = rawPkg.isEmpty() ? DEFAULT_PACKAGE : rawPkg;
    // TODO(corysmith): Address the GC thrash this creates by managing a simplified, mutable key to
    // do the instance check.
    FullyQualifiedName fqn = new FullyQualifiedName(pkg, immutableQualifiers, type, name);
    // Use putIfAbsent to get the canonical instance, if there. If it isn't, putIfAbsent will
    // return null, and we should return the current instance.
    FullyQualifiedName cached = instanceCache.putIfAbsent(fqn, fqn);
    if (cached == null) {
      return fqn;
    } else {
      cacheHit.incrementAndGet();
      return cached;
    }
  }

  /**
   * Creates a new FullyQualifiedName with normalized qualifiers.
   *
   * @param pkg The resource package of the name. If unknown the default should be "res-auto"
   * @param qualifiers The resource qualifiers of the name, such as "en" or "xhdpi".
   * @param type The resource type of the name.
   * @param name The name of the name.
   * @return A new FullyQualifiedName.
   */
  static FullyQualifiedName of(
      String pkg, List<String> qualifiers, ResourceType type, String name) {
    return of(pkg, qualifiers, new ResourceTypeWrapper(type), name);
  }

  public static FullyQualifiedName fromProto(SerializeFormat.DataKey protoKey) {
    return of(
        protoKey.getKeyPackage(),
        protoKey.getQualifiersList(),
        createTypeFrom(protoKey.getResourceType()),
        protoKey.getKeyValue());
  }

  // Note that while "$" is not allowed in source code, it's used in generated names (precisely to
  // avoid colliding with source code).
  static final Pattern QUALIFIED_REFERENCE =
      Pattern.compile("^((?<package>[^:]+):)?(?<type>\\w+)/(?<name>[A-Za-z0-9_.$]+)$");

  public static FullyQualifiedName fromReference(
      String qualifiedReference, Optional<String> packageName) {
    final Matcher matcher = QUALIFIED_REFERENCE.matcher(qualifiedReference);
    Preconditions.checkArgument(
        matcher.matches(),
        "%s is not a reference. Expected %s",
        qualifiedReference,
        QUALIFIED_REFERENCE.pattern());
    return of(
        Optional.ofNullable(emptyToNull(matcher.group("package")))
            .orElse(packageName.orElse(DEFAULT_PACKAGE)),
        ImmutableList.of(),
        ResourceType.getEnum(matcher.group("type")),
        matcher.group("name"));
  }

  public static void logCacheUsage(Logger logger) {
    logger.fine(
        String.format(
            "Total FullyQualifiedName instance cache hits %s out of %s",
            cacheHit.intValue(), instanceCache.size()));
  }

  /**
   * Returns a string path representation of the FullyQualifiedName.
   *
   * <p>Non-values Android Resource have a well defined file layout: From the resource directory,
   * they reside in &lt;resource type&gt;[-&lt;qualifier&gt;]/&lt;resource name&gt;[.extension]
   *
   * @param source The original source of the file-based resource's FullyQualifiedName
   * @return A string representation of the FullyQualifiedName with the provided extension.
   */
  public String toPathString(Path source) {
    String sourceExtension = FullyQualifiedName.Factory.getSourceExtension(source);
    return Paths.get(
            DASH_JOINER.join(
                ImmutableList.<String>builder().add(type.getName()).addAll(qualifiers).build()),
            name + sourceExtension)
        .toString();
  }

  @Override
  public String toPrettyString() {
    // TODO(corysmith): Add package when we start tracking it.
    return String.format(
        "%s%s/%s",
        DEFAULT_PACKAGE.equals(pkg) ? "" : pkg + ':',
        DASH_JOINER.join(
            ImmutableList.<String>builder().add(type.getName()).addAll(qualifiers).build()),
        name);
  }

  /**
   * Returns the string path representation of the values directory and qualifiers.
   *
   * <p>Certain resource types live in the "values" directory. This will calculate the directory and
   * ensure the qualifiers are represented.
   */
  // TODO(corysmith): Combine this with toPathString to clean up the interface of FullyQualifiedName
  // logically, the FullyQualifiedName should just be able to provide the relative path string for
  // the resource.
  public String valuesPath() {
    return Paths.get(
            DASH_JOINER.join(
                ImmutableList.<String>builder().add("values").addAll(qualifiers).build()),
            "values.xml")
        .toString();
  }

  @VisibleForTesting
  public String asUnqualifedName() {
    return String.format(
        "%s/%s",
        DASH_JOINER.join(
            ImmutableList.<String>builder().add(type.getName()).addAll(qualifiers).build()),
        name);
  }

  public String name() {
    return name;
  }

  public boolean isInPackage(String packageName) {
    return pkg.equals(packageName);
  }

  /** Provides the name qualified by the package it belongs to. */
  public String qualifiedName() {
    return (pkg.equals(DEFAULT_PACKAGE) ? "" : pkg + ":") + name;
  }

  public String asQualifiedReference() {
    return String.format(
        "%s%s/%s", (pkg.equals(DEFAULT_PACKAGE) ? "" : pkg + ":"), type.getName(), name);
  }

  public ResourceType type() {
    if (type instanceof ResourceTypeWrapper) {
      return ((ResourceTypeWrapper) type).resourceType;
    }
    return null;
  }

  public boolean isOverwritable() {
    return type.isOverwritable(this);
  }

  /** Creates a FullyQualifiedName from this one with a different package. */
  @CheckReturnValue
  public FullyQualifiedName replacePackage(String newPackage) {
    if (pkg.equals(newPackage)) {
      return this;
    }
    return of(newPackage, qualifiers, type, name);
  }

  @Override
  public int hashCode() {
    return Objects.hash(pkg, qualifiers, type, name);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof FullyQualifiedName)) {
      return false;
    }

    FullyQualifiedName other = getClass().cast(obj);
    return Objects.equals(pkg, other.pkg)
        && Objects.equals(type, other.type)
        && Objects.equals(name, other.name)
        && Objects.equals(qualifiers, other.qualifiers);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("pkg", pkg)
        .add("qualifiers", qualifiers)
        .add("type", type)
        .add("name", name)
        .toString();
  }

  @Override
  public int compareTo(DataKey otherKey) {
    if (!(otherKey instanceof FullyQualifiedName)) {
      return getKeyType().compareTo(otherKey.getKeyType());
    }
    FullyQualifiedName other = (FullyQualifiedName) otherKey;
    if (!pkg.equals(other.pkg)) {
      return pkg.compareTo(other.pkg);
    }
    if (!type.equals(other.type)) {
      return type.compareTo(other.type);
    }
    if (!name.equals(other.name)) {
      return name.compareTo(other.name);
    }
    if (!qualifiers.equals(other.qualifiers)) {
      if (qualifiers.size() != other.qualifiers.size()) {
        return qualifiers.size() - other.qualifiers.size();
      }
      // This works because the qualifiers are always in an ordered sequence.
      return qualifiers.toString().compareTo(other.qualifiers.toString());
    }
    return 0;
  }

  @Override
  public KeyType getKeyType() {
    return KeyType.FULL_QUALIFIED_NAME;
  }

  @Override
  public boolean shouldDetectConflicts() {
    // Ignore conflicts among pseudolocales.
    return qualifiers.stream()
        .noneMatch(q -> Ascii.equalsIgnoreCase(q, "en-rXA") || Ascii.equalsIgnoreCase(q, "ar-rXB"));
  }

  @Override
  public void serializeTo(OutputStream out, int valueSize) throws IOException {
    toSerializedBuilder().setValueSize(valueSize).build().writeDelimitedTo(out);
  }

  public SerializeFormat.DataKey.Builder toSerializedBuilder() {
    return SerializeFormat.DataKey.newBuilder()
        .setKeyPackage(pkg)
        .setResourceType(type.getName())
        .addAllQualifiers(qualifiers)
        .setKeyValue(name);
  }

  /** The non-resource {@link Type}s of a {@link FullyQualifiedName}. */
  public enum VirtualType implements Type {
    RESOURCES_ATTRIBUTE("<resources>", "Resources Attribute");

    private final String name;
    private final String displayName;

    private VirtualType(String name, String displayName) {
      this.name = name;
      this.displayName = displayName;
    }

    /** Returns the enum represented by the {@code name}. */
    public static VirtualType getEnum(String name) {
      for (VirtualType type : values()) {
        if (type.name.equals(name)) {
          return type;
        }
      }
      return null;
    }

    /** Returns an array with all the names defined by this enum. */
    public static String[] getNames() {
      VirtualType[] values = values();
      String[] names = new String[values.length];
      for (int i = values.length - 1; i >= 0; --i) {
        names[i] = values[i].getName();
      }
      return names;
    }

    /** Returns the resource type name. */
    @Override
    public String getName() {
      return name;
    }

    /** Returns a translated display name for the resource type. */
    public String getDisplayName() {
      return displayName;
    }

    @Override
    public ConcreteType getType() {
      return ConcreteType.VIRTUAL_TYPE;
    }

    @Override
    public boolean isOverwritable(FullyQualifiedName fqn) {
      if (this == RESOURCES_ATTRIBUTE) {
        return !ResourcesAttribute.AttributeType.from(fqn.name()).isCombining();
      }
      return true;
    }

    @Override
    public int compareTo(Type other) {
      if (!(other instanceof VirtualType)) {
        return getType().compareTo(other.getType());
      }
      return compareTo(((VirtualType) other));
    }

    @Override
    public String toString() {
      return getName();
    }
  }

  /** Represents the type of a {@link FullyQualifiedName}. */
  public interface Type {
    String getName();

    ConcreteType getType();

    boolean isOverwritable(FullyQualifiedName fqn);

    int compareTo(Type other);

    @Override
    boolean equals(Object obj);

    @Override
    int hashCode();

    @Override
    String toString();

    /**
     * The category of type that a {@link Type} can be.
     *
     * <p><em>Note:</em> used for strict ordering of {@link FullyQualifiedName}s.
     */
    enum ConcreteType {
      RESOURCE_TYPE,
      VIRTUAL_TYPE;
    }
  }

  private static class ResourceTypeWrapper implements Type {
    private final ResourceType resourceType;

    public ResourceTypeWrapper(ResourceType resourceType) {
      this.resourceType = resourceType;
    }

    @Override
    public String getName() {
      return resourceType.getName();
    }

    @Override
    public ConcreteType getType() {
      return ConcreteType.RESOURCE_TYPE;
    }

    @Override
    public boolean isOverwritable(FullyQualifiedName fqn) {
      return !(resourceType == ResourceType.ID
          || resourceType == ResourceType.PUBLIC
          || resourceType == ResourceType.STYLEABLE);
    }

    @Override
    public int compareTo(Type other) {
      if (!(other instanceof ResourceTypeWrapper)) {
        return getType().compareTo(other.getType());
      }
      return resourceType.compareTo(((ResourceTypeWrapper) other).resourceType);
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof ResourceTypeWrapper)) {
        return false;
      }
      ResourceTypeWrapper other = (ResourceTypeWrapper) obj;
      return Objects.equals(resourceType, other.resourceType);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(resourceType);
    }

    @Override
    public String toString() {
      return resourceType.toString();
    }
  }

  /** Represents the configuration qualifiers in a resource directory. */
  public static class Qualifiers {

    private static final Qualifiers EMPTY_QUALIFIERS =
        new Qualifiers(null, ImmutableList.of(), false);

    // Qualifiers are reasonably expensive to create, so cache them on directory names.
    private static final ConcurrentMap<String, Qualifiers> qualifierCache =
        new ConcurrentHashMap<>();

    public static final String INVALID_QUALIFIERS = "%s contains invalid qualifiers.";
    private final ResourceFolderType folderType;
    private final ImmutableList<String> qualifiers;
    private boolean defaultLocale;

    private Qualifiers(
        ResourceFolderType folderType, ImmutableList<String> qualifiers, boolean defaultLocale) {
      this.folderType = folderType;
      this.qualifiers = qualifiers;
      this.defaultLocale = defaultLocale;
    }

    public static Qualifiers parseFrom(String directoryName) {
      return qualifierCache.computeIfAbsent(
          directoryName, d -> getQualifiers(Splitter.on(SdkConstants.RES_QUALIFIER_SEP).split(d)));
    }

    private static Qualifiers getQualifiers(String... dirNameAndQualifiers) {
      return getQualifiers(Arrays.asList(dirNameAndQualifiers));
    }

    private static Qualifiers getQualifiers(Iterable<String> dirNameAndQualifiers) {
      PeekingIterator<String> rawQualifiers =
          Iterators.peekingIterator(dirNameAndQualifiers.iterator());
      // Remove directory name
      final ResourceFolderType folderType = ResourceFolderType.getTypeByName(rawQualifiers.next());

      // If there is no folder type, there are no qualifiers to parse.
      if (folderType == null) {
        return EMPTY_QUALIFIERS;
      }

      List<String> handledQualifiers = new ArrayList<>();
      // Do some substitution of language/region qualifiers.
      while (rawQualifiers.hasNext()) {
        handledQualifiers.add(rawQualifiers.next());
      }
      // Create a configuration
      FolderConfiguration config = FolderConfiguration.getConfigFromQualifiers(handledQualifiers);
      // FolderConfiguration returns an unhelpful null when it considers the qualifiers to be
      // invalid.
      if (config == null) {
        throw new IllegalArgumentException(
            String.format(INVALID_QUALIFIERS, DASH_JOINER.join(dirNameAndQualifiers)));
      }
      config.normalize();

      ImmutableList.Builder<String> builder = ImmutableList.<String>builder();
      // index 3 is past the country code, network code, and locale indices.
      for (int i = 0; i < FolderConfiguration.getQualifierCount(); ++i) {
        addIfNotNull(config.getQualifier(i), builder);
      }
      return new Qualifiers(folderType, builder.build(), config.getLocaleQualifier() == null);
    }

    private static void addIfNotNull(
        ResourceQualifier qualifier, ImmutableList.Builder<String> builder) {
      if (qualifier != null) {
        builder.add(qualifier.getFolderSegment());
      }
    }

    /** Returns the qualifiers as a list of strings. */
    public List<String> asList() {
      return qualifiers;
    }

    public ResourceFolderType asFolderType() {
      return folderType;
    }

    /** Creates a Qualifiers assuming that they are in the values directory. */
    @VisibleForTesting
    public static Qualifiers forValuesFolderFrom(List<String> qualifiers) {
      return Qualifiers.getQualifiers(
          ImmutableList.builder().add("values").addAll(qualifiers).build().toArray(new String[0]));
    }

    public boolean containDefaultLocale() {
      return defaultLocale;
    }
  }

  /** A factory for parsing an generating FullyQualified names with qualifiers and package. */
  public static class Factory {

    public static final String INVALID_QUALIFIED_NAME_MESSAGE_NO_MATCH =
        String.format(
            "%%s is not a valid qualified name. "
                + "It should be in the pattern [package:]{%s}/name",
            Joiner.on(",")
                .join(
                    ImmutableList.<String>builder()
                        .add(ResourceType.getNames())
                        .add(VirtualType.getNames())
                        .build()));
    public static final String INVALID_QUALIFIED_NAME_MESSAGE_NO_TYPE_OR_NAME =
        String.format(
            "Could not find either resource type (%%s) or name (%%s) in %%s. "
                + "It should be in the pattern [package:]{%s}/name",
            Joiner.on(",")
                .join(
                    ImmutableList.<String>builder()
                        .add(ResourceType.getNames())
                        .add(VirtualType.getNames())
                        .build()));
    private static final Pattern PARSING_REGEX =
        Pattern.compile(
            "(?:(?<package>[^:]+):){0,1}(?<type>[^-/]+)(?:[^/]*)/(?:(?:(?<namespace>\\{[^}]+\\}))"
                + "|(?:(?<misplacedPackage>[^:]+):)){0,1}(?<name>.+)");
    // private final ImmutableList<String> qualifiers;

    private final String pkg;

    private final Qualifiers qs;

    private Factory(Qualifiers qualifiers, String pkg) {
      // this.qualifiers = qualifiers;
      this.pkg = pkg;
      this.qs = qualifiers;
    }

    /** Creates a factory with default package from a directory name split on '-'. */
    @VisibleForTesting
    public static Factory fromDirectoryName(String... dirNameAndQualifiers) {
      return using(Qualifiers.getQualifiers(dirNameAndQualifiers), DEFAULT_PACKAGE);
    }

    /** Creates a factory with default package from a directory with '-' separating qualifiers. */
    public static Factory fromDirectoryName(String dirNameAndQualifiers) {
      return using(Qualifiers.parseFrom(dirNameAndQualifiers), DEFAULT_PACKAGE);
    }

    @VisibleForTesting
    public static Factory from(List<String> qualifiers, String pkg) {
      return using(Qualifiers.forValuesFolderFrom(qualifiers), pkg);
    }

    @VisibleForTesting
    public static Factory from(List<String> qualifiers) {
      return from(qualifiers, DEFAULT_PACKAGE);
    }

    /** Creates a factory with the qualifiers and package. */
    public static Factory using(Qualifiers qualifiers) {
      return using(qualifiers, DEFAULT_PACKAGE);
    }

    /** Creates a factory with the qualifiers and package. */
    public static Factory using(Qualifiers qualifiers, String pkg) {
      return new Factory(qualifiers, pkg.isEmpty() ? DEFAULT_PACKAGE : pkg);
    }

    private static String deriveRawFullyQualifiedName(Path source) {
      if (source.getNameCount() < 2) {
        throw new IllegalArgumentException(
            String.format(
                "The resource path %s is too short. "
                    + "The path is expected to be <resource type>/<file name>.",
                source));
      }
      // Compose the `pathWithExtension` manually to ensure it uses a forward slash.
      // Using Path.subpath would return a backslash-using path on Windows.
      String pathWithExtension = source.getParent().getFileName() + "/" + source.getFileName();
      int extensionStart = pathWithExtension.indexOf('.');
      if (extensionStart > 0) {
        return pathWithExtension.substring(0, extensionStart);
      }
      return pathWithExtension;
    }

    // Grabs the extension portion of the path removed by deriveRawFullyQualifiedName.
    private static String getSourceExtension(Path source) {
      // TODO(corysmith): Find out if there is a filename parser utility.
      String fileName = source.getFileName().toString();
      int extensionStart = fileName.indexOf('.');
      if (extensionStart > 0) {
        return fileName.substring(extensionStart);
      }
      return "";
    }

    public FullyQualifiedName create(Type type, String name, String pkg) {
      return FullyQualifiedName.of(pkg, qs.asList(), type, name);
    }

    public FullyQualifiedName create(ResourceType type, String name) {
      return create(new ResourceTypeWrapper(type), name, pkg);
    }

    public FullyQualifiedName create(ResourceType type, String name, String pkg) {
      return create(new ResourceTypeWrapper(type), name, pkg);
    }

    public FullyQualifiedName create(VirtualType type, String name) {
      return create(type, name, pkg);
    }

    /**
     * Parses a FullyQualifiedName from a string.
     *
     * @param raw A string in the expected format from
     *     [&lt;package&gt;:]&lt;ResourceType.name&gt;/&lt;resource name&gt;.
     * @throws IllegalArgumentException when the raw string is not valid qualified name.
     */
    public FullyQualifiedName parse(String raw) {
      Matcher matcher = PARSING_REGEX.matcher(raw);
      if (!matcher.matches()) {
        throw new IllegalArgumentException(
            String.format(INVALID_QUALIFIED_NAME_MESSAGE_NO_MATCH, raw));
      }
      String parsedPackage =
          firstNonNull(matcher.group("package"), matcher.group("misplacedPackage"), pkg);

      Type type = createTypeFrom(matcher.group("type"));
      String name =
          matcher.group("namespace") != null
              ? matcher.group("namespace") + matcher.group("name")
              : matcher.group("name");

      if (type == null || name == null) {
        throw new IllegalArgumentException(
            String.format(INVALID_QUALIFIED_NAME_MESSAGE_NO_TYPE_OR_NAME, type, name, raw));
      }

      return FullyQualifiedName.of(parsedPackage, qs.asList(), type, name);
    }

    private String firstNonNull(String... values) {
      for (String value : values) {
        if (value != null) {
          return value;
        }
      }
      throw new NullPointerException("Expected a nonnull value.");
    }

    /**
     * Generates a FullyQualifiedName for a file-based resource given the source Path.
     *
     * @param sourcePath the path of the file-based resource.
     * @throws IllegalArgumentException if the file-based resource has an invalid filename
     */
    public FullyQualifiedName parse(Path sourcePath) {
      return parse(deriveRawFullyQualifiedName(sourcePath));
    }
  }
}
