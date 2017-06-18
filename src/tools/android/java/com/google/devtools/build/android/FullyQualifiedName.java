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

import com.android.ide.common.resources.configuration.FolderConfiguration;
import com.android.ide.common.resources.configuration.ResourceQualifier;
import com.android.resources.ResourceType;
import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.xml.ResourcesAttribute;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
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
 * Each resource name consists of the resource package, name, type, and qualifiers.
 */
@Immutable
public class FullyQualifiedName implements DataKey {
  /** Represents the type of a {@link FullyQualifiedName}. */
  public interface Type {
    /**
     * The category of type that a {@link Type} can be.
     *
     * <p>
     * <em>Note:</em> used for strict ordering of {@link FullyQualifiedName}s.
     */
    public enum ConcreteType {
      RESOURCE_TYPE,
      VIRTUAL_TYPE;
    }

    public String getName();
    public ConcreteType getType();
    public boolean isOverwritable(FullyQualifiedName fqn);
    public int compareTo(Type other);
    @Override public boolean equals(Object obj);
    @Override public int hashCode();
    @Override public String toString();
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

  /** The non-resource {@link Type}s of a {@link FullyQualifiedName}. */
  public enum VirtualType implements Type {
    RESOURCES_ATTRIBUTE("<resources>", "Resources Attribute");

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

    private final String name;
    private final String displayName;

    private VirtualType(String name, String displayName) {
      this.name = name;
      this.displayName = displayName;
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

  public static final String DEFAULT_PACKAGE = "res-auto";
  private static final Joiner DASH_JOINER = Joiner.on('-');

  // To save on memory, always return one instance for each FullyQualifiedName.
  // Using a HashMap to deduplicate the instances -- the key retrieves a single instance.
  private static final ConcurrentMap<FullyQualifiedName, FullyQualifiedName> instanceCache =
      new ConcurrentHashMap<>();
  private static final AtomicInteger cacheHit = new AtomicInteger(0);

  /**
   * A factory for parsing an generating FullyQualified names with qualifiers and package.
   */
  public static class Factory {

    private static final Pattern PARSING_REGEX =
        Pattern.compile("(?:(?<package>[^:]+):){0,1}(?<type>[^-/]+)(?:[^/]*)/(?<name>.+)");
    public static final String INVALID_QUALIFIED_NAME_MESSAGE_NO_MATCH =
        String.format(
            "%%s is not a valid qualified name. "
                + "It should be in the pattern [package:]{%s}/name",
            Joiner.on(",").join(ImmutableList.<String>builder()
                .add(ResourceType.getNames())
                .add(VirtualType.getNames())
                .build()));
    public static final String INVALID_QUALIFIED_NAME_MESSAGE_NO_TYPE_OR_NAME =
        String.format(
            "Could not find either resource type (%%s) or name (%%s) in %%s. "
                + "It should be in the pattern [package:]{%s}/name",
            Joiner.on(",").join(ImmutableList.<String>builder()
                .add(ResourceType.getNames())
                .add(VirtualType.getNames())
                .build()));
    public static final String INVALID_QUALIFIERS = "%s contains invalid qualifiers.";
    private final ImmutableList<String> qualifiers;
    private final String pkg;

    private Factory(ImmutableList<String> qualifiers, String pkg) {
      this.qualifiers = qualifiers;
      this.pkg = pkg;
    }

    /** Creates a factory with default package from a directory name split on '-'. */
    public static Factory fromDirectoryName(String[] dirNameAndQualifiers) {
      return from(getQualifiers(dirNameAndQualifiers));
    }

    private static List<String> getQualifiers(String[] dirNameAndQualifiers) {
      PeekingIterator<String> rawQualifiers =
          Iterators.peekingIterator(Iterators.forArray(dirNameAndQualifiers));
      // Remove directory name
      rawQualifiers.next();
      List<String> transformedLocaleQualifiers = new ArrayList<>();
      List<String> handledQualifiers = new ArrayList<>();
      // Do some substitution of language/region qualifiers.
      while (rawQualifiers.hasNext()) {
        String qualifier = rawQualifiers.next();
        if ("es".equalsIgnoreCase(qualifier)
            && rawQualifiers.hasNext()
            && "419".equalsIgnoreCase(rawQualifiers.peek())) {
          // Replace the es-419.
          transformedLocaleQualifiers.add("b+es+419");
          // Consume the next value, as it's been replaced.
          rawQualifiers.next();
        } else if ("sr".equalsIgnoreCase(qualifier)
            && rawQualifiers.hasNext()
            && "rlatn".equalsIgnoreCase(rawQualifiers.peek())) {
          // Replace the sr-rLatn.
          transformedLocaleQualifiers.add("b+sr+Latn");
          // Consume the next value, as it's been replaced.
          rawQualifiers.next();
        } else {
          // This qualifier can probably be handled by FolderConfiguration.
          handledQualifiers.add(qualifier);
        }
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

      // This is fragile but better than the Gradle scheme of just dropping
      // entire subtrees.
      Builder<String> builder = ImmutableList.<String>builder();
      addIfNotNull(config.getCountryCodeQualifier(), builder);
      addIfNotNull(config.getNetworkCodeQualifier(), builder);
      if (transformedLocaleQualifiers.isEmpty()) {
        addIfNotNull(config.getLocaleQualifier(), builder);
      } else {
        builder.addAll(transformedLocaleQualifiers);
      }
      // index 3 is past the country code, network code, and locale indices.
      for (int i = 3; i < FolderConfiguration.getQualifierCount(); ++i) {
        addIfNotNull(config.getQualifier(i), builder);
      }
      return builder.build();
    }

    private static void addIfNotNull(
        ResourceQualifier qualifier, ImmutableList.Builder<String> builder) {
      if (qualifier != null) {
        builder.add(qualifier.getFolderSegment());
      }
    }

    public static Factory from(List<String> qualifiers, String pkg) {
      return new Factory(ImmutableList.copyOf(qualifiers), pkg);
    }

    public static Factory from(List<String> qualifiers) {
      return from(ImmutableList.copyOf(qualifiers), DEFAULT_PACKAGE);
    }

    public FullyQualifiedName create(Type type, String name, String pkg) {
      return FullyQualifiedName.of(pkg, qualifiers, type, name);
    }

    public FullyQualifiedName create(ResourceType type, String name) {
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
      String parsedPackage = matcher.group("package");
      Type type = createTypeFrom(matcher.group("type"));
      String name = matcher.group("name");

      if (type == null || name == null) {
        throw new IllegalArgumentException(
            String.format(
                INVALID_QUALIFIED_NAME_MESSAGE_NO_TYPE_OR_NAME, type, name, raw));
      }
      return FullyQualifiedName.of(
          parsedPackage == null ? pkg : parsedPackage, qualifiers, type, name);
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

    private static String deriveRawFullyQualifiedName(Path source) {
      if (source.getNameCount() < 2) {
        throw new IllegalArgumentException(
            String.format(
                "The resource path %s is too short. "
                    + "The path is expected to be <resource type>/<file name>.",
                source));
      }
      String pathWithExtension =
          source.subpath(source.getNameCount() - 2, source.getNameCount()).toString();
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
  }

  /**
   * Creates a new FullyQualifiedName with normalized qualifiers.
   *
   * @param pkg The resource package of the name. If unknown the default should be "res-auto"
   * @param qualifiers The resource qualifiers of the name, such as "en" or "xhdpi".
   * @param type The type of the name.
   * @param name The name of the name.
   * @return A new FullyQualifiedName.
   */
  public static FullyQualifiedName of(
      String pkg, List<String> qualifiers, Type type, String name) {
    checkNotNull(pkg);
    checkNotNull(qualifiers);
    checkNotNull(type);
    checkNotNull(name);
    ImmutableList<String> immutableQualifiers = ImmutableList.copyOf(qualifiers);
    // TODO(corysmith): Address the GC thrash this creates by managing a simplified, mutable key to
    // do the instance check.
    FullyQualifiedName fqn =
        new FullyQualifiedName(pkg, immutableQualifiers, type, name);
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

  public static void logCacheUsage(Logger logger) {
    logger.fine(
        String.format(
            "Total FullyQualifiedName instance cache hits %s out of %s",
            cacheHit.intValue(),
            instanceCache.size()));
  }

  private final String pkg;
  private final ImmutableList<String> qualifiers;
  private final Type type;
  private final String name;

  private FullyQualifiedName(
      String pkg,
      ImmutableList<String> qualifiers,
      Type type,
      String name) {
    this.pkg = pkg;
    this.qualifiers = qualifiers;
    this.type = type;
    this.name = name;
  }

  /**
   * Returns a string path representation of the FullyQualifiedName.
   *
   * Non-values Android Resource have a well defined file layout: From the resource directory, they
   * reside in &lt;resource type&gt;[-&lt;qualifier&gt;]/&lt;resource name&gt;[.extension]
   *
   * @param source The original source of the file-based resource's FullyQualifiedName
   * @return A string representation of the FullyQualifiedName with the provided extension.
   */
  public String toPathString(Path source) {
    String sourceExtension = FullyQualifiedName.Factory.getSourceExtension(source);
    return Paths.get(
            DASH_JOINER.join(
                ImmutableList.<String>builder()
                    .add(type.getName())
                    .addAll(qualifiers)
                    .build()),
            name + sourceExtension)
        .toString();
  }

  @Override
  public String toPrettyString() {
    // TODO(corysmith): Add package when we start tracking it.
    return String.format(
        "%s/%s",
        DASH_JOINER.join(
            ImmutableList.<String>builder().add(type.getName()).addAll(qualifiers).build()),
        name);
  }

  /**
   * Returns the string path representation of the values directory and qualifiers.
   *
   * Certain resource types live in the "values" directory. This will calculate the directory and
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

  public String name() {
    return name;
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
}
