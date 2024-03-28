package com.google.devtools.build.lib.packages;

/**
 * A wrapper around a {@link com.google.devtools.build.lib.packages.Package.Builder} that only
 * allows access to methods that are safe to call from a rule initializer.
 */
public final class InitializerStarlarkContext extends TargetDefinitionContext {
  private final Package.Builder pkgBuilder;

  public InitializerStarlarkContext(Package.Builder pkgBuilder) {
    // Objects created in initializers do not use reference equality.
    super(Phase.INITIALIZER, new SymbolGenerator<>(new Object()));
    this.pkgBuilder = pkgBuilder;
  }

  @Override
  public LabelConverter getLabelConverter() {
    return pkgBuilder.getLabelConverter();
  }
}
