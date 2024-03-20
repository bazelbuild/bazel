package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.packages.TargetDefinitionContext;

import java.util.Optional;

/**
 * A wrapper around a {@link com.google.devtools.build.lib.packages.Package.Builder} that only
 * allows access to methods that are safe to call from a rule initializer.
 */
public final class InitializerStarlarkContext extends TargetDefinitionContext {
  private final Package.Builder pkgBuilder;

  public InitializerStarlarkContext(Package.Builder pkgBuilder) {
    super(Phase.INITIALIZER, new SymbolGenerator<>(new Object()));
    this.pkgBuilder = pkgBuilder;
  }

  @Override
  PackageIdentifier getPackageIdentifier() {
    return pkgBuilder.getPackageIdentifier();
  }

  @Override
  Optional<String> getAssociatedModuleName() {
    return pkgBuilder.getAssociatedModuleName();
  }

  @Override
  Optional<String> getAssociatedModuleVersion() {
    return pkgBuilder.getAssociatedModuleVersion();
  }

  @Override
  public LabelConverter getLabelConverter() {
    return pkgBuilder.getLabelConverter();
  }
}
