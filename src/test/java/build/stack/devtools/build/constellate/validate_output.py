#!/usr/bin/env python3
"""Validation script for constellate output.

This script reads a ModuleInfo protobuf and validates:
- OriginKey presence and correctness
- Advertised providers for rules
- Provider init callbacks
- Provider schema field documentation
"""

import sys
import argparse
from pathlib import Path

# Add the proto path to Python path
proto_path = Path(__file__).parent.parent.parent.parent.parent.parent / "main" / "protobuf"
sys.path.insert(0, str(proto_path))

try:
    from stardoc_output_pb2 import ModuleInfo
except ImportError:
    print("ERROR: Could not import stardoc_output_pb2")
    print("You may need to generate Python protos first:")
    print("  bazel build //src/main/protobuf:stardoc_output_py_pb2")
    sys.exit(1)


class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def validate_origin_key(origin_key, entity_type, entity_name):
    """Validate that an OriginKey has required fields."""
    issues = []

    if not origin_key.name:
        issues.append(f"Missing origin_key.name for {entity_type} '{entity_name}'")

    if not origin_key.file:
        issues.append(f"Missing origin_key.file for {entity_type} '{entity_name}'")

    return issues


def validate_rule_info(rule_info):
    """Validate a RuleInfo message."""
    issues = []
    name = rule_info.rule_name or "(unnamed)"

    # Check OriginKey
    if rule_info.HasField('origin_key'):
        issues.extend(validate_origin_key(rule_info.origin_key, "rule", name))
    else:
        issues.append(f"Missing origin_key for rule '{name}'")

    # Check advertised providers (Phase 2 feature)
    if rule_info.HasField('advertised_providers'):
        providers = rule_info.advertised_providers
        if len(providers.provider_name) != len(providers.origin_key):
            issues.append(
                f"Rule '{name}': advertised_providers has {len(providers.provider_name)} "
                f"names but {len(providers.origin_key)} origin keys"
            )
        for i, provider_name in enumerate(providers.provider_name):
            if i < len(providers.origin_key):
                origin_key = providers.origin_key[i]
                if not origin_key.name and provider_name != "Unknown Provider":
                    issues.append(
                        f"Rule '{name}': advertised provider '{provider_name}' "
                        f"missing origin_key.name"
                    )

    return issues


def validate_provider_info(provider_info):
    """Validate a ProviderInfo message."""
    issues = []
    name = provider_info.provider_name or "(unnamed)"

    # Check OriginKey
    if provider_info.HasField('origin_key'):
        issues.extend(validate_origin_key(provider_info.origin_key, "provider", name))
    else:
        issues.append(f"Missing origin_key for provider '{name}'")

    # Check init callback (Phase 3 feature)
    if provider_info.HasField('init'):
        init_info = provider_info.init
        if not init_info.function_name:
            issues.append(f"Provider '{name}': init callback missing function_name")
        if not init_info.HasField('origin_key'):
            issues.append(f"Provider '{name}': init callback missing origin_key")

    # Check field documentation (Phase 3 feature)
    for field in provider_info.field_info:
        if not field.doc_string:
            issues.append(
                f"Provider '{name}': field '{field.name}' missing documentation"
            )

    return issues


def validate_function_info(function_info):
    """Validate a StarlarkFunctionInfo message."""
    issues = []
    name = function_info.function_name or "(unnamed)"

    # Check OriginKey
    if function_info.HasField('origin_key'):
        issues.extend(validate_origin_key(function_info.origin_key, "function", name))
    else:
        issues.append(f"Missing origin_key for function '{name}'")

    return issues


def validate_aspect_info(aspect_info):
    """Validate an AspectInfo message."""
    issues = []
    name = aspect_info.aspect_name or "(unnamed)"

    # Check OriginKey
    if aspect_info.HasField('origin_key'):
        issues.extend(validate_origin_key(aspect_info.origin_key, "aspect", name))
    else:
        issues.append(f"Missing origin_key for aspect '{name}'")

    return issues


def validate_macro_info(macro_info):
    """Validate a MacroInfo message."""
    issues = []
    name = macro_info.macro_name or "(unnamed)"

    # Check OriginKey
    if macro_info.HasField('origin_key'):
        issues.extend(validate_origin_key(macro_info.origin_key, "macro", name))
    else:
        issues.append(f"Missing origin_key for macro '{name}'")

    return issues


def validate_repository_rule_info(repo_rule_info):
    """Validate a RepositoryRuleInfo message."""
    issues = []
    name = repo_rule_info.rule_name or "(unnamed)"

    # Check OriginKey
    if repo_rule_info.HasField('origin_key'):
        issues.extend(validate_origin_key(repo_rule_info.origin_key, "repository_rule", name))
    else:
        issues.append(f"Missing origin_key for repository_rule '{name}'")

    return issues


def validate_module_extension_info(module_ext_info):
    """Validate a ModuleExtensionInfo message."""
    issues = []
    name = module_ext_info.extension_name or "(unnamed)"

    # Check OriginKey
    if module_ext_info.HasField('origin_key'):
        issues.extend(validate_origin_key(module_ext_info.origin_key, "module_extension", name))
    else:
        issues.append(f"Missing origin_key for module_extension '{name}'")

    return issues


def print_summary(module_info):
    """Print a summary of the ModuleInfo contents."""
    print(f"\n{Colors.BLUE}=== Module Summary ==={Colors.NC}")
    print(f"File: {module_info.file or '(not set)'}")
    print(f"Rules: {len(module_info.rule_info)}")
    print(f"Providers: {len(module_info.provider_info)}")
    print(f"Functions: {len(module_info.func_info)}")
    print(f"Aspects: {len(module_info.aspect_info)}")
    print(f"Macros: {len(module_info.macro_info)}")
    print(f"Repository Rules: {len(module_info.repository_rule_info)}")
    print(f"Module Extensions: {len(module_info.module_extension_info)}")

    # Detailed entity listing
    if module_info.rule_info:
        print(f"\n{Colors.BLUE}Rules:{Colors.NC}")
        for rule in module_info.rule_info:
            origin = f" [{rule.origin_key.file}]" if rule.HasField('origin_key') else ""
            providers = f" provides={len(rule.advertised_providers.provider_name)}" if rule.HasField('advertised_providers') else ""
            print(f"  - {rule.rule_name}{origin}{providers}")

    if module_info.provider_info:
        print(f"\n{Colors.BLUE}Providers:{Colors.NC}")
        for provider in module_info.provider_info:
            origin = f" [{provider.origin_key.file}]" if provider.HasField('origin_key') else ""
            init = " +init" if provider.HasField('init') else ""
            fields = f" fields={len(provider.field_info)}"
            print(f"  - {provider.provider_name}{origin}{init}{fields}")

    if module_info.func_info:
        print(f"\n{Colors.BLUE}Functions:{Colors.NC}")
        for func in module_info.func_info:
            origin = f" [{func.origin_key.file}]" if func.HasField('origin_key') else ""
            print(f"  - {func.function_name}{origin}")


def validate_module_info(module_info, verbose=False):
    """Validate a ModuleInfo protobuf message."""
    all_issues = []

    # Validate each entity type
    for rule_info in module_info.rule_info:
        all_issues.extend(validate_rule_info(rule_info))

    for provider_info in module_info.provider_info:
        all_issues.extend(validate_provider_info(provider_info))

    for function_info in module_info.func_info:
        all_issues.extend(validate_function_info(function_info))

    for aspect_info in module_info.aspect_info:
        all_issues.extend(validate_aspect_info(aspect_info))

    for macro_info in module_info.macro_info:
        all_issues.extend(validate_macro_info(macro_info))

    for repo_rule_info in module_info.repository_rule_info:
        all_issues.extend(validate_repository_rule_info(repo_rule_info))

    for module_ext_info in module_info.module_extension_info:
        all_issues.extend(validate_module_extension_info(module_ext_info))

    # Print results
    print_summary(module_info)

    print(f"\n{Colors.BLUE}=== Validation Results ==={Colors.NC}")
    if all_issues:
        print(f"{Colors.YELLOW}Found {len(all_issues)} issue(s):{Colors.NC}")
        for issue in all_issues:
            print(f"  {Colors.YELLOW}⚠{Colors.NC} {issue}")
        return False
    else:
        print(f"{Colors.GREEN}✓ All validations passed!{Colors.NC}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate constellate output')
    parser.add_argument('input_file', help='Path to ModuleInfo protobuf file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Read and parse the protobuf
    try:
        with open(args.input_file, 'rb') as f:
            module_info = ModuleInfo()
            module_info.ParseFromString(f.read())
    except Exception as e:
        print(f"{Colors.RED}ERROR: Failed to read protobuf: {e}{Colors.NC}")
        return 1

    # Validate
    success = validate_module_info(module_info, args.verbose)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
