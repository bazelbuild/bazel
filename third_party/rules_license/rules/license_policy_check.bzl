# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""License compliance checking at analysis time."""

load(
    "@rules_license//rules:gather_licenses_info.bzl",
    "gather_licenses_info",
)
load(
    "@rules_license//rules:license_policy_provider.bzl",
    "LicensePolicyInfo",
)
load(
    "@rules_license//rules:providers.bzl",
    "LicensesInfo",
)

def _license_policy_check_impl(ctx):
    policy = ctx.attr.policy[LicensePolicyInfo]
    allowed_conditions = policy.conditions
    if LicensesInfo in ctx.attr.target:
        for license in ctx.attr.target[LicensesInfo].licenses.to_list():
            for kind in license.license_kinds:
                # print(kind.conditions)
                for condition in kind.conditions:
                    if condition not in allowed_conditions:
                        fail("Condition %s violates policy %s" % (
                            condition,
                            policy.label,
                        ))
    return [DefaultInfo()]

_license_policy_check = rule(
    implementation = _license_policy_check_impl,
    doc = """Internal tmplementation method for license_policy_check().""",
    attrs = {
        "policy": attr.label(
            doc = """Policy definition.""",
            mandatory = True,
            providers = [LicensePolicyInfo],
        ),
        "target": attr.label(
            doc = """Target to collect LicenseInfo for.""",
            aspects = [gather_licenses_info],
            mandatory = True,
            allow_single_file = True,
        ),
    },
)

def license_policy_check(name, target, policy, **kwargs):
    """Checks a target against a policy.

    Args:
      name: The target.
      target: A target to test for compliance with a policy
      policy: A rule providing LicensePolicyInfo.
      **kwargs: other args.

    Usage:

      license_policy_check(
          name = "license_info",
          target = ":my_app",
          policy = "//my_org/compliance/policies:mobile_application",
      )
    """
    _license_policy_check(name = name, target = target, policy = policy, **kwargs)
