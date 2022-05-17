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

"""license_policy rule.

A license_policy names a set of conditions allowed in the union of all
license_kinds use by a target. The name of the rule is typically an
application type (e.g. production_server, mobile_application, ...)

"""

load("@rules_license//rules:license_policy_provider.bzl", "LicensePolicyInfo")

def _license_policy_impl(ctx):
    provider = LicensePolicyInfo(
        name = ctx.attr.name,
        label = "@%s//%s:%s" % (
            ctx.label.workspace_name,
            ctx.label.package,
            ctx.label.name,
        ),
        conditions = ctx.attr.conditions,
    )
    return [provider]

_license_policy = rule(
    implementation = _license_policy_impl,
    attrs = {
        "conditions": attr.string_list(
            doc = "Conditions to be met when using software under this license." +
                  "  Conditions are defined by the organization using this license.",
            mandatory = True,
        ),
    },
)

def license_policy(name, conditions):
    _license_policy(
        name = name,
        conditions = conditions,
        applicable_licenses = [],
    )
