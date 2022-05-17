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

"""Proof of concept. License restriction."""

load("@rules_license//rules:providers.bzl", "LicenseKindInfo")

#
# License Kind: The declaration of a well known category of license, for example,
# Apache, MIT, LGPL v2. An organization may also declare its own license kinds
# that it may user privately.
#

def _license_kind_impl(ctx):
    provider = LicenseKindInfo(
        name = ctx.attr.name,
        label = "@%s//%s:%s" % (
            ctx.label.workspace_name,
            ctx.label.package,
            ctx.label.name,
        ),
        conditions = ctx.attr.conditions,
    )
    return [provider]

_license_kind = rule(
    implementation = _license_kind_impl,
    attrs = {
        "canonical_text": attr.label(
            doc = "File containing the canonical text for this license. Must be UTF-8 encoded.",
            allow_single_file = True,
        ),
        "conditions": attr.string_list(
            doc = "Conditions to be met when using software under this license." +
                  "  Conditions are defined by the organization using this license.",
            mandatory = True,
        ),
        "url": attr.string(doc = "URL pointing to canonical license definition"),
    },
)

def license_kind(name, **kwargs):
    if "conditions" not in kwargs:
        kwargs["conditions"] = []
    _license_kind(
        name = name,
        applicable_licenses = [],
        **kwargs
    )
