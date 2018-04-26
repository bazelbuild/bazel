package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;


import javax.annotation.Nullable;

public class WorkspaceMappingsFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException, InterruptedException {
        SkyKey externalPackageKey = PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
        PackageValue externalPackageValue = (PackageValue) env.getValue(externalPackageKey);
        if (externalPackageValue == null) {
            return null;
        }
        Package externalPackage = externalPackageValue.getPackage();
        if (externalPackage.containsErrors()) {
            throw new WorkspaceMappingsFunctionException();
        }

        ImmutableMap<RepositoryName, RepositoryName> mappings =
            externalPackage.getWorkspaceMappings((RepositoryName) skyKey.argument());
        return WorkspaceMappingsValue.withMappings(mappings);
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
        return null;
    }

    private class WorkspaceMappingsFunctionException extends SkyFunctionException {
        WorkspaceMappingsFunctionException() {
            super(new BuildFileContainsErrorsException(Label.EXTERNAL_PACKAGE_IDENTIFIER),
                    Transience.PERSISTENT);
        }
    }
}
