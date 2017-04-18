/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.transform.jar;

import com.tonicsystems.jarjar.transform.Transformable;
import java.io.IOException;
import javax.annotation.Nonnull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author shevek
 */
public abstract class AbstractFilterJarProcessor implements JarProcessor {

    private static final Logger LOG = LoggerFactory.getLogger(AbstractFilterJarProcessor.class);

    protected abstract boolean isFiltered(@Nonnull String name);

    protected boolean isVerbose() {
        return true;
    }

    @Override
    public Result scan(Transformable struct) throws IOException {
        if (isFiltered(struct.name)) {
            if (isVerbose())
                LOG.debug("{}.scan discarded {}", getClass().getSimpleName(), struct.name);
            return Result.DISCARD;
        }
        return Result.KEEP;
    }

    @Override
    public Result process(Transformable struct) throws IOException {
        if (isFiltered(struct.name)) {
            if (isVerbose())
                LOG.debug("{}.process discarded {}", getClass().getSimpleName(), struct.name);
            return Result.DISCARD;
        }
        return Result.KEEP;
    }
}
