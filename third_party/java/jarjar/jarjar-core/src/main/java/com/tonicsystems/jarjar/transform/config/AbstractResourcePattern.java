/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.transform.config;

import javax.annotation.Nonnull;

/**
 *
 * @author shevek
 */
public abstract class AbstractResourcePattern extends AbstractPattern {

    public AbstractResourcePattern(@Nonnull String patternText) {
        super(patternText);
    }

}
