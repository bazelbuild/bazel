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
public abstract class AbstractClassPattern extends AbstractPattern {

    private static String check(String patternText) {
        if (patternText.indexOf('/') >= 0)
            throw new IllegalArgumentException("Class patterns cannot contain slashes");
        return patternText.replace('.', '/');
    }

    public AbstractClassPattern(@Nonnull String patternText) {
        super(check(patternText));
    }

}
